// 2D photo -> 3D Gaussian splats: estimates depth with a monocular depth model
// (Depth Anything V2 or MiDaS, ONNX Runtime on the GPU), then a custom CUDA kernel
// back-projects each pixel through a pinhole camera into 3D, drops the points that sit
// on depth edges and turns the rest into flat Gaussians lying on the surface.
// The result is a binary 3DGS-style PLY (see gaussians.h), rendered by splat_server.
//
// Usage: splat <image.jpg> [output.ply] [model.onnx] [max_points] [depth_range] [edge_threshold]
//   depth_range     far/near ratio of the scene (default 10); larger = more depth
//   edge_threshold  a point whose neighbour depth differs by more than this fraction
//                   (default 0.1) is moved back to the background; 0 turns this off

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "gaussians.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err = (call);                                                     \
        if (err != cudaSuccess) {                                                     \
            cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__   \
                 << ":" << __LINE__ << endl;                                          \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

// Models with a dynamic input size (Depth Anything V2) get the photo resized so its
// short side is this long, keeping the aspect ratio, rounded to a multiple of 14 (ViT patches)
const int DYNAMIC_SHORT_SIDE = 518;
const int PATCH = 14;

// Gaussian size along the surface, as a fraction of the spacing between neighbouring
// samples (neighbours overlap enough to leave no gaps), and the disc thickness
const float SPLAT_SIGMA = 0.6f;
const float SPLAT_THICKNESS = 0.1f;
const float SPLAT_OPACITY = 0.95f;
const float MAX_ANISOTROPY = 8.0f;

// Pinhole camera math to project pixel (x, y) at depth z back into 3D
__device__ float3 backProject(float x, float y, float z, int width, int height, float focal_length) {
    return make_float3((x - width / 2.0f) * z / focal_length, (y - height / 2.0f) * z / focal_length, z);
}

__device__ float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 scale3(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float3 normalize3(float3 a) { return scale3(a, rsqrtf(fmaxf(dot3(a, a), 1e-20f))); }

// Rotation matrix with columns c0 c1 c2 -> unit quaternion (w, x, y, z)
__device__ void matrixToQuaternion(float3 c0, float3 c1, float3 c2, float* q) {
    float trace = c0.x + c1.y + c2.z;
    float w, x, y, z;
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;
        w = 0.25f * s; x = (c1.z - c2.y) / s; y = (c2.x - c0.z) / s; z = (c0.y - c1.x) / s;
    } else if (c0.x > c1.y && c0.x > c2.z) {
        float s = sqrtf(1.0f + c0.x - c1.y - c2.z) * 2.0f;
        w = (c1.z - c2.y) / s; x = 0.25f * s; y = (c1.x + c0.y) / s; z = (c2.x + c0.z) / s;
    } else if (c1.y > c2.z) {
        float s = sqrtf(1.0f + c1.y - c0.x - c2.z) * 2.0f;
        w = (c2.x - c0.z) / s; x = (c1.x + c0.y) / s; y = 0.25f * s; z = (c2.y + c1.z) / s;
    } else {
        float s = sqrtf(1.0f + c2.z - c0.x - c1.y) * 2.0f;
        w = (c0.y - c1.x) / s; x = (c2.x + c0.z) / s; y = (c2.y + c1.z) / s; z = 0.25f * s;
    }
    float n = rsqrtf(w * w + x * x + y * y + z * z);
    q[0] = w * n; q[1] = x * n; q[2] = y * n; q[3] = z * n;
}

// --- CUSTOM CUDA KERNEL ---
// This kernel converts 2D pixels + Depth into 3D Gaussians. One thread per sampled
// pixel; every `stride`-th pixel is sampled so the file stays small enough to load quickly.
// Each pixel becomes a flat Gaussian disc lying on the surface: the neighbouring
// samples give two tangent vectors, their cross product the normal, and the tangent
// lengths the size, so neighbouring discs just overlap.
// The depth model blurs depth across object outlines, which would leave a "curtain"
// of splats floating between foreground and background. So a pixel whose depth differs
// from a neighbour `radius` pixels away by more than `edgeThreshold` (relative) is pushed
// back to the farthest of those depths, as a small disc facing the camera: it fills in the
// background behind the outline instead of hanging in between.
// Gaussians are packed to the front of `splats` with atomicAdd; counts[0] returns how
// many there are, counts[1] how many of them were on depth edges.
__global__ void generateGaussians(const unsigned char* color, const float* depth, Gaussian* splats, int* counts,
                                  int width, int height, int stride, int outWidth, int outHeight,
                                  int radius, float edgeThreshold, float focal_length) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outWidth || oy >= outHeight) return;

    int x = ox * stride;
    int y = oy * stride;
    int idx = y * width + x;
    float z_val = depth[idx];

    // Depth-discontinuity test against the 4 neighbours (clamped at the image border)
    bool onEdge = false;
    if (edgeThreshold > 0.0f) {
        float zmin = z_val, zmax = z_val;
        int nx[4] = {max(x - radius, 0), min(x + radius, width - 1), x, x};
        int ny[4] = {y, y, max(y - radius, 0), min(y + radius, height - 1)};
        for (int i = 0; i < 4; ++i) {
            float zn = depth[ny[i] * width + nx[i]];
            zmin = fminf(zmin, zn);
            zmax = fmaxf(zmax, zn);
        }
        if (zmax - zmin > edgeThreshold * zmin) {
            // The blur spans a few depth-model pixels, so look further out (up to 3x) for
            // the real background depth; otherwise the point would still hang mid-air
            onEdge = true;
            for (int k = 2; k <= 3; ++k) {
                int r = k * radius;
                zmax = fmaxf(zmax, depth[y * width + max(x - r, 0)]);
                zmax = fmaxf(zmax, depth[y * width + min(x + r, width - 1)]);
                zmax = fmaxf(zmax, depth[max(y - r, 0) * width + x]);
                zmax = fmaxf(zmax, depth[min(y + r, height - 1) * width + x]);
            }
            z_val = zmax;
        }
    }

    float3 p = backProject(x, y, z_val, width, height, focal_length);
    float footprint = stride * z_val / focal_length;
    float3 n, a1, a2;
    float s1, s2;

    if (onEdge) {
        // round disc facing the camera, one sample spacing wide
        n = scale3(normalize3(p), -1.0f);
        a1 = normalize3(cross3(make_float3(0.0f, 1.0f, 0.0f), n));
        a2 = cross3(n, a1);
        s1 = s2 = SPLAT_SIGMA * footprint;
    } else {
        // Tangents: 3D step to the next sample along x and along y (central difference,
        // one-sided at the image border)
        int xl = max(x - stride, 0), xr = min(x + stride, width - 1);
        int yu = max(y - stride, 0), yd = min(y + stride, height - 1);
        float3 tu = sub(backProject(xr, y, depth[y * width + xr], width, height, focal_length),
                        backProject(xl, y, depth[y * width + xl], width, height, focal_length));
        float3 tv = sub(backProject(x, yd, depth[yd * width + x], width, height, focal_length),
                        backProject(x, yu, depth[yu * width + x], width, height, focal_length));
        tu = scale3(tu, (float)stride / max(xr - xl, 1));
        tv = scale3(tv, (float)stride / max(yd - yu, 1));

        // Local frame: a1 along tu, normal n facing the camera, a2 = n x a1
        n = normalize3(cross3(tu, tv));
        if (dot3(n, p) > 0.0f) n = scale3(n, -1.0f);
        a1 = normalize3(tu);
        a2 = cross3(n, a1);

        // Sizes along a1 and a2, limited so grazing surfaces don't give needle-like splats
        s1 = SPLAT_SIGMA * fminf(sqrtf(dot3(tu, tu)), MAX_ANISOTROPY * footprint);
        s2 = SPLAT_SIGMA * fminf(fabsf(dot3(tv, a2)), MAX_ANISOTROPY * footprint);
        float smax = fmaxf(fmaxf(s1, s2), 1e-6f);
        s1 = fmaxf(s1, smax / MAX_ANISOTROPY);
        s2 = fmaxf(s2, smax / MAX_ANISOTROPY);
    }

    if (onEdge) atomicAdd(&counts[1], 1);
    int out = atomicAdd(&counts[0], 1);
    Gaussian& g = splats[out];
    g.pos[0] = p.x; g.pos[1] = p.y; g.pos[2] = p.z;
    matrixToQuaternion(a1, a2, n, g.rot);
    g.scale[0] = s1;
    g.scale[1] = s2;
    g.scale[2] = SPLAT_THICKNESS * fminf(s1, s2);
    g.opacity = SPLAT_OPACITY;

    // OpenCV stores images in BGR format, so we swap them to RGB
    g.rgb[0] = color[idx * 3 + 2] / 255.0f;
    g.rgb[1] = color[idx * 3 + 1] / 255.0f;
    g.rgb[2] = color[idx * 3 + 0] / 255.0f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./splat <image.jpg> [output.ply] [model.onnx] [max_points] [depth_range] [edge_threshold]"
             << endl;
        return -1;
    }
    string outPath = argc > 2 ? argv[2] : "output.ply";
    string modelPath = argc > 3 ? argv[3] : "depth_anything_v2_small.onnx";
    long maxPoints = argc > 4 ? atol(argv[4]) : 500000;
    float depthRange = argc > 5 ? (float)atof(argv[5]) : 10.0f;
    float edgeThreshold = argc > 6 ? (float)atof(argv[6]) : 0.1f;
    depthRange = max(depthRange, 1.01f);

    // 1. Load the Image
    Mat img = imread(argv[1]);
    if (img.empty()) {
        cerr << "Failed to load image " << argv[1] << endl;
        return -1;
    }
    int width = img.cols;
    int height = img.rows;
    int num_pixels = width * height;

    // 2. Load the depth model into ONNX Runtime with the CUDA provider.
    // There is deliberately no CPU fallback: if CUDA can't be used, this throws.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "splat");
    Ort::SessionOptions options;
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;
    Ort::Session* session;
    try {
        options.AppendExecutionProvider_CUDA(cudaOptions);
        session = new Ort::Session(env, modelPath.c_str(), options);
    } catch (const Ort::Exception& e) {
        cerr << "Could not start the depth model on the GPU: " << e.what() << endl;
        return -1;
    }

    // Input size: fixed by the model (MiDaS small: 256x256), or chosen here when dynamic
    vector<int64_t> modelShape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int inH = (int)modelShape[2];
    int inW = (int)modelShape[3];
    if (inH <= 0 || inW <= 0) {
        double scale = (double)DYNAMIC_SHORT_SIDE / min(width, height);
        inH = max(PATCH, (int)lround(height * scale / PATCH) * PATCH);
        inW = max(PATCH, (int)lround(width * scale / PATCH) * PATCH);
    }
    cout << "Depth model: " << modelPath << ", ONNX Runtime " << Ort::GetVersionString() << " on CUDA, input "
         << inW << "x" << inH << endl;

    // 3. Generate Depth Map: (rgb/255 - mean) / std, NCHW
    Mat blob = blobFromImage(img, 1.0 / 255.0, Size(inW, inH), Scalar(), true, false, CV_32F);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdev[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        Mat plane(inH, inW, CV_32F, blob.ptr<float>(0, c));
        plane = (plane - mean[c]) / stdev[c];
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};
    int64_t inputShape[] = {1, 3, inH, inW};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(memInfo, blob.ptr<float>(), blob.total(), inputShape, 4);

    // The first run includes cuDNN setup, so time a second run as well
    vector<Ort::Value> outputs;
    for (int run = 0; run < 2; ++run) {
        auto t0 = chrono::steady_clock::now();
        try {
            outputs = session->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, 1);
        } catch (const Ort::Exception& e) {
            cerr << "Depth model failed on the GPU: " << e.what() << endl;
            return -1;
        }
        auto ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
        cout << "Depth inference " << (run == 0 ? "(first run)" : "(warm)") << ": " << ms << " ms" << endl;
    }

    // Output is [1, H, W] (or [1, 1, H, W]); resize back to the original image size.
    // Linear, not cubic: cubic overshoots at depth edges and adds even more floating points.
    vector<int64_t> outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int dh = (int)outShape[outShape.size() - 2];
    int dw = (int)outShape[outShape.size() - 1];
    Mat depthMap(dh, dw, CV_32F, outputs[0].GetTensorMutableData<float>());
    resize(depthMap, depthMap, img.size(), 0, 0, INTER_LINEAR);

    // Both models output relative inverse depth (disparity) with an unknown scale and
    // shift, so the real distances can't be recovered. Normalise it to [0, 1] and map it
    // linearly onto inverse depth between 1/depthRange (farthest) and 1 (nearest),
    // so depth runs from 1 to depthRange.
    double dmin, dmax;
    minMaxLoc(depthMap, &dmin, &dmax);
    Mat disparity = (depthMap - dmin) / max(dmax - dmin, 1e-6);
    float invFar = 1.0f / depthRange;
    Mat trueDepth;
    divide(1.0, disparity * (1.0f - invFar) + invFar, trueDepth);

    // Sample every stride-th pixel so that at most maxPoints points are written
    int stride = max(1, (int)ceil(sqrt((double)num_pixels / max(maxPoints, 1L))));
    int outWidth = (width + stride - 1) / stride;
    int outHeight = (height + stride - 1) / stride;
    int num_samples = outWidth * outHeight;
    // Compare with neighbours at least one depth-model pixel away, where the blur shows up
    int radius = max(stride, (int)ceil((double)width / dw));

    // 4. Allocate GPU Memory
    unsigned char* d_color;
    float* d_depth;
    Gaussian* d_splats;
    int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_color, num_pixels * 3));
    CUDA_CHECK(cudaMalloc(&d_depth, num_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_splats, num_samples * sizeof(Gaussian)));
    CUDA_CHECK(cudaMalloc(&d_counts, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counts, 0, 2 * sizeof(int)));

    // imread and divide give continuous Mats, so one flat copy each is enough
    CUDA_CHECK(cudaMemcpy(d_color, img.data, num_pixels * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth, trueDepth.ptr<float>(), num_pixels * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Launch Custom CUDA Kernel
    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

    float focal_length = width * 0.8f; // Estimated focal length
    generateGaussians<<<grid, block>>>(d_color, d_depth, d_splats, d_counts, width, height, stride, outWidth,
                                       outHeight, radius, edgeThreshold, focal_length);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy back to Host and Save
    int counts[2];
    CUDA_CHECK(cudaMemcpy(counts, d_counts, sizeof(counts), cudaMemcpyDeviceToHost));
    int num_points = counts[0];
    vector<Gaussian> splats(num_points);
    CUDA_CHECK(cudaMemcpy(splats.data(), d_splats, num_points * sizeof(Gaussian), cudaMemcpyDeviceToHost));

    cout << "Image " << width << "x" << height << ", stride " << stride << ", depth range 1.." << depthRange
         << ", kept " << num_points << " of " << num_samples << " points (" << counts[1]
         << " on depth edges, moved to the background)" << endl;
    cout << "Saving " << num_points << " Gaussians to " << outPath << "..." << endl;
    SplatCamera cam;
    cam.fovY = 2.0f * atanf(height / 2.0f / focal_length) * 180.0f / (float)M_PI;
    cam.width = width;
    cam.height = height;
    if (!writeGaussianPly(outPath, splats, cam)) {
        cerr << "Failed to write " << outPath << endl;
        return -1;
    }

    // Cleanup
    cudaFree(d_color); cudaFree(d_depth); cudaFree(d_splats); cudaFree(d_counts);
    delete session;

    return 0;
}
