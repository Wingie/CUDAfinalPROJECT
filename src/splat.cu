// 2D photo -> 3D point cloud: estimates depth with a monocular depth model
// (Depth Anything V2 or MiDaS, ONNX Runtime on the GPU), then a custom CUDA kernel
// back-projects each pixel through a pinhole camera into 3D and drops the points
// that sit on depth edges. The result is an ASCII PLY that src/index.html shows as soft "splats".
//
// Usage: splat <image.jpg> [output.ply] [model.onnx] [max_points] [depth_range] [edge_threshold]
//   depth_range     far/near ratio of the scene (default 10); larger = more depth
//   edge_threshold  drop a point when a neighbour's depth differs by more than this
//                   fraction (default 0.1); 0 keeps every point

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

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

// Structure to hold our 3D point data
struct Point3D {
    float x, y, z;
    unsigned char r, g, b;
};

// --- CUSTOM CUDA KERNEL ---
// This kernel converts 2D pixels + Depth into 3D space. One thread per sampled
// pixel; every `stride`-th pixel is sampled so the PLY stays small enough for a browser.
// The depth model blurs depth across object outlines, which would leave a "curtain"
// of points floating between foreground and background. So a pixel whose depth differs
// from a neighbour `radius` pixels away by more than `edgeThreshold` (relative) is dropped.
// Kept points are packed to the front of `points`; `count` returns how many there are.
__global__ void generatePointCloud(const unsigned char* color, const float* depth, Point3D* points, int* count,
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
    if (edgeThreshold > 0.0f) {
        float zmin = z_val, zmax = z_val;
        int nx[4] = {max(x - radius, 0), min(x + radius, width - 1), x, x};
        int ny[4] = {y, y, max(y - radius, 0), min(y + radius, height - 1)};
        for (int i = 0; i < 4; ++i) {
            float zn = depth[ny[i] * width + nx[i]];
            zmin = fminf(zmin, zn);
            zmax = fmaxf(zmax, zn);
        }
        if (zmax - zmin > edgeThreshold * zmin) return;
    }

    int out = atomicAdd(count, 1);

    // Pinhole camera math to project 2D to 3D
    points[out].x = (x - width / 2.0f) * z_val / focal_length;
    points[out].y = (y - height / 2.0f) * z_val / focal_length;
    points[out].z = z_val;

    // OpenCV stores images in BGR format, so we swap them to RGB
    points[out].r = color[idx * 3 + 2];
    points[out].g = color[idx * 3 + 1];
    points[out].b = color[idx * 3 + 0];
}

void savePLY(const string& filename, Point3D* points, int num_points) {
    ofstream file(filename);
    file << "ply\nformat ascii 1.0\nelement vertex " << num_points << "\n";
    file << "property float x\nproperty float y\nproperty float z\n";
    file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    file << "end_header\n";
    for (int i = 0; i < num_points; i++) {
        file << points[i].x << " " << points[i].y << " " << points[i].z << " "
             << (int)points[i].r << " " << (int)points[i].g << " " << (int)points[i].b << "\n";
    }
    file.close();
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
    Point3D* d_points;
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_color, num_pixels * 3));
    CUDA_CHECK(cudaMalloc(&d_depth, num_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_points, num_samples * sizeof(Point3D)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    // imread and divide give continuous Mats, so one flat copy each is enough
    CUDA_CHECK(cudaMemcpy(d_color, img.data, num_pixels * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth, trueDepth.ptr<float>(), num_pixels * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Launch Custom CUDA Kernel
    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

    float focal_length = width * 0.8f; // Estimated focal length
    generatePointCloud<<<grid, block>>>(d_color, d_depth, d_points, d_count, width, height, stride, outWidth,
                                        outHeight, radius, edgeThreshold, focal_length);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy back to Host and Save
    int num_points;
    CUDA_CHECK(cudaMemcpy(&num_points, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    Point3D* h_points = new Point3D[num_points];
    CUDA_CHECK(cudaMemcpy(h_points, d_points, num_points * sizeof(Point3D), cudaMemcpyDeviceToHost));

    cout << "Image " << width << "x" << height << ", stride " << stride << ", depth range 1.." << depthRange
         << ", kept " << num_points << " of " << num_samples << " points (" << num_samples - num_points
         << " on depth edges)" << endl;
    cout << "Saving 3D Splat to " << outPath << "..." << endl;
    savePLY(outPath, h_points, num_points);

    // Cleanup
    cudaFree(d_color); cudaFree(d_depth); cudaFree(d_points); cudaFree(d_count);
    delete[] h_points;
    delete session;

    return 0;
}
