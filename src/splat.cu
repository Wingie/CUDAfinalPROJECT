// 2D photo -> 3D point cloud: estimates depth with MiDaS (ONNX Runtime on the GPU),
// then a custom CUDA kernel back-projects each pixel through a pinhole camera into 3D.
// The result is an ASCII PLY that src/index.html shows as soft "splats".
//
// Usage: splat <image.jpg> [output.ply] [model.onnx] [max_points]

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

// midas_v21_small_256 takes a 256x256 RGB image normalised with ImageNet mean/std
const int MIDAS_SIZE = 256;

// Structure to hold our 3D point data
struct Point3D {
    float x, y, z;
    unsigned char r, g, b;
};

// --- CUSTOM CUDA KERNEL ---
// This kernel converts 2D pixels + Depth into 3D space. One thread per output
// point; every `stride`-th pixel is sampled so the PLY stays small enough for a browser.
__global__ void generatePointCloud(unsigned char* color, float* depth, Point3D* points, int width, int height,
                                   int stride, int outWidth, int outHeight, float focal_length) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox < outWidth && oy < outHeight) {
        int x = ox * stride;
        int y = oy * stride;
        int idx = y * width + x;
        int out = oy * outWidth + ox;
        float z_val = depth[idx];

        // Pinhole camera math to project 2D to 3D
        points[out].x = (x - width / 2.0f) * z_val / focal_length;
        points[out].y = (y - height / 2.0f) * z_val / focal_length;
        points[out].z = z_val;

        // OpenCV stores images in BGR format, so we swap them to RGB
        points[out].r = color[idx * 3 + 2];
        points[out].g = color[idx * 3 + 1];
        points[out].b = color[idx * 3 + 0];
    }
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
        cout << "Usage: ./splat <image.jpg> [output.ply] [model.onnx] [max_points]" << endl;
        return -1;
    }
    string outPath = argc > 2 ? argv[2] : "output.ply";
    string modelPath = argc > 3 ? argv[3] : "midas.onnx";
    long maxPoints = argc > 4 ? atol(argv[4]) : 500000;

    // 1. Load the Image
    Mat img = imread(argv[1]);
    if (img.empty()) {
        cerr << "Failed to load image " << argv[1] << endl;
        return -1;
    }
    int width = img.cols;
    int height = img.rows;
    int num_pixels = width * height;

    // 2. Load MiDaS Depth Estimation Model into ONNX Runtime with the CUDA provider.
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
        cerr << "Could not start MiDaS on the GPU: " << e.what() << endl;
        return -1;
    }
    cout << "MiDaS: ONNX Runtime " << Ort::GetVersionString() << " on CUDA" << endl;

    // 3. Generate Depth Map: (rgb/255 - mean) / std, NCHW
    Mat blob = blobFromImage(img, 1.0 / 255.0, Size(MIDAS_SIZE, MIDAS_SIZE), Scalar(), true, false, CV_32F);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdev[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        Mat plane(MIDAS_SIZE, MIDAS_SIZE, CV_32F, blob.ptr<float>(0, c));
        plane = (plane - mean[c]) / stdev[c];
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
    const char* inputNames[] = {inputName.get()};
    const char* outputNames[] = {outputName.get()};
    int64_t inputShape[] = {1, 3, MIDAS_SIZE, MIDAS_SIZE};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(memInfo, blob.ptr<float>(), blob.total(), inputShape, 4);

    // The first run includes cuDNN setup, so time a second run as well
    vector<Ort::Value> outputs;
    for (int run = 0; run < 2; ++run) {
        auto t0 = chrono::steady_clock::now();
        try {
            outputs = session->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, 1);
        } catch (const Ort::Exception& e) {
            cerr << "MiDaS failed on the GPU: " << e.what() << endl;
            return -1;
        }
        auto ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
        cout << "MiDaS inference " << (run == 0 ? "(first run)" : "(warm)") << ": " << ms << " ms" << endl;
    }

    // Output is [1, H, W] (or [1, 1, H, W]); resize back to the original image size
    vector<int64_t> outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int dh = (int)outShape[outShape.size() - 2];
    int dw = (int)outShape[outShape.size() - 1];
    Mat depthMap(dh, dw, CV_32F, outputs[0].GetTensorMutableData<float>());
    resize(depthMap, depthMap, img.size(), 0, 0, INTER_CUBIC);

    // MiDaS outputs relative inverse depth (disparity). Normalise it to [0, 1]
    // and invert into a depth range of about 0.67..2 so the cloud fits the viewer's camera.
    double dmin, dmax;
    minMaxLoc(depthMap, &dmin, &dmax);
    Mat disparity = (depthMap - dmin) / max(dmax - dmin, 1e-6);
    Mat trueDepth;
    divide(1.0, disparity + 0.5, trueDepth);

    // Sample every stride-th pixel so that at most maxPoints points are written
    int stride = max(1, (int)ceil(sqrt((double)num_pixels / max(maxPoints, 1L))));
    int outWidth = (width + stride - 1) / stride;
    int outHeight = (height + stride - 1) / stride;
    int num_points = outWidth * outHeight;

    // 4. Allocate GPU Memory
    unsigned char* d_color;
    float* d_depth;
    Point3D* d_points;
    CUDA_CHECK(cudaMalloc(&d_color, num_pixels * 3));
    CUDA_CHECK(cudaMalloc(&d_depth, num_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_points, num_points * sizeof(Point3D)));

    // imread and divide give continuous Mats, so one flat copy each is enough
    CUDA_CHECK(cudaMemcpy(d_color, img.data, num_pixels * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth, trueDepth.ptr<float>(), num_pixels * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Launch Custom CUDA Kernel
    dim3 block(16, 16);
    dim3 grid((outWidth + block.x - 1) / block.x, (outHeight + block.y - 1) / block.y);

    float focal_length = width * 0.8f; // Estimated focal length
    generatePointCloud<<<grid, block>>>(d_color, d_depth, d_points, width, height, stride, outWidth, outHeight,
                                        focal_length);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy back to Host and Save
    Point3D* h_points = new Point3D[num_points];
    CUDA_CHECK(cudaMemcpy(h_points, d_points, num_points * sizeof(Point3D), cudaMemcpyDeviceToHost));

    cout << "Image " << width << "x" << height << ", stride " << stride << ", " << num_points << " points" << endl;
    cout << "Saving 3D Splat to " << outPath << "..." << endl;
    savePLY(outPath, h_points, num_points);

    // Cleanup
    cudaFree(d_color); cudaFree(d_depth); cudaFree(d_points);
    delete[] h_points;
    delete session;

    return 0;
}
