// splat_server: renders the 3D Gaussians made by splat with CUDA and serves them to a browser.
//
// The browser page (src/viewer.html) only shows images: whenever the camera moves it asks
// for a new frame, which is rasterised here on the GPU and sent back as a JPEG.
//
//   GET /                  the viewer page
//   GET /list              JSON list of data/ply/*.ply
//   GET /load?ply=NAME     load data/ply/NAME onto the GPU, returns JSON with the photo camera
//   GET /frame?w=&h=&px=&py=&pz=&tx=&ty=&tz=&fov=
//                          render from camera position p looking at target t (vertical fov in
//                          degrees), returns a JPEG
//
// Rendering follows the tile rasteriser of 3D Gaussian Splatting (Kerbl et al. 2023):
//   1. preprocess      one thread per Gaussian: project its 3D covariance to a 2D ellipse
//                      (EWA splatting), find the 16x16 pixel tiles it overlaps
//   2. scan            prefix sum of the tile counts (CUB)
//   3. duplicate       one (tile, depth) key per Gaussian per overlapped tile
//   4. sort            radix sort of the keys (CUB): grouped by tile, front to back within a tile
//   5. tile ranges     where each tile's list starts and ends
//   6. render          one thread block per tile, front-to-back alpha blending
//   7. encode          JPEG straight from GPU memory with nvJPEG
//
// Usage: splat_server [port] [ply_dir]   (defaults 8080 and data/ply/, run from the repository root)

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include "gaussians.h"

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

#define NVJPEG_CHECK(call)                                                            \
    do {                                                                              \
        nvjpegStatus_t st = (call);                                                   \
        if (st != NVJPEG_STATUS_SUCCESS) {                                            \
            cerr << "nvJPEG error " << st << " at " << __FILE__ << ":" << __LINE__    \
                 << endl;                                                             \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

const int TILE = 16;
const int BLOCK_PIXELS = TILE * TILE;
string PLY_DIR = "data/ply/";
const char* VIEWER_PAGE = "src/viewer.html";
const float3 BACKGROUND = {0.07f, 0.07f, 0.07f};
const int JPEG_QUALITY = 85;
const int MAX_SIZE = 2560;

// ---------------------------------------------------------------- camera

struct Camera {
    float r[9];       // world -> camera rotation, rows = right, down, forward
    float3 pos;
    float fx, fy, cx, cy;
    float tanFovX, tanFovY;
    int width, height;
};

// Camera at p looking at t. World and camera both use y pointing down (like the photo),
// so the "down" hint is +y.
Camera makeCamera(float3 p, float3 t, float fovYDeg, int width, int height) {
    Camera c;
    float3 f = make_float3(t.x - p.x, t.y - p.y, t.z - p.z);
    float fl = sqrtf(f.x * f.x + f.y * f.y + f.z * f.z);
    f = fl > 1e-9f ? make_float3(f.x / fl, f.y / fl, f.z / fl) : make_float3(0, 0, 1);
    float3 downHint = fabsf(f.y) > 0.999f ? make_float3(0, 0, 1) : make_float3(0, 1, 0);
    // right = down x forward
    float3 r = make_float3(downHint.y * f.z - downHint.z * f.y, downHint.z * f.x - downHint.x * f.z,
                           downHint.x * f.y - downHint.y * f.x);
    float rl = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
    r = make_float3(r.x / rl, r.y / rl, r.z / rl);
    // down = forward x right
    float3 d = make_float3(f.y * r.z - f.z * r.y, f.z * r.x - f.x * r.z, f.x * r.y - f.y * r.x);
    float m[9] = {r.x, r.y, r.z, d.x, d.y, d.z, f.x, f.y, f.z};
    memcpy(c.r, m, sizeof(m));
    c.pos = p;
    c.width = width;
    c.height = height;
    c.tanFovY = tanf(fovYDeg * (float)M_PI / 360.0f);
    c.tanFovX = c.tanFovY * width / height;
    c.fy = height / (2.0f * c.tanFovY);
    c.fx = c.fy;
    c.cx = width / 2.0f;
    c.cy = height / 2.0f;
    return c;
}

// ---------------------------------------------------------------- kernels

// 1. Project each Gaussian to the screen: 2D mean, conic (inverse 2D covariance), depth,
//    and the rectangle of tiles its 3-sigma ellipse overlaps.
__global__ void preprocess(const Gaussian* gs, int n, Camera cam, int tilesX, int tilesY,
                           float2* means2D, float4* conicOpacity, float* depths, float3* colors,
                           uint32_t* tilesTouched, int4* rects) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    tilesTouched[i] = 0;
    const Gaussian g = gs[i];

    // view space
    float wx = g.pos[0] - cam.pos.x, wy = g.pos[1] - cam.pos.y, wz = g.pos[2] - cam.pos.z;
    float tx = cam.r[0] * wx + cam.r[1] * wy + cam.r[2] * wz;
    float ty = cam.r[3] * wx + cam.r[4] * wy + cam.r[5] * wz;
    float tz = cam.r[6] * wx + cam.r[7] * wy + cam.r[8] * wz;
    if (tz < 0.05f) return;

    // 3D covariance: Sigma = R S S^T R^T, M = R S
    float qw = g.rot[0], qx = g.rot[1], qy = g.rot[2], qz = g.rot[3];
    float R[9] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy),
                  2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                  2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)};
    float M[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) M[row * 3 + col] = R[row * 3 + col] * g.scale[col];
    float S[9];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            S[a * 3 + b] = M[a * 3 + 0] * M[b * 3 + 0] + M[a * 3 + 1] * M[b * 3 + 1] + M[a * 3 + 2] * M[b * 3 + 2];

    // EWA: 2D covariance = J W Sigma W^T J^T, with W the view rotation and J the Jacobian
    // of the perspective projection at the Gaussian's centre (clamped near the frustum edge)
    float limX = 1.3f * cam.tanFovX, limY = 1.3f * cam.tanFovY;
    float txz = fminf(limX, fmaxf(-limX, tx / tz)) * tz;
    float tyz = fminf(limY, fmaxf(-limY, ty / tz)) * tz;
    float J[6] = {cam.fx / tz, 0.0f, -cam.fx * txz / (tz * tz),
                  0.0f, cam.fy / tz, -cam.fy * tyz / (tz * tz)};
    float T[9];  // T = J W (2x3, stored in the first 6 entries)
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 3; ++col)
            T[row * 3 + col] = J[row * 3 + 0] * cam.r[0 * 3 + col] + J[row * 3 + 1] * cam.r[1 * 3 + col] +
                               J[row * 3 + 2] * cam.r[2 * 3 + col];
    float TS[6];
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 3; ++col)
            TS[row * 3 + col] = T[row * 3 + 0] * S[0 * 3 + col] + T[row * 3 + 1] * S[1 * 3 + col] +
                                T[row * 3 + 2] * S[2 * 3 + col];
    float a = TS[0] * T[0] + TS[1] * T[1] + TS[2] * T[2] + 0.3f;  // +0.3: at least ~1 pixel wide
    float b = TS[0] * T[3] + TS[1] * T[4] + TS[2] * T[5];
    float c = TS[3] * T[3] + TS[4] * T[4] + TS[5] * T[5] + 0.3f;

    float det = a * c - b * b;
    if (det <= 0.0f) return;
    float invDet = 1.0f / det;
    float mid = 0.5f * (a + c);
    float lambda = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    float radius = ceilf(3.0f * sqrtf(lambda));

    float2 px = make_float2(cam.fx * tx / tz + cam.cx, cam.fy * ty / tz + cam.cy);
    int4 rect = make_int4(min(tilesX, max(0, (int)((px.x - radius) / TILE))),
                          min(tilesY, max(0, (int)((px.y - radius) / TILE))),
                          min(tilesX, max(0, (int)((px.x + radius + TILE - 1) / TILE))),
                          min(tilesY, max(0, (int)((px.y + radius + TILE - 1) / TILE))));
    int area = (rect.z - rect.x) * (rect.w - rect.y);
    if (area == 0) return;

    means2D[i] = px;
    conicOpacity[i] = make_float4(c * invDet, -b * invDet, a * invDet, g.opacity);
    depths[i] = tz;
    colors[i] = make_float3(g.rgb[0], g.rgb[1], g.rgb[2]);
    rects[i] = rect;
    tilesTouched[i] = area;
}

// 3. One key per (Gaussian, tile): high 32 bits = tile index, low 32 bits = depth
//    (a positive float's bits sort like the float), so one sort gives per-tile depth order.
__global__ void duplicateWithKeys(int n, const uint32_t* tilesTouched, const uint32_t* offsets, const int4* rects,
                                  const float* depths, int tilesX, uint64_t* keys, uint32_t* values) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || tilesTouched[i] == 0) return;
    uint32_t off = offsets[i];
    int4 r = rects[i];
    uint64_t depthBits = __float_as_uint(depths[i]);
    for (int y = r.y; y < r.w; ++y)
        for (int x = r.x; x < r.z; ++x) {
            keys[off] = ((uint64_t)(y * tilesX + x) << 32) | depthBits;
            values[off] = i;
            ++off;
        }
}

// 5. Start and end of each tile's run in the sorted list
__global__ void identifyTileRanges(int count, const uint64_t* keys, uint2* ranges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t tile = keys[i] >> 32;
    if (i == 0) {
        ranges[tile].x = 0;
    } else {
        uint32_t prev = keys[i - 1] >> 32;
        if (tile != prev) {
            ranges[prev].y = i;
            ranges[tile].x = i;
        }
    }
    if (i == count - 1) ranges[tile].y = count;
}

// 6. One block per 16x16 tile, one thread per pixel. The tile's Gaussians are loaded into
//    shared memory in batches and blended front to back; a pixel stops once it is opaque,
//    and the block stops when all its pixels are.
__global__ void renderTiles(const uint2* ranges, const uint32_t* values, const float2* means2D,
                            const float4* conicOpacity, const float3* colors, int width, int height,
                            float3 background, unsigned char* image) {
    int px = blockIdx.x * TILE + threadIdx.x;
    int py = blockIdx.y * TILE + threadIdx.y;
    bool inside = px < width && py < height;
    bool done = !inside;
    uint2 range = ranges[blockIdx.y * gridDim.x + blockIdx.x];

    __shared__ float2 sMean[BLOCK_PIXELS];
    __shared__ float4 sConic[BLOCK_PIXELS];
    __shared__ float3 sColor[BLOCK_PIXELS];

    float T = 1.0f;
    float3 C = make_float3(0, 0, 0);
    int tid = threadIdx.y * TILE + threadIdx.x;
    float2 pixel = make_float2(px + 0.5f, py + 0.5f);

    for (int start = range.x; start < (int)range.y; start += BLOCK_PIXELS) {
        if (__syncthreads_count(done) == BLOCK_PIXELS) break;
        int j = start + tid;
        if (j < (int)range.y) {
            uint32_t id = values[j];
            sMean[tid] = means2D[id];
            sConic[tid] = conicOpacity[id];
            sColor[tid] = colors[id];
        }
        __syncthreads();
        int batch = min(BLOCK_PIXELS, (int)range.y - start);
        for (int k = 0; k < batch && !done; ++k) {
            float2 d = make_float2(sMean[k].x - pixel.x, sMean[k].y - pixel.y);
            float4 co = sConic[k];
            float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0.0f) continue;
            float alpha = fminf(0.99f, co.w * __expf(power));
            if (alpha < 1.0f / 255.0f) continue;
            float nextT = T * (1.0f - alpha);
            if (nextT < 1e-4f) { done = true; break; }
            C.x += sColor[k].x * alpha * T;
            C.y += sColor[k].y * alpha * T;
            C.z += sColor[k].z * alpha * T;
            T = nextT;
        }
    }

    if (inside) {
        int o = (py * width + px) * 3;
        image[o + 0] = (unsigned char)(fminf(1.0f, C.x + T * background.x) * 255.0f + 0.5f);
        image[o + 1] = (unsigned char)(fminf(1.0f, C.y + T * background.y) * 255.0f + 0.5f);
        image[o + 2] = (unsigned char)(fminf(1.0f, C.z + T * background.z) * 255.0f + 0.5f);
    }
}

// ---------------------------------------------------------------- renderer

// GPU buffers, grown when needed and reused from frame to frame
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t size = 0;
    void reserve(size_t n) {
        if (n <= size) return;
        if (ptr) cudaFree(ptr);
        size = max(n, size + size / 2);
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    }
};

struct Renderer {
    int count = 0;
    DeviceBuffer<Gaussian> gaussians;
    DeviceBuffer<float2> means2D;
    DeviceBuffer<float4> conicOpacity;
    DeviceBuffer<float> depths;
    DeviceBuffer<float3> colors;
    DeviceBuffer<uint32_t> tilesTouched, offsets;
    DeviceBuffer<int4> rects;
    DeviceBuffer<uint64_t> keys, keysSorted;
    DeviceBuffer<uint32_t> values, valuesSorted;
    DeviceBuffer<uint2> ranges;
    DeviceBuffer<unsigned char> image, scanTemp, sortTemp;
    cudaEvent_t start, stop;
    nvjpegHandle_t jpegHandle;
    nvjpegEncoderState_t jpegState;
    nvjpegEncoderParams_t jpegParams;

    Renderer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        NVJPEG_CHECK(nvjpegCreateSimple(&jpegHandle));
        NVJPEG_CHECK(nvjpegEncoderStateCreate(jpegHandle, &jpegState, 0));
        NVJPEG_CHECK(nvjpegEncoderParamsCreate(jpegHandle, &jpegParams, 0));
        NVJPEG_CHECK(nvjpegEncoderParamsSetQuality(jpegParams, JPEG_QUALITY, 0));
        NVJPEG_CHECK(nvjpegEncoderParamsSetSamplingFactors(jpegParams, NVJPEG_CSS_420, 0));
    }

    // 7. JPEG-encode the last rendered frame on the GPU; returns the time in ms
    float encode(int width, int height, string& jpeg) {
        auto t0 = chrono::steady_clock::now();
        nvjpegImage_t src = {};
        src.channel[0] = image.ptr;
        src.pitch[0] = width * 3;
        NVJPEG_CHECK(nvjpegEncodeImage(jpegHandle, jpegState, jpegParams, &src, NVJPEG_INPUT_RGBI, width, height, 0));
        size_t size = 0;
        NVJPEG_CHECK(nvjpegEncodeRetrieveBitstream(jpegHandle, jpegState, nullptr, &size, 0));
        jpeg.resize(size);
        NVJPEG_CHECK(nvjpegEncodeRetrieveBitstream(jpegHandle, jpegState, (unsigned char*)&jpeg[0], &size, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
        jpeg.resize(size);
        return (float)chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
    }

    void load(const vector<Gaussian>& gs) {
        count = (int)gs.size();
        gaussians.reserve(count);
        means2D.reserve(count);
        conicOpacity.reserve(count);
        depths.reserve(count);
        colors.reserve(count);
        tilesTouched.reserve(count);
        offsets.reserve(count);
        rects.reserve(count);
        CUDA_CHECK(cudaMemcpy(gaussians.ptr, gs.data(), count * sizeof(Gaussian), cudaMemcpyHostToDevice));
    }

    // Renders into `image` on the GPU (RGB, width*height*3) and returns the GPU time in ms
    float render(const Camera& cam, uint32_t& instances) {
        int tilesX = (cam.width + TILE - 1) / TILE, tilesY = (cam.height + TILE - 1) / TILE;
        int numTiles = tilesX * tilesY;
        image.reserve((size_t)cam.width * cam.height * 3);
        ranges.reserve(numTiles);
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemset(ranges.ptr, 0, numTiles * sizeof(uint2)));

        instances = 0;
        if (count > 0) {
            int threads = 256, blocks = (count + threads - 1) / threads;
            preprocess<<<blocks, threads>>>(gaussians.ptr, count, cam, tilesX, tilesY, means2D.ptr,
                                            conicOpacity.ptr, depths.ptr, colors.ptr, tilesTouched.ptr, rects.ptr);
            CUDA_CHECK(cudaGetLastError());

            // 2. prefix sum -> where each Gaussian's keys go, and the total
            size_t tempBytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, tilesTouched.ptr, offsets.ptr, count);
            scanTemp.reserve(tempBytes);
            cub::DeviceScan::ExclusiveSum(scanTemp.ptr, tempBytes, tilesTouched.ptr, offsets.ptr, count);
            uint32_t lastOffset, lastCount;
            CUDA_CHECK(cudaMemcpy(&lastOffset, offsets.ptr + count - 1, 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&lastCount, tilesTouched.ptr + count - 1, 4, cudaMemcpyDeviceToHost));
            instances = lastOffset + lastCount;
        }

        if (instances > 0) {
            keys.reserve(instances);
            keysSorted.reserve(instances);
            values.reserve(instances);
            valuesSorted.reserve(instances);
            int threads = 256;
            duplicateWithKeys<<<(count + threads - 1) / threads, threads>>>(
                count, tilesTouched.ptr, offsets.ptr, rects.ptr, depths.ptr, tilesX, keys.ptr, values.ptr);
            CUDA_CHECK(cudaGetLastError());

            // 4. sort only the bits in use: 32 depth bits + enough bits for the tile index
            int tileBits = 1;
            while ((1 << tileBits) < numTiles) ++tileBits;
            size_t tempBytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, keys.ptr, keysSorted.ptr, values.ptr,
                                            valuesSorted.ptr, (int)instances, 0, 32 + tileBits);
            sortTemp.reserve(tempBytes);
            cub::DeviceRadixSort::SortPairs(sortTemp.ptr, tempBytes, keys.ptr, keysSorted.ptr, values.ptr,
                                            valuesSorted.ptr, (int)instances, 0, 32 + tileBits);

            identifyTileRanges<<<(instances + threads - 1) / threads, threads>>>(instances, keysSorted.ptr,
                                                                              ranges.ptr);
            CUDA_CHECK(cudaGetLastError());
        }

        renderTiles<<<dim3(tilesX, tilesY), dim3(TILE, TILE)>>>(ranges.ptr, valuesSorted.ptr, means2D.ptr,
                                                                conicOpacity.ptr, colors.ptr, cam.width,
                                                                cam.height, BACKGROUND, image.ptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ---------------------------------------------------------------- HTTP

struct Request {
    string path;
    map<string, string> query;
};

string urlDecode(const string& s) {
    string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            out += (char)strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
            i += 2;
        } else {
            out += s[i] == '+' ? ' ' : s[i];
        }
    }
    return out;
}

bool readRequest(int fd, Request& req) {
    string data;
    char buf[4096];
    while (data.find("\r\n\r\n") == string::npos && data.size() < 65536) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        data.append(buf, n);
    }
    // "GET /path?query HTTP/1.1"
    size_t sp1 = data.find(' '), sp2 = data.find(' ', sp1 + 1);
    if (sp1 == string::npos || sp2 == string::npos || data.compare(0, sp1, "GET") != 0) return false;
    string target = data.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t q = target.find('?');
    req.path = urlDecode(target.substr(0, q));
    if (q != string::npos) {
        string qs = target.substr(q + 1);
        size_t pos = 0;
        while (pos <= qs.size()) {
            size_t amp = qs.find('&', pos);
            if (amp == string::npos) amp = qs.size();
            string kv = qs.substr(pos, amp - pos);
            size_t eq = kv.find('=');
            if (eq != string::npos) req.query[urlDecode(kv.substr(0, eq))] = urlDecode(kv.substr(eq + 1));
            pos = amp + 1;
        }
    }
    return true;
}

void sendAll(int fd, const char* data, size_t size) {
    while (size > 0) {
        ssize_t n = send(fd, data, size, MSG_NOSIGNAL);
        if (n <= 0) return;
        data += n;
        size -= n;
    }
}

void respond(int fd, int status, const string& type, const string& body, const string& extraHeaders = "") {
    const char* text = status == 200 ? "OK" : status == 404 ? "Not Found" : "Bad Request";
    string head = "HTTP/1.1 " + to_string(status) + " " + text + "\r\nContent-Type: " + type +
                  "\r\nContent-Length: " + to_string(body.size()) +
                  "\r\nCache-Control: no-store\r\nConnection: close\r\n" + extraHeaders + "\r\n";
    sendAll(fd, head.data(), head.size());
    sendAll(fd, body.data(), body.size());
}

string jsonEscape(const string& s) {
    string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

vector<string> listPlyFiles() {
    vector<string> names;
    if (DIR* dir = opendir(PLY_DIR.c_str())) {
        while (dirent* e = readdir(dir)) {
            string name = e->d_name;
            if (name.size() > 4 && name.compare(name.size() - 4, 4, ".ply") == 0) names.push_back(name);
        }
        closedir(dir);
    }
    sort(names.begin(), names.end());
    return names;
}

float queryFloat(const Request& req, const char* key, float fallback) {
    auto it = req.query.find(key);
    return it == req.query.end() ? fallback : strtof(it->second.c_str(), nullptr);
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    int port = argc > 1 ? atoi(argv[1]) : 8080;
    if (argc > 2) PLY_DIR = string(argv[2]) + "/";
    signal(SIGPIPE, SIG_IGN);

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int server = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(server, (sockaddr*)&addr, sizeof(addr)) != 0 || listen(server, 16) != 0) {
        perror("bind/listen");
        return 1;
    }
    cout << "splat_server on " << prop.name << ": open http://localhost:" << port << "/  (Ctrl+C stops it)" << endl;

    Renderer renderer;
    string loaded;
    SplatCamera loadedCam;

    while (true) {
        int fd = accept(server, nullptr, nullptr);
        if (fd < 0) continue;
        Request req;
        if (!readRequest(fd, req)) {
            close(fd);
            continue;
        }

        if (req.path == "/" || req.path == "/viewer.html") {
            ifstream in(VIEWER_PAGE, ios::binary);
            stringstream ss;
            ss << in.rdbuf();
            if (in) respond(fd, 200, "text/html; charset=utf-8", ss.str());
            else respond(fd, 404, "text/plain", string("missing ") + VIEWER_PAGE + " (run from the repository root)");

        } else if (req.path == "/list") {
            string json = "[";
            for (const string& name : listPlyFiles()) json += (json.size() > 1 ? ",\"" : "\"") + jsonEscape(name) + "\"";
            respond(fd, 200, "application/json", json + "]");

        } else if (req.path == "/load") {
            string name = req.query["ply"];
            if (name.empty() || name.find('/') != string::npos || name.find("..") != string::npos) {
                respond(fd, 400, "text/plain", "bad file name");
                close(fd);
                continue;
            }
            vector<Gaussian> gs;
            SplatCamera cam;
            string error;
            auto t0 = chrono::steady_clock::now();
            if (!readGaussianPly(PLY_DIR + name, gs, cam, error)) {
                respond(fd, 400, "text/plain", name + ": " + error);
                close(fd);
                continue;
            }
            renderer.load(gs);
            loaded = name;
            loadedCam = cam;
            // middle depth (median z), for the orbit centre
            vector<float> zs;
            for (size_t i = 0; i < gs.size(); i += 16) zs.push_back(gs[i].pos[2]);
            nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
            float zMid = zs.empty() ? 1.0f : zs[zs.size() / 2];
            double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
            cout << "loaded " << name << ": " << gs.size() << " Gaussians in " << (int)ms << " ms" << endl;
            respond(fd, 200, "application/json",
                    "{\"name\":\"" + jsonEscape(name) + "\",\"count\":" + to_string(gs.size()) +
                        ",\"fov_y\":" + to_string(cam.fovY) + ",\"width\":" + to_string(cam.width) +
                        ",\"height\":" + to_string(cam.height) + ",\"z_mid\":" + to_string(zMid) + "}");

        } else if (req.path == "/frame") {
            int w = min(MAX_SIZE, max(16, (int)queryFloat(req, "w", 1280)));
            int h = min(MAX_SIZE, max(16, (int)queryFloat(req, "h", 720)));
            float3 p = make_float3(queryFloat(req, "px", 0), queryFloat(req, "py", 0), queryFloat(req, "pz", 0));
            float3 t = make_float3(queryFloat(req, "tx", 0), queryFloat(req, "ty", 0), queryFloat(req, "tz", 1));
            float fov = min(170.0f, max(1.0f, queryFloat(req, "fov", loadedCam.fovY)));
            Camera cam = makeCamera(p, t, fov, w, h);

            uint32_t instances;
            float gpuMs = renderer.render(cam, instances);
            string jpeg;
            float encodeMs = renderer.encode(w, h, jpeg);
            char headers[256];
            snprintf(headers, sizeof(headers),
                     "X-Render-Ms: %.2f\r\nX-Encode-Ms: %.2f\r\nX-Splats: %d\r\nX-Instances: %u\r\n"
                     "Access-Control-Expose-Headers: X-Render-Ms, X-Encode-Ms, X-Splats, X-Instances\r\n",
                     gpuMs, encodeMs, renderer.count, instances);
            respond(fd, 200, "image/jpeg", jpeg, headers);

        } else {
            respond(fd, 404, "text/plain", "not found");
        }
        close(fd);
    }
}
