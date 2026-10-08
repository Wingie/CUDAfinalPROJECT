// 3D Gaussians shared by splat (writes them) and splat_server (renders them).
//
// On disk they use the PLY layout of 3D Gaussian Splatting (Kerbl et al. 2023), so other
// splat viewers can open the files too:
//   x y z, nx ny nz, f_dc_0..2 (colour as SH degree 0), opacity (logit), scale_0..2 (log),
//   rot_0..3 (quaternion w x y z), plus red green blue as uchar for plain point viewers.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// In memory everything is linear: scale is the standard deviation along each local axis,
// rgb is 0..1, opacity is 0..1. rot is a unit quaternion (w, x, y, z) whose rotation matrix
// has the local x, y, z axes as columns.
struct Gaussian {
    float pos[3];
    float rot[4];
    float scale[3];
    float rgb[3];
    float opacity;
};

// Camera of the photo the Gaussians came from: at the origin, looking down +z, y down
struct SplatCamera {
    float fovY = 45.0f;  // degrees
    int width = 0, height = 0;
};

const float SH_C0 = 0.28209479177387814f;

inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

inline bool writeGaussianPly(const std::string& path, const std::vector<Gaussian>& gs, const SplatCamera& cam) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fprintf(f, "ply\nformat binary_little_endian 1.0\n");
    fprintf(f, "comment splat_camera fov_y=%f width=%d height=%d\n", cam.fovY, cam.width, cam.height);
    fprintf(f, "element vertex %zu\n", gs.size());
    const char* floats[] = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
    for (const char* name : floats) fprintf(f, "property float %s\n", name);
    fprintf(f, "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");

    std::vector<unsigned char> buf(gs.size() * (17 * 4 + 3));
    unsigned char* p = buf.data();
    for (const Gaussian& g : gs) {
        float w = g.rot[0], x = g.rot[1], y = g.rot[2], z = g.rot[3];
        // normal = local z axis = third column of the rotation matrix
        float n[3] = {2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)};
        float op = fminf(fmaxf(g.opacity, 1e-4f), 1 - 1e-4f);
        float rec[17] = {g.pos[0], g.pos[1], g.pos[2], n[0], n[1], n[2],
                         (g.rgb[0] - 0.5f) / SH_C0, (g.rgb[1] - 0.5f) / SH_C0, (g.rgb[2] - 0.5f) / SH_C0,
                         logf(op / (1 - op)),
                         logf(g.scale[0]), logf(g.scale[1]), logf(g.scale[2]),
                         w, x, y, z};
        memcpy(p, rec, sizeof(rec));
        p += sizeof(rec);
        for (int c = 0; c < 3; ++c) *p++ = (unsigned char)lrintf(clamp01(g.rgb[c]) * 255.0f);
    }
    size_t written = fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    return written == buf.size();
}

// Reads a binary little-endian Gaussian PLY (ours, or a standard 3DGS file; extra
// properties such as f_rest_* are skipped). Returns false with a message in `error`.
inline bool readGaussianPly(const std::string& path, std::vector<Gaussian>& gs, SplatCamera& cam,
                            std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open " + path; return false; }

    std::string line;
    std::getline(in, line);
    if (line != "ply") { error = "not a PLY file"; return false; }
    size_t count = 0;
    bool binary = false;
    struct Prop { std::string name; int offset; bool isFloat; };
    std::vector<Prop> props;
    int stride = 0;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string word;
        ss >> word;
        if (word == "format") {
            std::string fmt;
            ss >> fmt;
            binary = fmt == "binary_little_endian";
        } else if (word == "element") {
            std::string name;
            ss >> name >> count;
            if (name != "vertex") { error = "unexpected element " + name; return false; }
        } else if (word == "property") {
            std::string type, name;
            ss >> type >> name;
            int size = (type == "float" || type == "float32" || type == "int" || type == "uint") ? 4
                     : (type == "uchar" || type == "uint8" || type == "char") ? 1
                     : (type == "double") ? 8 : (type == "short" || type == "ushort") ? 2 : 0;
            if (size == 0) { error = "unsupported property type " + type; return false; }
            props.push_back({name, stride, type == "float" || type == "float32"});
            stride += size;
        } else if (word == "comment") {
            std::string tag;
            ss >> tag;
            if (tag == "splat_camera") {
                std::string kv;
                while (ss >> kv) {
                    size_t eq = kv.find('=');
                    if (eq == std::string::npos) continue;
                    std::string key = kv.substr(0, eq), val = kv.substr(eq + 1);
                    if (key == "fov_y") cam.fovY = std::stof(val);
                    else if (key == "width") cam.width = std::stoi(val);
                    else if (key == "height") cam.height = std::stoi(val);
                }
            }
        } else if (word == "end_header") {
            break;
        }
    }
    if (!binary) { error = "not a binary Gaussian PLY (regenerate it with make run-splat)"; return false; }

    auto find = [&](const char* name) -> int {
        for (const Prop& p : props) if (p.name == name && p.isFloat) return p.offset;
        return -1;
    };
    const char* needed[] = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
    int off[14];
    for (int i = 0; i < 14; ++i) {
        off[i] = find(needed[i]);
        if (off[i] < 0) { error = std::string("no float property ") + needed[i] + " (not a Gaussian PLY)"; return false; }
    }

    std::vector<unsigned char> data(count * stride);
    in.read((char*)data.data(), data.size());
    if ((size_t)in.gcount() != data.size()) { error = "file is truncated"; return false; }

    gs.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const unsigned char* rec = data.data() + i * stride;
        float v[14];
        for (int k = 0; k < 14; ++k) memcpy(&v[k], rec + off[k], 4);
        Gaussian& g = gs[i];
        g.pos[0] = v[0]; g.pos[1] = v[1]; g.pos[2] = v[2];
        for (int c = 0; c < 3; ++c) g.rgb[c] = clamp01(0.5f + SH_C0 * v[3 + c]);
        g.opacity = 1.0f / (1.0f + expf(-v[6]));
        for (int c = 0; c < 3; ++c) g.scale[c] = expf(v[7 + c]);
        float qn = sqrtf(v[10] * v[10] + v[11] * v[11] + v[12] * v[12] + v[13] * v[13]);
        if (qn < 1e-12f) qn = 1.0f;
        for (int c = 0; c < 4; ++c) g.rot[c] = v[10 + c] / qn;
    }
    return true;
}
