# CUDA Image Projects

Two GPU image programs built with CUDA on an RTX 3090:

1. **[Splat](#splat-2d-photo-to-3d-gaussian-splats)**, the final course project: turns a single 2D photo into 3D Gaussian splats
   that you can spin around in the browser, rendered live by a CUDA web server.
2. **[ImageColourNPP](#imagecolournpp-colour-splash)**: the previous final course projet "colour splash" filter built on NPP (NVIDIA Performance Primitives).

# Splat: 2D photo to 3D Gaussian splats

`splat` estimates how far away every pixel is, then uses a custom CUDA kernel to turn
each pixel into a small 3D Gaussian lying on the surface it belongs to. `splat_server`
renders those Gaussians with CUDA and serves them to the browser, where you can orbit
around the scene with the mouse. Everything runs on the GPU: the depth model, making the
Gaussians, rasterising them, and encoding the frames as JPEG.

## How it works

### Making the Gaussians (`src/splat.cu`)

1. **Load:** read the photo with OpenCV (`imread`, BGR).
2. **Estimate depth on the GPU:** run [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
   Small (Apache-2.0, [ONNX export](https://huggingface.co/onnx-community/depth-anything-v2-small))
   with ONNX Runtime and its CUDA execution provider (cuDNN). The photo is resized so its
   short side is 518 pixels, keeping the aspect ratio (784x518 for a 3:2 photo), and
   normalised with the ImageNet mean and standard deviation. The model returns relative
   inverse depth (disparity), which is resized back to the photo's size. The older
   [MiDaS](https://github.com/isl-org/MiDaS) v2.1 small model (256x256) still works with
   `MODEL=midas.onnx`, but its depth edges are much blurrier. There is no CPU fallback:
   if CUDA can't be used, the program stops with an error.
3. **Turn disparity into depth:** the disparity has an unknown scale and shift, so real
   distances can't be recovered from one photo. It is normalised to 0..1 and mapped onto
   inverse depth, so depth runs from 1 (nearest) to `DEPTH_RANGE` (farthest, default 10).
   A small range squashes the scene into flat layers (an "embossed" look); a larger range
   pushes the background further back.
4. **Upload:** copy the colour image and the depth map to the GPU.
5. **Make the Gaussians (custom CUDA kernel `generateGaussians`):** one thread per sampled
   pixel. Only every `stride`-th pixel is used, so a scene has at most 500,000 Gaussians
   (a 24 MP photo would otherwise give 24 million).
   - Pinhole-camera maths puts the pixel in 3D:
     `x = (u - w/2) * z / f`, `y = (v - h/2) * z / f`, with `f = 0.8 * width`.
   - The neighbouring samples to the left/right and above/below give two tangent vectors
     of the surface; their cross product is the normal. The Gaussian becomes a flat disc
     in that plane (rotation stored as a quaternion), sized to 0.6x the distance to its
     neighbours so the discs overlap without gaps, and limited to 8:1 so surfaces seen
     edge-on don't turn into needles.
   - The depth model blurs depth across object outlines. Without a fix, those pixels
     become a "curtain" floating between the subject and the background. So each thread
     compares its depth with the 4 neighbours one depth-model pixel away; if the relative
     difference is larger than `EDGE` (default 0.1), the pixel is pushed back to the
     background depth (looking up to 3 depth-model pixels away) as a small disc facing
     the camera. Seen from the photo's viewpoint the outline stays filled; seen from the
     side nothing hangs in mid-air.
   - Gaussians are packed together with an `atomicAdd` counter.
6. **Save:** download the Gaussians and write a binary PLY in the layout used by 3D
   Gaussian Splatting (position, normal, colour as `f_dc`, opacity, `scale_*`, `rot_*`,
   plus plain `red green blue`), so other splat viewers can open it too. The photo's
   field of view is stored in a header comment. See `src/gaussians.h`.

The program prints the depth-model input size and timings, the image size, the stride,
and how many Gaussians were made and how many of them were on depth edges.

### Rendering them (`src/splat_server.cu`)

`splat_server` is a small web server and a CUDA rasteriser in one program. The browser
page (`src/viewer.html`) only shows images: when you move the camera it asks for a new
frame, which is rendered on the GPU and sent back as a JPEG. The rasteriser follows the
tile renderer of 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023):

1. **`preprocess` kernel:** one thread per Gaussian. Move it into camera space, build its
   3D covariance from rotation and scale (R S S^T R^T), and project that to a 2D ellipse
   on the screen (EWA splatting: J W Σ W^T J^T, with J the Jacobian of the perspective
   projection). Work out the 16x16-pixel tiles the ellipse overlaps (3 standard deviations).
2. **Scan:** prefix sum of the tile counts with CUB, to know where each Gaussian's
   entries go.
3. **`duplicateWithKeys` kernel:** one entry per Gaussian per tile it touches, with a
   64-bit key: tile number in the high 32 bits, depth in the low 32 bits.
4. **Sort:** CUB radix sort of the keys. Afterwards every tile's Gaussians are together,
   sorted front to back.
5. **`identifyTileRanges` kernel:** where each tile's list starts and ends.
6. **`renderTiles` kernel:** one thread block per tile, one thread per pixel. The tile's
   Gaussians are loaded into shared memory in batches of 256, and each pixel blends them
   front to back (C += c * alpha * T, T *= 1 - alpha). A pixel stops once it's opaque,
   and the block stops when all its pixels are.
7. **Encode:** nvJPEG compresses the frame straight from GPU memory; only the JPEG is
   copied back to the CPU.

For 490,000 Gaussians at 1280x720 the rendering takes about 2-3 ms and the JPEG encoding
under 1 ms on the RTX 3090 (the `X-Render-Ms` and `X-Encode-Ms` response headers, also
shown in the viewer).

| Request | Answer |
|---|---|
| `GET /` | the viewer page |
| `GET /list` | JSON list of `data/ply/*.ply` |
| `GET /load?ply=NAME` | loads the file onto the GPU; JSON with the count, field of view and middle depth |
| `GET /frame?w&h&px&py&pz&tx&ty&tz&fov` | JPEG seen from camera position `p` looking at `t` |

## Requirements

Both programs are built and run on Linux or in **WSL** (Ubuntu 24.04 on Windows, where
WSL passes the GPU through):

```sh
sudo apt install nvidia-cuda-toolkit libopencv-dev pkg-config make curl python3-pip
make ort      # ONNX Runtime GPU 1.30.0 + CUDA 12 / cuDNN 9 runtime libraries into lib/ort (~3.2 GB, once)
```

OpenCV is used only to load and resize images. `make ort` downloads the prebuilt
ONNX Runtime GPU release, plus NVIDIA's CUDA 12 and cuDNN 9 libraries from their pip
wheels, all into `lib/ort/`. Nothing is installed system-wide, and `bin/splat` finds
the libraries through its RPATH. `splat_server` only needs the CUDA toolkit (CUB and
nvJPEG come with it).

## Running

Run these from PowerShell / cmd / Git Bash on Windows, or from a WSL shell in the project folder. On Windows, `make` forwards the splat targets (`ort`, `model`, `splat`, `splat-server`, `run-splat`, `servecuda`, `servepy`) to WSL (`WSL_DISTRO=Ubuntu`), passing on your variables. Windows paths such as `INPUT=D:/images/photo.jpg` are converted to `/mnt/d/...`.

```sh
make model                                  # download Depth Anything V2 Small (once, 99 MB)
make run-splat INPUT=data/Lena.png          # build bin/splat and write data/ply/Lena.ply
make run-splat INPUT=photo.jpg PLY=data/ply/other.ply  # choose the output file
make run-splat INPUT=photo.jpg DEPTH_RANGE=5 EDGE=0.05  # flatter scene, stricter edge filter
make run-splat INPUT=photo.jpg MODEL=midas.onnx         # compare with MiDaS (downloaded on first use)
make servecuda                              # CUDA renderer + web server on port 8080 (alias: make server)
make servepy                                # the older three.js point viewer, served by python
```

| Variable      | Default                         | Meaning                                                         |
|---------------|---------------------------------|-----------------------------------------------------------------|
| `MODEL`       | `depth_anything_v2_small.onnx`  | Depth model (`midas.onnx` also works)                           |
| `DEPTH_RANGE` | `10`                            | Far/near ratio of the scene; larger = more depth                |
| `EDGE`        | `0.1`                           | Relative depth jump at which a pixel counts as an edge and is moved to the background; `0` turns it off |
| `MAX_POINTS`  | `500000`                        | Upper limit on the number of Gaussians                          |
| `PORT`        | `8080`                          | Port of `servecuda` / `servepy`                                 |

After `make servecuda`, open **http://localhost:8080/**. Each scene opens from where the
photo was taken, with the photo's field of view, so it first looks like the photo.

- **Left drag:** orbit, **right drag:** pan, **wheel:** zoom, **R:** back to the photo view
- **Space / Shift+Space:** next / previous file in `data/ply/`; `?ply=NAME.ply` picks the first one
- The label shows the file, the number of Gaussians, the CUDA render and JPEG times, and the frame rate.

`make servepy` serves the older three.js viewer at http://localhost:8080/src/index.html,
which draws the same files as plain points (it reads their `red green blue`).
Browsers block pages opened straight from disk (`file://`) from loading the files, so
both viewers have to be opened through their server.

You can also run the programs directly:

```sh
./bin/splat <image> [output.ply] [model.onnx] [max_points] [depth_range] [edge_threshold]
# defaults: output.ply depth_anything_v2_small.onnx 500000 10 0.1
./bin/splat_server [port] [ply_dir]          # defaults: 8080 data/ply
```

Example output for a 24-megapixel Canon EOS R8 photo on the RTX 3090:

```
Depth model: depth_anything_v2_small.onnx, ONNX Runtime 1.30.0 on CUDA, input 784x518
Depth inference (first run): 6108.09 ms
Depth inference (warm): 116.703 ms
Image 6000x4000, stride 7, depth range 1..10, kept 490776 of 490776 points (9209 on depth edges, moved to the background)
Saving 490776 Gaussians to data/ply/IMG_1515.ply...
```

The first inference includes cuDNN start-up; after that Depth Anything V2 Small takes
50-120 ms (MiDaS small about 7 ms, but with much blurrier depth).

### A whole photo library (batch)

`splat_batch.sh` (run in WSL) picks N random photos from every sub-folder of a photo
library and makes a Gaussian file for each, named `data/ply/<folder>__<photo>.ply`, so
you can step through them in the viewer with Space:

```sh
./splat_batch.sh [models_dir] [per_folder] [log_dir]   # defaults: /mnt/d/images/models 10 data/splat_logs
```

`<log_dir>/summary.csv` has one line per photo (folder, image, size, depth-model time,
Gaussians, edge points, seconds, status), next to the per-photo logs and `run.log`.

# ImageColourNPP: colour splash

GPU-accelerated "colour splash" for photographs, built on NVIDIA's NPP library with CUDA.

## Overview
`imageColourNPP` finds the most common colour in a photo, then uses it to split
the image in two:

- **Default mode:** pixels of the most common colour stay sharp and in full colour.
  Everything else is turned grey and heavily Gaussian-blurred, so the dominant
  colour stands out from a soft monochrome background.
- **Invert mode (`--invert`):** the opposite. The most common colour is greyed and
  blurred, and the rest of the image stays untouched.

In portraits the most common colour is usually skin and hair. The default mode
gives a "subject in colour, background in black and white with bokeh" look, and
the invert mode desaturates the subject while keeping clothing and scenery in colour.

The program was tested on my own photo archive: 100 full-resolution 24-megapixel
(6000x4000) Canon EOS R8 photos, processed in both modes on an RTX 3090 with no
failures, at about 2.2 s per image including JPEG decode and encode. Example
output is in `data/`.

## How it works

The work is split into 7 steps. The pixel-heavy steps run on the GPU through NPP:

1. **Load:** decode the JPEG/PNG on the host into 8-bit RGB (`stb_image`).
2. **Upload:** allocate pitched GPU buffers (`nppiMalloc_8u_C3/C1`) and copy the photo to the device.
3. **Find the common colour:** convert RGB to HSV (`nppiRGBToHSV_8u_C3R`), split out
   the hue and saturation channels (`nppiCopy_8u_C3C1R`), and build a 256-bin hue
   histogram on the GPU (`nppiHistogramEven_8u_C1R`). Pixels with saturation below 60
   (white, black, grey) are left out (`nppiCompareC_8u_C1R` + `nppiAnd_8u_C1R`) so that
   neutral backgrounds don't win. Only the 256 bin counts are downloaded, and the
   fullest bin is the most common hue.
4. **Build the mask:** on the GPU, mark each saturated pixel whose hue is within
   `--range` of the common hue (`nppiAbsDiffC_8u_C1R`, `nppiCompareC_8u_C1R`,
   `nppiOr`/`nppiAnd`). Hue wraps around, so red at 0 and red at 255 count as close.
   `--invert` flips the mask (`nppiNot_8u_C1IR`), and `nppiCountInRange_8u_C1R`
   counts the kept pixels.
5. **Grey and blur:** convert the photo to grayscale (`nppiRGBToGray_8u_C3C1R`),
   expand it back to 3 channels (`nppiDup_8u_C1C3R`), then apply `--blur` passes
   of a 15x15 Gaussian filter (`nppiFilterGaussBorder_8u_C3R`).
6. **Composite:** copy the original colour pixels back over the blurred image
   wherever the mask is set (`nppiCopy_8u_C3MR`).
7. **Save:** download the result and write it as a quality-95 JPEG (`stb_image_write`).

The program prints the common hue in degrees and the percentage of pixels kept in colour.

The repository also contains `imageRotationNPP`, an NPP image-rotation sample used as a starting point.

## Requirements

- An NVIDIA GPU and the CUDA Toolkit, including NPP (tested with CUDA 13.2 on an RTX 3090)
- GNU `make`
- **Windows:** Visual Studio with the "Desktop development with C++" workload,
  because `nvcc` needs `cl.exe`. The Makefile finds it through `vswhere`. Run
  `make` from Git Bash, PowerShell or cmd; Git for Windows must be installed.
- **Linux:** `g++` and CUDA in `/usr/local/cuda`, or set `CUDA_PATH`

## Building

```sh
make                 # builds bin/imageColourNPP(.exe) and bin/imageRotationNPP(.exe)
make SM=75           # build for a different GPU architecture (default 86 = RTX 30xx)
make clean           # remove the binaries
make help            # list targets and variables
```

On Windows you can point to a specific MSVC compiler with `make CCBIN="<folder containing cl.exe>"`.

## Running

### A single image

```sh
./bin/imageColourNPP --input=data/Lena.png --output=data/Lena_colour.jpg
./bin/imageColourNPP --input=data/Lena.png --output=data/Lena_colour_invert.jpg --invert
```

Or let make build and run it in one step:

```sh
make run-colour INPUT=data/Lena.png      # writes data/Lena_colour.jpg
```

| Option            | Default                    | Meaning                                                  |
|-------------------|----------------------------|----------------------------------------------------------|
| `--input=<file>`  | `data/Lena.png`            | JPEG, PNG, BMP, TGA, ... (anything `stb_image` can read)   |
| `--output=<file>` | `<input>_colour.jpg`       | Output JPEG                                              |
| `--range=<n>`     | `12`                       | Hue distance (0-255 scale, about 1.4° per step) still counted as the common colour |
| `--blur=<n>`      | `20`                       | Number of 15x15 Gaussian passes on the background        |
| `--invert`        | off                        | Grey and blur the common colour instead of the rest      |
| `--device=<n>`    | auto                       | Which CUDA GPU to use                                    |

Example output:

```
GPU Device 0: "Ampere" with compute capability 8.6

Most common hue: 29 degrees, kept 32.033% of the pixels in colour
Saved image: data/charlize/results/IMG_1515_after_colour.jpg
```

### A whole folder (batch)

`run.sh` processes the first N `.jpg` files in a folder in both modes, and keeps
the originals, the results and the logs together:

```sh
./run.sh [input_dir] [output_dir] [count]
./run.sh D:/images/models/charlize data/charlize 100
```

It produces:

```
<output_dir>/
  source/   <name>_before.jpg            copy of each original
  results/  <name>_after_colour.jpg      default mode
            <name>_after_invert.jpg      --invert mode
  logs/     run.log                      GPU, driver and nvcc info, one line per run, total time
            summary.csv                  image, mode, width, height, common_hue_deg, kept_percent, seconds, status
            <name>_<mode>.log            the program's full output for each run
```

The script exits with a non-zero status if any image fails.

# Code Organization

```bin/```
Compiled executables (`splat` and `splat_server` on Linux/WSL, `imageColourNPP`, `imageRotationNPP`), built by `make`.

```data/```
Sample input (`Lena.png`), example output, and the generated Gaussian files in `data/ply/` (not committed). `data/splat_logs/` holds the logs of the photo-library batch run, and `data/charlize/` the 100-image colour-splash batch run (results and logs).

```include/```
Headers: NPP C++ image helpers from the CUDA samples (`Image*.h`, `helper_cuda.h`, ...) and the single-header `stb_image` / `stb_image_write` libraries.

```lib/```
Third-party libraries that are not installed by the system package manager. `make ort` puts ONNX Runtime GPU and the CUDA 12 / cuDNN 9 libraries in `lib/ort/` (not committed).

```src/```
`splat.cu` (photo to 3D Gaussians), `splat_server.cu` (CUDA Gaussian rasteriser and web server), `gaussians.h` (the Gaussian struct and PLY reading/writing shared by both), `viewer.html` (the page `splat_server` serves), `index.html` (the older three.js point viewer), `imageColourNPP.cpp` (the colour-splash program) and `imageRotationNPP.cpp` (the rotation sample).

```Makefile```
Builds the programs on Windows and Linux. Also has the `ort`, `model`, `run-splat`, `servecuda` and `servepy` targets for splat, and the `run` / `run-colour` targets for the NPP programs.

```run.sh```
Colour-splash batch runner described above. `splat_batch.sh` is the batch runner for splat.

```INSTALL```
Placeholder for installation notes. The requirements and build steps above cover installation for now.
