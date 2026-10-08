# CUDA Image Projects

Two GPU image programs built with CUDA on an RTX 3090:

1. **[Splat](#splat-2d-photo-to-3d-point-cloud)**, the final course project: turns a single 2D photo into a 3D point cloud
   ("splat") that you can spin around in the browser.
2. **[ImageColourNPP](#imagecolournpp-colour-splash)**: the previous final course projet "colour splash" filter built on NPP (NVIDIA Performance Primitives).

# Splat: 2D photo to 3D point cloud

`splat` estimates how far away every pixel is, then uses a custom CUDA kernel to place
each pixel in 3D space. Both steps run on the GPU. The result is saved as a PLY point
cloud and shown in a Three.js viewer (`src/index.html`), where every point is drawn as
a soft round "splat" and you can rotate and zoom with the mouse.

## How it works

1. **Load:** read the photo with OpenCV (`imread`, BGR).
2. **Estimate depth on the GPU:** run the [MiDaS](https://github.com/isl-org/MiDaS) v2.1
   small model (`midas.onnx`) with ONNX Runtime and its CUDA execution provider
   (cuDNN). The photo is resized to 256x256 and normalised with the ImageNet mean and
   standard deviation. MiDaS returns relative inverse depth (disparity), which is
   resized back to the photo's size. There is no CPU fallback: if CUDA can't be used,
   the program stops with an error.
3. **Turn disparity into depth:** normalise the disparity to 0..1 and invert it, which
   gives a depth between about 0.67 (near) and 2 (far). That range fits the viewer's camera.
4. **Upload:** copy the colour image and the depth map to the GPU.
5. **Back-project (custom CUDA kernel `generatePointCloud`):** one thread per output
   point. Each thread applies pinhole-camera maths
   (`x = (u - w/2) * z / f`, `y = (v - h/2) * z / f`, with `f = 0.8 * width`) and
   stores the position and the RGB colour. Only every `stride`-th pixel is used, so the
   cloud stays under 500,000 points and loads in a browser (a 24 MP photo would
   otherwise give 24 million points).
6. **Save:** download the points and write an ASCII PLY file.

The program prints the depth-model timings, the image size, the stride and the number of points.

## Requirements

`splat` is built and run on Linux or in **WSL** (Ubuntu 24.04 on Windows, where WSL
passes the GPU through):

```sh
sudo apt install nvidia-cuda-toolkit libopencv-dev pkg-config make curl python3-pip
make ort      # ONNX Runtime GPU 1.30.0 + CUDA 12 / cuDNN 9 runtime libraries into lib/ort (~3.2 GB, once)
```

OpenCV is used only to load and resize images. `make ort` downloads the prebuilt
ONNX Runtime GPU release, plus NVIDIA's CUDA 12 and cuDNN 9 libraries from their pip
wheels, all into `lib/ort/`. Nothing is installed system-wide, and `bin/splat` finds
the libraries through its RPATH.

## Running

From a WSL shell in the project folder (for example `/mnt/d/System32/cuda/CUDAfinalPROJECT`):

```sh
make model                                  # download the MiDaS model to midas.onnx (once, 64 MB)
make run-splat INPUT=data/Lena.png          # build bin/splat and write data/ply/Lena.ply
make run-splat INPUT=photo.jpg PLY=data/ply/other.ply  # choose the output file
make serve                                  # start a web server on port 8080
```

Then open **http://localhost:8080/src/index.html**. Browsers won't load a local
file from a page opened by double-clicking, so the page has to come from the server.
The viewer lists every `.ply` in `data/ply/`: press **Space** for the next point cloud and **Shift+Space** for the previous one. The file name is shown at the top left. To start at a particular file, add
`?ply=`, for example `http://localhost:8080/src/index.html?ply=IMG_1515.ply`.

You can also run the program directly:

```sh
./bin/splat <image> [output.ply] [model.onnx] [max_points]   # defaults: output.ply midas.onnx 500000
```

Example output for a 24-megapixel Canon EOS R8 photo on the RTX 3090:

```
MiDaS: ONNX Runtime 1.30.0 on CUDA
MiDaS inference (first run): 1124.06 ms
MiDaS inference (warm): 5.79425 ms
Image 6000x4000, stride 7, 490776 points
Saving 3D Splat to data/ply/IMG_1515.ply...
```

The first inference includes cuDNN start-up; after that the depth model takes about 6 ms.

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
Compiled executables (`splat` on Linux/WSL, `imageColourNPP`, `imageRotationNPP`), built by `make`.

```data/```
Sample input (`Lena.png`), example output, and the generated point clouds in `data/ply/` (not committed). `data/charlize/` holds the 100-image colour-splash batch run (results and logs).

```include/```
Headers: NPP C++ image helpers from the CUDA samples (`Image*.h`, `helper_cuda.h`, ...) and the single-header `stb_image` / `stb_image_write` libraries.

```lib/```
Third-party libraries that are not installed by the system package manager. `make ort` puts ONNX Runtime GPU and the CUDA 12 / cuDNN 9 libraries in `lib/ort/` (not committed).

```src/```
`splat.cu` (photo to point cloud), `index.html` (the 3D viewer), `imageColourNPP.cpp` (the colour-splash program) and `imageRotationNPP.cpp` (the rotation sample).

```Makefile```
Builds the programs on Windows and Linux. Also has the `model`, `run-splat` and `serve` targets for splat, and the `run` / `run-colour` targets for the NPP programs.

```run.sh```
Colour-splash batch runner described above.

```INSTALL```
Placeholder for installation notes. The requirements and build steps above cover installation for now.
