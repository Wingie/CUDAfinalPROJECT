# ImageColourNPP

GPU-accelerated "colour splash" for photographs, built on NVIDIA's NPP (NVIDIA
Performance Primitives) library with CUDA.

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
3. **Find the common colour:** convert RGB to HSV on the GPU (`nppiRGBToHSV_8u_C3R`), then
   build a 256-bin hue histogram. Pixels with saturation below 60 (white, black,
   grey) are skipped so that neutral backgrounds don't win. The fullest bin is the
   most common hue.
4. **Build the mask:** mark each pixel whose hue is within `--range` of the common
   hue. Hue wraps around, so red at 0 and red at 255 count as close. `--invert`
   flips the mask.
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

## Code Organization

```bin/```
Compiled executables (`imageColourNPP`, `imageRotationNPP`), built by `make`.

```data/```
Sample input (`Lena.png`) and example output. `data/charlize/` holds the 100-image batch run (source, results and logs).

```include/```
Headers: NPP C++ image helpers from the CUDA samples (`Image*.h`, `helper_cuda.h`, ...) and the single-header `stb_image` / `stb_image_write` libraries.

```lib/```
Third-party libraries that are not installed by the system package manager (currently empty).

```src/```
`imageColourNPP.cpp` (the colour-splash program) and `imageRotationNPP.cpp` (the rotation sample).

```Makefile```
Builds both programs on Windows and Linux, and provides the `run` / `run-colour` targets.

```run.sh```
Batch runner described above.

```INSTALL```
Placeholder for installation notes. The requirements and build steps above cover installation for now.
