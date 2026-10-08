################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################
#
# Builds with nvcc on Linux, and on Windows (GNU make run from Git Bash/MSYS)
# with the MSVC host compiler located automatically through vswhere.
#
################################################################################

# Define directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
LIB_DIR = lib
INCLUDE_DIR = include

# GPU architecture to generate code for (86 = RTX 30xx / Ampere)
SM ?= 86

ifeq ($(OS),Windows_NT)
    # use Git for Windows' sh so the recipes also work when make is started
    # from PowerShell or cmd
    SHELL := C:/Program Files/Git/usr/bin/sh.exe
    export PATH := C:/Program Files/Git/usr/bin;$(PATH)
    EXE = .exe
    NVCC ?= nvcc
    # nvcc needs cl.exe; find the newest MSVC toolset unless CCBIN is given
    ifeq ($(origin CCBIN),undefined)
        VSWHERE = C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe
        VS_PATH := $(subst \,/,$(shell "$(VSWHERE)" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath))
        MSVC_VER := $(strip $(shell cat "$(VS_PATH)/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"))
        CCBIN := $(VS_PATH)/VC/Tools/MSVC/$(MSVC_VER)/bin/Hostx64/x64
    endif
else
    EXE =
    CUDA_PATH ?= /usr/local/cuda
    NVCC ?= $(or $(shell command -v nvcc),$(CUDA_PATH)/bin/nvcc)
endif

# Define the compiler and flags
NVCCFLAGS = -std=c++17 -arch=sm_$(SM) -I$(INCLUDE_DIR)
ifneq ($(CCBIN),)
    NVCCFLAGS += -ccbin "$(CCBIN)"
endif
LDFLAGS = -lcudart -lnppc -lnppisu -lnppig -lnppidei -lnppicc -lnppif -lnppial -lnppist -lnppitc

HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)
ROTATE = $(BIN_DIR)/imageRotationNPP$(EXE)
COLOUR = $(BIN_DIR)/imageColourNPP$(EXE)

INPUT ?= $(DATA_DIR)/Lena.png

# 2D photo -> 3D point cloud, built on Linux/WSL only. Needs OpenCV for image I/O
# (sudo apt install libopencv-dev pkg-config) and ONNX Runtime GPU for the depth
# model (make ort: ONNX Runtime + CUDA 12 / cuDNN 9 libraries, all into lib/ort)
SPLAT = $(BIN_DIR)/splat$(EXE)
ORT_VERSION = 1.30.0
ORT_URL = https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-linux-x64-gpu_cuda12-$(ORT_VERSION).tgz
ORT_DIR = $(LIB_DIR)/ort
ORT_LIB = $(ORT_DIR)/lib/libonnxruntime.so
CUDA12_WHEELS = nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 \
                nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
# Depth models (relative inverse depth, ONNX). Depth Anything V2 Small (Apache-2.0) is the
# default: much sharper edges than MiDaS v2.1 small, which is kept for comparison (MODEL=midas.onnx).
DA2_URL = https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx
DA2_MODEL = depth_anything_v2_small.onnx
MIDAS_URL = https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx
MIDAS_MODEL = midas.onnx
MODEL ?= $(DA2_MODEL)
MAX_POINTS ?= 500000
# far/near ratio of the generated scene, and the relative depth jump at which a point counts
# as sitting on an object edge and is dropped (0 = keep all)
DEPTH_RANGE ?= 10
EDGE ?= 0.1
PLY ?= $(DATA_DIR)/ply/$(notdir $(basename $(INPUT))).ply
PORT ?= 8080
# On Windows the splat targets are forwarded to this WSL distro
WSL_DISTRO ?= Ubuntu

# A Windows path forwarded from PowerShell (INPUT=D:/images/x.jpg) becomes /mnt/d/images/x.jpg
ifneq ($(OS),Windows_NT)
ifneq ($(findstring :,$(INPUT)),)
override INPUT := $(shell wslpath -u '$(INPUT)')
endif
endif

.PHONY: all run run-colour splat ort ort-links model run-splat serve server clean help

# Define the default rule
all: $(ROTATE) $(COLOUR)

# Each program is a single source file: bin/<name> from src/<name>.cpp
$(BIN_DIR)/%$(EXE): $(SRC_DIR)/%.cpp $(HEADERS)
	mkdir -p $(BIN_DIR)
	"$(NVCC)" $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

# Build if needed, then rotate INPUT
run: $(ROTATE)
	./$(ROTATE) --input=$(INPUT) --output=$(basename $(INPUT))_rotate.pgm

# Build if needed, then keep INPUT's most common colour and blur the rest
run-colour: $(COLOUR)
	./$(COLOUR) --input=$(INPUT) --output=$(basename $(INPUT))_colour.jpg

ifeq ($(OS),Windows_NT)

# splat needs Linux (OpenCV, ONNX Runtime GPU), so on Windows these targets rerun
# make inside WSL in the same folder, passing on command-line variables (INPUT=..., PORT=...)
WSL_DIR := $(shell echo "$(CURDIR)" | sed -E 's|^([A-Za-z]):|/mnt/\L\1|; s|^/([a-z])/|/mnt/\1/|')

splat ort ort-links model run-splat serve server:
	MSYS_NO_PATHCONV=1 wsl.exe -d $(WSL_DISTRO) --cd "$(WSL_DIR)" -- make $@ $(MAKEOVERRIDES)

else

splat: $(SPLAT)

$(SPLAT): $(SRC_DIR)/splat.cu $(ORT_LIB)
	mkdir -p $(BIN_DIR)
	"$(NVCC)" $(NVCCFLAGS) -diag-suppress 611 -I$(ORT_DIR)/include $< -o $@ \
	    $$(pkg-config opencv4 --cflags --libs) -lcudart -L$(ORT_DIR)/lib -lonnxruntime \
	    -Xlinker --disable-new-dtags -Xlinker -rpath -Xlinker $(abspath $(ORT_DIR)/lib)

# ONNX Runtime GPU plus the CUDA 12 and cuDNN 9 libraries its CUDA provider loads.
# Everything goes into one folder so the provider finds them next to itself.
ort: $(ORT_LIB)

$(ORT_LIB):
	rm -rf $(ORT_DIR) $(ORT_DIR).tmp && mkdir -p $(ORT_DIR).tmp/wheels
	curl -L --fail $(ORT_URL) | tar -xz -C $(ORT_DIR).tmp
	python3 -m pip download --quiet --no-deps --only-binary=:all: -d $(ORT_DIR).tmp/wheels $(CUDA12_WHEELS)
	for w in $(ORT_DIR).tmp/wheels/*.whl; do python3 -m zipfile -e "$$w" $(ORT_DIR).tmp/cuda; done
	mv $(ORT_DIR).tmp/onnxruntime-linux-x64-gpu* $(ORT_DIR)
	find $(ORT_DIR).tmp/cuda -name '*.so*' -exec cp {} $(ORT_DIR)/lib/ \;
	rm -rf $(ORT_DIR).tmp
	$(MAKE) ort-links

# the wheels ship only libX.so.N, but ONNX Runtime also dlopens plain libX.so
ort-links:
	cd $(ORT_DIR)/lib && for f in libcu*.so.* libnv*.so.*; do \
	    [ -e "$${f%%.so.*}.so" ] || ln -s "$$f" "$${f%%.so.*}.so"; done

# Download the depth models (once); .part keeps a failed download from looking finished
model: $(MODEL)

$(DA2_MODEL):
	curl -L --fail -o $@.part $(DA2_URL)
	mv $@.part $@

$(MIDAS_MODEL):
	curl -L --fail -o $@.part $(MIDAS_URL)
	mv $@.part $@

# Build and download if needed, then turn INPUT into PLY (default data/ply/<input name>.ply)
run-splat: $(SPLAT) $(MODEL)
	mkdir -p $(dir $(PLY))
	./$(SPLAT) $(INPUT) $(PLY) $(MODEL) $(MAX_POINTS) $(DEPTH_RANGE) $(EDGE)

# View the point clouds: serves the repo root so src/index.html can list data/ply/ (Space = next)
serve:
	@echo "Open http://localhost:$(PORT)/src/index.html  (Ctrl+C stops the server)"
	python3 -m http.server $(PORT)

server: serve

endif

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Build, then rotate INPUT (default $(INPUT))."
	@echo "  make run-colour - Build, then colour-splash INPUT."
	@echo "Point clouds (run in WSL $(WSL_DISTRO); from Windows these are forwarded to WSL automatically):"
	@echo "  make ort    - Download ONNX Runtime GPU + CUDA 12/cuDNN 9 libs to $(ORT_DIR)."
	@echo "  make model  - Download the depth model MODEL (default $(DA2_MODEL))."
	@echo "  make run-splat INPUT=<photo> - Build splat, then turn INPUT into data/ply/<name>.ply."
	@echo "                 Also MODEL, DEPTH_RANGE, EDGE, MAX_POINTS, PLY."
	@echo "  make server - View the point clouds at http://localhost:$(PORT)/src/index.html (alias: serve)"
	@echo "  make clean  - Clean up the build files."
	@echo "  make help   - Display this help message."
	@echo "Variables: SM=86 (GPU arch), CCBIN=<dir of cl.exe> (Windows), CUDA_PATH (Linux)."
