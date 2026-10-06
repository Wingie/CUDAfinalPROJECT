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
LDFLAGS = -lcudart -lnppc -lnppisu -lnppig -lnppidei -lnppicc -lnppif

HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)
ROTATE = $(BIN_DIR)/imageRotationNPP$(EXE)
COLOUR = $(BIN_DIR)/imageColourNPP$(EXE)

INPUT ?= $(DATA_DIR)/Lena.png

.PHONY: all run run-colour clean help

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

# Clean up
clean:
	rm -rf $(BIN_DIR)/*

# Help command
help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Build, then rotate INPUT (default $(INPUT))."
	@echo "  make run-colour - Build, then colour-splash INPUT."
	@echo "  make clean  - Clean up the build files."
	@echo "  make help   - Display this help message."
	@echo "Variables: SM=86 (GPU arch), CCBIN=<dir of cl.exe> (Windows), CUDA_PATH (Linux)."
