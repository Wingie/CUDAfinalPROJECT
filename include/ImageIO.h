/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


// Modified from the cuda-samples original: FreeImage replaced with the
// header-only stb_image (loading) and a minimal binary PGM writer (saving),
// so no external image library has to be installed or linked.

#ifndef NV_UTIL_NPP_IMAGE_IO_H
#define NV_UTIL_NPP_IMAGE_IO_H

#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "Exceptions.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include <cstdio>
#include <string>
#include "string.h"

namespace npp
{
    // Load an image from disk (PNG, PGM, JPEG, BMP, ...), converting it to
    // 8-bit gray-scale.
    void
    loadImage(const std::string &rFileName, ImageCPU_8u_C1 &rImage)
    {
        int nWidth, nHeight, nChannels;
        unsigned char *pPixels = stbi_load(rFileName.c_str(), &nWidth, &nHeight, &nChannels, 1);
        if (pPixels == 0)
        {
            throw npp::Exception(std::string("Failed to load image <") + rFileName + ">: " + stbi_failure_reason(),
                                 __FILE__, __LINE__);
        }

        // create an ImageCPU to receive the loaded image data
        ImageCPU_8u_C1 oImage(nWidth, nHeight);

        const Npp8u *pSrcLine = pPixels;
        Npp8u *pDstLine = oImage.data();
        unsigned int nDstPitch = oImage.pitch();

        for (int iLine = 0; iLine < nHeight; ++iLine)
        {
            memcpy(pDstLine, pSrcLine, nWidth * sizeof(Npp8u));
            pSrcLine += nWidth;
            pDstLine += nDstPitch;
        }

        stbi_image_free(pPixels);

        // swap the user given image with our result image, effecively
        // moving our newly loaded image data into the user provided shell
        oImage.swap(rImage);
    }

    // Save a gray-scale image to disk as binary PGM (P5).
    void
    saveImage(const std::string &rFileName, const ImageCPU_8u_C1 &rImage)
    {
        FILE *pFile = fopen(rFileName.c_str(), "wb");
        NPP_ASSERT_MSG(pFile != 0, "Failed to open result image for writing.");

        fprintf(pFile, "P5\n%u %u\n255\n", rImage.width(), rImage.height());

        const Npp8u *pSrcLine = rImage.data();
        unsigned int nSrcPitch = rImage.pitch();
        bool bSuccess = true;

        for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
        {
            bSuccess &= fwrite(pSrcLine, sizeof(Npp8u), rImage.width(), pFile) == rImage.width();
            pSrcLine += nSrcPitch;
        }

        bSuccess &= fclose(pFile) == 0;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
    }

    // Load a gray-scale image from disk.
    void
    loadImage(const std::string &rFileName, ImageNPP_8u_C1 &rImage)
    {
        ImageCPU_8u_C1 oImage;
        loadImage(rFileName, oImage);
        ImageNPP_8u_C1 oResult(oImage);
        rImage.swap(oResult);
    }

    // Save an gray-scale image to disk.
    void
    saveImage(const std::string &rFileName, const ImageNPP_8u_C1 &rImage)
    {
        ImageCPU_8u_C1 oHostImage(rImage.size());
        // copy the device result data
        rImage.copyTo(oHostImage.data(), oHostImage.pitch());
        saveImage(rFileName, oHostImage);
    }
}


#endif // NV_UTIL_NPP_IMAGE_IO_H
