// Colour splash with NPP: finds the most common colour (hue) in a photo, keeps
// the pixels of that colour sharp and in colour, and turns everything else
// into a blurred grayscale background.
//
// Usage: imageColourNPP --input=photo.jpg [--output=out.jpg] [--range=12] [--blur=20] [--invert]
//   --range  how far (on NPP's 0-255 hue scale) a hue may be from the most common one
//   --blur   how many 15x15 Gaussian passes to apply to the background

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Pixels less saturated than this (white, black, grey) 
const int MIN_SATURATION = 60;

NppStreamContext makeStreamContext()
{
    NppStreamContext ctx = {};
    ctx.hStream = 0;
    checkCudaErrors(cudaGetDevice(&ctx.nCudaDeviceId));
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId));
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    checkCudaErrors(cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags));
    return ctx;
}

int main(int argc, char *argv[])
{
    try
    {
        findCudaDevice(argc, (const char **)argv);
        NppStreamContext ctx = makeStreamContext();

        // command line
        char *inputPath = nullptr, *outputPath = nullptr;
        getCmdLineArgumentString(argc, (const char **)argv, "input", &inputPath);
        std::string sFilename = inputPath ? inputPath : "data/Lena.png";
        std::string sResultFilename = sFilename.substr(0, sFilename.rfind('.')) + "_colour.jpg";
        if (getCmdLineArgumentString(argc, (const char **)argv, "output", &outputPath))
        {
            sResultFilename = outputPath;
        }
        int hueRange = checkCmdLineFlag(argc, (const char **)argv, "range")
                           ? getCmdLineArgumentInt(argc, (const char **)argv, "range") : 12;
        int blurPasses = checkCmdLineFlag(argc, (const char **)argv, "blur")
                             ? getCmdLineArgumentInt(argc, (const char **)argv, "blur") : 20;
        // --invert: blur and grey the most common colour instead, leave the rest alone
        bool invert = checkCmdLineFlag(argc, (const char **)argv, "invert");

        // 1\ load the photo on the host as 8-bit RGB
        int width, height, channels;
        Npp8u *pHostRGB = stbi_load(sFilename.c_str(), &width, &height, &channels, 3);
        if (pHostRGB == nullptr)
        {
            throw npp::Exception("Failed to load image <" + sFilename + ">: " + stbi_failure_reason());
        }
        NppiSize oSize = {width, height};
        int hostStep = width * 3;

        // 2\ (same width and type => same step) and upload the photo
        int step3, step1;
        Npp8u *pRGB = nppiMalloc_8u_C3(width, height, &step3);
        Npp8u *pHSV = nppiMalloc_8u_C3(width, height, &step3);
        Npp8u *pOut = nppiMalloc_8u_C3(width, height, &step3);
        Npp8u *pTmp = nppiMalloc_8u_C3(width, height, &step3);
        Npp8u *pGray = nppiMalloc_8u_C1(width, height, &step1);
        Npp8u *pMask = nppiMalloc_8u_C1(width, height, &step1);
        Npp8u *pHue = nppiMalloc_8u_C1(width, height, &step1);
        Npp8u *pSatMask = nppiMalloc_8u_C1(width, height, &step1);
        Npp8u *pWrap = nppiMalloc_8u_C1(width, height, &step1);
        checkCudaErrors(cudaMemcpy2D(pRGB, step3, pHostRGB, hostStep, hostStep, height, cudaMemcpyHostToDevice));

        // scratch for the histogram and the pixel counts (same buffer, used one at a time)
        size_t histBufSize, countBufSize;
        NPP_CHECK_NPP(nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(oSize, 257, &histBufSize, ctx));
        NPP_CHECK_NPP(nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx(oSize, &countBufSize, ctx));
        Npp8u *pScratch;
        int *pHistogram, *pCount;
        checkCudaErrors(cudaMalloc(&pScratch, std::max(histBufSize, countBufSize)));
        checkCudaErrors(cudaMalloc(&pHistogram, 256 * sizeof(int)));
        checkCudaErrors(cudaMalloc(&pCount, sizeof(int)));
        auto countInRange = [&](const Npp8u *pSrc, Npp8u lo, Npp8u hi)
        {
            int count;
            NPP_CHECK_NPP(nppiCountInRange_8u_C1R_Ctx(pSrc, step1, oSize, pCount, lo, hi, pScratch, ctx));
            checkCudaErrors(cudaMemcpy(&count, pCount, sizeof(int), cudaMemcpyDeviceToHost));
            return count;
        };

        // 3\ to HSV, then a 256-bin hue histogram of the saturated pixels, all on the GPU
        NPP_CHECK_NPP(nppiRGBToHSV_8u_C3R_Ctx(pRGB, step3, pHSV, step3, oSize, ctx));
        NPP_CHECK_NPP(nppiCopy_8u_C3C1R_Ctx(pHSV, step3, pHue, step1, oSize, ctx));
        NPP_CHECK_NPP(nppiCopy_8u_C3C1R_Ctx(pHSV + 1, step3, pGray, step1, oSize, ctx));
        NPP_CHECK_NPP(nppiCompareC_8u_C1R_Ctx(pGray, step1, MIN_SATURATION, pSatMask, step1, oSize,
                                              NPP_CMP_GREATER_EQ, ctx));
        // unsaturated pixels get hue 0, and are taken back out of bin 0 below
        NPP_CHECK_NPP(nppiAnd_8u_C1R_Ctx(pHue, step1, pSatMask, step1, pGray, step1, oSize, ctx));
        NPP_CHECK_NPP(nppiHistogramEven_8u_C1R_Ctx(pGray, step1, oSize, pHistogram, 257, 0, 256, pScratch, ctx));
        int histogram[256];
        checkCudaErrors(cudaMemcpy(histogram, pHistogram, sizeof(histogram), cudaMemcpyDeviceToHost));
        histogram[0] -= countInRange(pSatMask, 0, 0);
        int commonHue = (int)(std::max_element(histogram, histogram + 256) - histogram);

        // 4\ mask 255 where a pixel's hue is close to the common hue (hue wraps around,
        //    so a distance d >= 256 - range is close too), on the GPU
        NPP_CHECK_NPP(nppiAbsDiffC_8u_C1R_Ctx(pHue, step1, pGray, step1, oSize, (Npp8u)commonHue, ctx));
        NPP_CHECK_NPP(nppiCompareC_8u_C1R_Ctx(pGray, step1, (Npp8u)std::min(hueRange, 255), pMask, step1, oSize,
                                              NPP_CMP_LESS_EQ, ctx));
        if (hueRange > 0)
        {
            NPP_CHECK_NPP(nppiCompareC_8u_C1R_Ctx(pGray, step1, (Npp8u)std::max(256 - hueRange, 0), pWrap, step1,
                                                  oSize, NPP_CMP_GREATER_EQ, ctx));
            NPP_CHECK_NPP(nppiOr_8u_C1IR_Ctx(pWrap, step1, pMask, step1, oSize, ctx));
        }
        NPP_CHECK_NPP(nppiAnd_8u_C1IR_Ctx(pSatMask, step1, pMask, step1, oSize, ctx));
        if (invert)
        {
            NPP_CHECK_NPP(nppiNot_8u_C1IR_Ctx(pMask, step1, oSize, ctx));
        }
        size_t keptPixels = countInRange(pMask, 255, 255);
        size_t numPixels = (size_t)width * height;

        // 5\ back to 3 identical channels, then blur repeatedly
        NPP_CHECK_NPP(nppiRGBToGray_8u_C3C1R_Ctx(pRGB, step3, pGray, step1, oSize, ctx));
        NPP_CHECK_NPP(nppiDup_8u_C1C3R_Ctx(pGray, step1, pOut, step3, oSize, ctx));
        for (int i = 0; i < blurPasses; ++i)
        {
            NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C3R_Ctx(pOut, step3, oSize, {0, 0}, pTmp, step3, oSize,
                                                           NPP_MASK_SIZE_15_X_15, NPP_BORDER_REPLICATE, ctx));
            std::swap(pOut, pTmp);
        }

        // 6\ paste the originalcolour pixels back wherever the mask is set
        NPP_CHECK_NPP(nppiCopy_8u_C3MR_Ctx(pRGB, step3, pOut, step3, oSize, pMask, step1, ctx));

        // 7\ download the result and save it as JPEG
        checkCudaErrors(cudaMemcpy2D(pHostRGB, hostStep, pOut, step3, hostStep, height, cudaMemcpyDeviceToHost));
        if (!stbi_write_jpg(sResultFilename.c_str(), width, height, 3, pHostRGB, 95))
        {
            throw npp::Exception("Failed to save result image <" + sResultFilename + ">");
        }

        std::cout << "Most common hue: " << commonHue * 360 / 256 << " degrees, kept "
                  << 100.0 * keptPixels / numPixels << "% of the pixels in colour" << std::endl;
        std::cout << "Saved image: " << sResultFilename << std::endl;

        for (Npp8u *p : {pRGB, pHSV, pOut, pTmp, pGray, pMask, pHue, pSatMask, pWrap})
        {
            nppiFree(p);
        }
        cudaFree(pScratch);
        cudaFree(pHistogram);
        cudaFree(pCount);
        stbi_image_free(pHostRGB);
        return EXIT_SUCCESS;
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        return EXIT_FAILURE;
    }
}
