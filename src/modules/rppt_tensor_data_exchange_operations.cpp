/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_data_exchange_operations.h"
#include "cpu/host_tensor_data_exchange_operations.hpp"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_data_exchange_operations.hpp"
#endif // HIP_COMPILE

/******************** copy ********************/

RppStatus rppt_copy_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        copy_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               layoutParams,
                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        copy_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        copy_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 layoutParams,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        copy_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               layoutParams,
                               rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** swap_channels ********************/

RppStatus rppt_swap_channels_host(RppPtr_t srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        swap_channels_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        swap_channels_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          layoutParams,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        swap_channels_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          layoutParams,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        swap_channels_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_to_greyscale ********************/

RppStatus rppt_color_to_greyscale_host(RppPtr_t srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       RppPtr_t dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptSubpixelLayout srcSubpixelLayout,
                                       rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
        return RPP_ERROR_INVALID_SRC_CHANNELS;
    if (dstDescPtr->c != 1)
        return RPP_ERROR_INVALID_DST_CHANNELS;
    if (dstDescPtr->layout != RpptLayout::NCHW)
        return RPP_ERROR_INVALID_DST_LAYOUT;

    Rpp32f channelWeights[3];
    if (srcSubpixelLayout == RpptSubpixelLayout::RGBtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_RED;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_BLUE;
    }
    else if (srcSubpixelLayout == RpptSubpixelLayout::BGRtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_BLUE;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_RED;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_to_greyscale_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             channelWeights,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_to_greyscale_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               channelWeights,
                                               layoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_to_greyscale_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                               srcDescPtr,
                                               (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                               dstDescPtr,
                                               channelWeights,
                                               layoutParams,
                                               rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_to_greyscale_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                             srcDescPtr,
                                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                             dstDescPtr,
                                             channelWeights,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

NppStatus nppiRGBToGray_8u_C3C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

	    srcDescPtr->layout = RpptLayout::NHWC;
	    dstDescPtr->layout = RpptLayout::NCHW;
	    srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

	    srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = 1;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

	    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

	    unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

	    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

	    unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

	    Rpp8u *d_input,*d_output;

	    hipMalloc(&d_input,ioBufferSizeInBytes_u8);
	    hipMalloc(&d_output,oBufferSizeInBytes_u8);
	    Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        //offsetted_output = d_output + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

	    Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

	    for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

	    RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

        //RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        //RppiSize maxSize;
        //srcSize->width  = oSizeROI.width;
        //srcSize->height = oSizeROI.height;
        //maxSize.width  = oSizeROI.width;
        //maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status =rppt_color_to_greyscale_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, srcSubpixelLayout, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
	    Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
	    offsetted_output = d_output + dstDescPtr->offsetInBytes;
	    Rpp8u *temp_output = pDst;
	    for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        //free(srcSize);
        hipFree(d_input);
        //hipFree(d_input_second);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiRGBToGray_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = 1;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;

        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
	    Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

        //RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        //RppiSize maxSize;
        //srcSize->width  = oSizeROI.width;
        //srcSize->height = oSizeROI.height;
        //maxSize.width  = oSizeROI.width;
        //maxSize.height = oSizeROI.height;
        //oBufferSize = (unsigned long long)maxSize.height * (unsigned long long)maxSize.width * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        //status = rppi_add_u8_pkd3_batchPD_gpu(d_input, d_input_second, srcSize, maxSize, d_output, noOfImages, handle);
        status =rppt_color_to_greyscale_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, srcSubpixelLayout, handle);
        //hipMemcpy(pSrcDst, pSrc, oBufferSize * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        //free(srcSize);
        hipFree(d_input);
        //hipFree(d_input_second);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
	
NppStatus nppiRGBToGray_32f_C3C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = 1;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;

        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
	        temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_color_to_greyscale_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, srcSubpixelLayout, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiRGBToGray_32f_C3C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = 1;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;

        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
	    {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_color_to_greyscale_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, srcSubpixelLayout, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** copy ********************/

RppStatus rppt_copy_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_copy_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_copy_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_copy_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_copy_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

NppStatus nppiCopy_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
	    hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();
      
        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
	    }

	    hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
		
	    Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
	    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
		
	    Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

	    Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8s *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_i8);
        hipMalloc(&d_output,oBufferSizeInBytes_i8);
        Rpp8s *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8s *temp_in = (Rpp8s *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8s *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8s_C1R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8s *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_i8);
        hipMalloc(&d_output,oBufferSizeInBytes_i8);
        Rpp8s *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8s *temp_in = (Rpp8s *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8s *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8s_C3R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
	unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8s *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_i8);
        hipMalloc(&d_output,oBufferSizeInBytes_i8);
        Rpp8s *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8s *temp_in = (Rpp8s *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8s *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8s_C3R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8s *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_i8);
        hipMalloc(&d_output,oBufferSizeInBytes_i8);
        Rpp8s *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8s *temp_in = (Rpp8s *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8s *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;

        Rpp16f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f16);
        hipMalloc(&d_output,oBufferSizeInBytes_f16);
        Rpp16f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp16f *temp_in = (Rpp16f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *const aDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp8u *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(aDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *const aDst[3], int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp8u *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(aDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_C3P3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *const aDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp32f *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(aDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_C3P3R(const Npp32f *pSrc, int nSrcStep, Npp32f *const aDst[3], int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp32f *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(aDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8u_P3C3R_Ctx(const Npp8u *const aSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_i8);
        hipMalloc(&d_output,oBufferSizeInBytes_i8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	    Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), aSrc[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_8u_P3C3R(const Npp8u *const aSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_i8);
        hipMalloc(&d_output,oBufferSizeInBytes_i8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	    Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), aSrc[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_P3C3R_Ctx(const Npp32f *const aSrc[3], int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	    Rpp32f *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), aSrc[i], oSizeROI.height * oSizeROI.width * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiCopy_32f_P3C3R(const Npp32f *const aSrc[3], int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 0;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;
		
        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        
	    Rpp32f *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), aSrc[i], oSizeROI.height * oSizeROI.width * sizeof(Rpp32f), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}	

/******************** swap_channels ********************/

RppStatus rppt_swap_channels_gpu(RppPtr_t srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 RppPtr_t dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_swap_channels_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_swap_channels_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_swap_channels_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                      srcDescPtr,
                                      (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_swap_channels_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

NppStatus nppiSwapChannels_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3])
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

        Rpp8u *d_input,*d_output;
	    hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();
      
        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
	    }

	    hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSwapChannels_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
		
	    Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSwapChannels_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
		
	    Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pSrcDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSwapChannels_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3])
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
        srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
		
	    Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_u8);
        hipMalloc(&d_output,oBufferSizeInBytes_u8);
        Rpp8u *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp8u *temp_in = (Rpp8u *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8u *temp_output = pSrcDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8u),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSwapChannels_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
	    srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
		
NppStatus nppiSwapChannels_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const int aDstOrder[3])
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
	    srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
		unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrc;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSwapChannels_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
	    srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pSrcDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSwapChannels_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3])
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//pkd
        dstDescPtr->layout = RpptLayout::NHWC;
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;

        srcDescPtr->numDims = 4;
        dstDescPtr->numDims = 4;

        srcDescPtr->offsetInBytes = 64;
        dstDescPtr->offsetInBytes = 0;

        srcDescPtr->n = noOfImages;
        srcDescPtr->h = oSizeROI.height;
        srcDescPtr->w = oSizeROI.width;
        srcDescPtr->c = ip_channel;

        dstDescPtr->n = noOfImages;
        dstDescPtr->h = oSizeROI.height;
        dstDescPtr->w = oSizeROI.width;
        dstDescPtr->c = ip_channel;

    // Optionally set w stride as a multiple of 8 for src/dst

        srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
        dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
	    srcDescPtr->strides.wStride = srcDescPtr->c;
        srcDescPtr->strides.cStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
        dstDescPtr->strides.wStride = dstDescPtr->c;
        dstDescPtr->strides.cStride = 1;

        unsigned long long ioBufferSize = 0;
        unsigned long long oBufferSize = 0;

        ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
        oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

        unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
	    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,ioBufferSizeInBytes_f32);
        hipMalloc(&d_output,oBufferSizeInBytes_f32);
        Rpp32f *offsetted_input, *offsetted_output;
        offsetted_input = d_input + srcDescPtr->offsetInBytes;
        Rpp32f *temp_in = (Rpp32f *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp32f *temp_output = pSrcDst;
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp32f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** color_to_greyscale ********************/

RppStatus rppt_color_to_greyscale_gpu(RppPtr_t srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptSubpixelLayout srcSubpixelLayout,
                                      rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if (srcDescPtr->c != 3)
        return RPP_ERROR_INVALID_SRC_CHANNELS;
    if (dstDescPtr->c != 1)
        return RPP_ERROR_INVALID_DST_CHANNELS;
    if (dstDescPtr->layout != RpptLayout::NCHW)
        return RPP_ERROR_INVALID_DST_LAYOUT;

    Rpp32f channelWeights[3];
    if (srcSubpixelLayout == RpptSubpixelLayout::RGBtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_RED;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_BLUE;
    }
    else if (srcSubpixelLayout == RpptSubpixelLayout::BGRtype)
    {
        channelWeights[0] = RGB_TO_GREY_WEIGHT_BLUE;
        channelWeights[1] = RGB_TO_GREY_WEIGHT_GREEN;
        channelWeights[2] = RGB_TO_GREY_WEIGHT_RED;
    }

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_to_greyscale_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_to_greyscale_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_to_greyscale_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_to_greyscale_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           channelWeights,
                                           rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
