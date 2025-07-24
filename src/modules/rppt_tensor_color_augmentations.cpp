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
#include "rppt_tensor_color_augmentations.h"
#include "cpu/host_tensor_color_augmentations.hpp"

#ifdef HIP_COMPILE
    #include "hip/hip_tensor_color_augmentations.hpp"
#endif // HIP_COMPILE

/******************** brightness ********************/

RppStatus rppt_brightness_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *alphaTensor,
                               Rpp32f *betaTensor,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        brightness_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     alphaTensor,
                                     betaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        brightness_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       alphaTensor,
                                       betaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        brightness_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       alphaTensor,
                                       betaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        brightness_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     alphaTensor,
                                     betaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** gamma_correction ********************/

RppStatus rppt_gamma_correction_host(RppPtr_t srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *gammaTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        gamma_correction_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           gammaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        gamma_correction_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        gamma_correction_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams,
                                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        gamma_correction_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           gammaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams,
                                           rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

NppStatus nppiGammaFwd_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI)
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
		
	    RpptROI *roiTensorPtrSrc;
	    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
	    RpptRoiType roiTypeSrc;
	    roiTypeSrc = RpptRoiType::XYWH;
		
	    roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

	    Rpp32f gammaVal = 1.9;
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
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

        hipHostFree(roiTensorPtrSrc);
	    hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiGammaFwd_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
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

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        Rpp32f gammaVal = 1.9;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
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

        hipHostFree(roiTensorPtrSrc);
        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiGammaFwd_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
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

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        Rpp32f gammaVal = 1.9;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
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

        hipHostFree(roiTensorPtrSrc);
        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiGammaFwd_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI)
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

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        Rpp32f gammaVal = 1.9;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
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

        hipHostFree(roiTensorPtrSrc);
        hipFree(d_input);
        hipFree(d_output);

        return(hipRppStatusTocudaNppStatus(status));
}
	
NppStatus nppiGammaFwd_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

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

	    Rpp8u *temp_in;
	    hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
	    hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
	    }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        Rpp32f gammaVal = 1.9;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        //Rpp8u *temp_output = pDst;
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
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipHostFree(roiTensorPtrSrc);
        hipFree(d_input);
        hipFree(d_output);
	    hipFree(temp_in);
	    hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}
	
NppStatus nppiGammaFwd_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
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

        Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        Rpp32f gammaVal = 1.9;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        //Rpp8u *temp_output = pDst;
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
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipHostFree(roiTensorPtrSrc);
        hipFree(d_input);
        hipFree(d_output);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiGammaFwd_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
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

        Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        Rpp32f gammaVal = 1.9;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        //Rpp8u *temp_output = pDst;
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
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipHostFree(roiTensorPtrSrc);
        hipFree(d_input);
        hipFree(d_output);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiGammaFwd_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;

        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;
		
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

        Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        Rpp32f gammaVal = 1.9;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status =rppt_gamma_correction_gpu((RppPtr_t)d_input, srcDescPtr, (RppPtr_t)d_output, dstDescPtr, &gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        //Rpp8u *temp_output = pDst;
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
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipHostFree(roiTensorPtrSrc);
        hipFree(d_input);
        hipFree(d_output);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** blend ********************/

RppStatus rppt_blend_host(RppPtr_t srcPtr1,
                          RppPtr_t srcPtr2,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *alphaTensor,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        blend_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                alphaTensor,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        blend_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  (Rpp16f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  alphaTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        blend_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  alphaTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams,
                                  rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        blend_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                alphaTensor,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams,
                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_twist ********************/

RppStatus rppt_color_twist_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *brightnessTensor,
                                Rpp32f *contrastTensor,
                                Rpp32f *hueTensor,
                                Rpp32f *saturationTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_twist_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      brightnessTensor,
                                      contrastTensor,
                                      hueTensor,
                                      saturationTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_twist_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_twist_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        brightnessTensor,
                                        contrastTensor,
                                        hueTensor,
                                        saturationTensor,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_twist_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                      srcDescPtr,
                                      static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                      dstDescPtr,
                                      brightnessTensor,
                                      contrastTensor,
                                      hueTensor,
                                      saturationTensor,
                                      roiTensorPtrSrc,
                                      roiType,
                                      layoutParams,
                                      rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

NppStatus nppiColorTwist32f_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
{
    int noOfImages = 1;
    int ip_channel = 1;
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

    srcDescPtr->offsetInBytes = 64;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfImages;
    srcDescPtr->h = oSizeROI.height;
    srcDescPtr->w = oSizeROI.width;
    srcDescPtr->c = ip_channel; // c = 1

    dstDescPtr->n = noOfImages;
    dstDescPtr->h = oSizeROI.height;
    dstDescPtr->w = oSizeROI.width;
    dstDescPtr->c = ip_channel; // c = 1

    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    RpptROI *roiTensorPtrSrc;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    roiTensorPtrSrc[0].xywhROI.xy.x = 0;
    roiTensorPtrSrc[0].xywhROI.xy.y = 0;
    roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
    roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

    srcDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.hStride = srcDescPtr->w * srcDescPtr->c;
    srcDescPtr->strides.wStride = srcDescPtr->c; // c = 1
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->h * dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.hStride = dstDescPtr->w * dstDescPtr->c;
    dstDescPtr->strides.wStride = dstDescPtr->c; // c = 1
    dstDescPtr->strides.cStride = 1;

    unsigned long long ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)noOfImages;
    unsigned long long oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)noOfImages;

    unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;

    Rpp8u *d_input, *d_output;
    hipMalloc(&d_input, ioBufferSizeInBytes_u8);
    hipMalloc(&d_output, oBufferSizeInBytes_u8);
    Rpp8u *offsetted_input = d_input + srcDescPtr->offsetInBytes;
    Rpp8u *temp_in = (Rpp8u *)pSrc;


    Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c; // c = 1
    for (int j = 0; j < oSizeROI.height; j++)
    {
        hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        temp_in += elementsInRow;
        offsetted_input += srcDescPtr->strides.hStride;
    }

    Rpp32f brightness = aTwist[0][3];
    Rpp32f contrast = aTwist[0][0];

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

    RppStatus status;
    status = rppt_color_twist_gpu_c1r(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, roiTensorPtrSrc, roiTypeSrc, handle);
    hipDeviceSynchronize();

    rppDestroyGPU(handle);

    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
    Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
    Rpp8u *offsetted_output = d_output + dstDescPtr->offsetInBytes;
    Rpp8u *temp_output = pDst;
    for (int k = 0; k < oSizeROI.height; k++)
    {
        hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof(Rpp8u), hipMemcpyDeviceToDevice);
        temp_output += elementsInRowout;
        offsetted_output += elementsInRowMax;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipHostFree(roiTensorPtrSrc);

    return hipRppStatusTocudaNppStatus(status);
}

NppStatus nppiColorTwist32f_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
	    Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
	    hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;


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
        
	    Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
	    hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
	    srcDescPtr->strides.hStride = srcDescPtr->w;
	    srcDescPtr->strides.wStride = 1;


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
        
	    Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
	    hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


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

        Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


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

        Rpp8u *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8u), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width, hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8s_C3R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8s_C3R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8s_C3IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp8s *temp_in = (Rpp8s *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;
		
        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8s *temp_output = pSrcDst;
	    for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8s_C3IR(Npp8s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp8s *temp_in = (Rpp8s *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;
		
        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp8s *temp_output = pSrcDst;
	    for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8s_P3R_Ctx(const Npp8s *const pSrc[3], int nSrcStep, Npp8s *const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

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

        Rpp8s *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp8s *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}
		
NppStatus nppiColorTwist32f_8s_P3R(const Npp8s *const pSrc[3], int nSrcStep, Npp8s *const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

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

        Rpp8s *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp8s *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8s_IP3R_Ctx(Npp8s *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;

        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

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

        Rpp8s *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp8s *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_8s_IP3R(Npp8s *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


        dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;

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

        Rpp8s *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
	    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;

        Rpp8s *temp_output;
        hipMalloc(&temp_output, oSizeROI.height * oSizeROI.width * ip_channel);
        for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output + (k * elementsInRowout), offsetted_output, elementsInRowout * sizeof (Rpp8s),hipMemcpyDeviceToDevice);
            //temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }
        for (int m = 0; m < ip_channel; m++)
        {
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp8s), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
	    for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof(Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pDst;
	    for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof(Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp16f *temp_in = (Rpp16f *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;
		
        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pSrcDst;
	    for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof(Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist32f_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp16f *temp_in = (Rpp16f *)pSrcDst;

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;
		
        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof(Rpp16f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->strides.hStride;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        rppDestroyGPU(handle);
        Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
        Rpp32u elementsInRowout = oSizeROI.width * dstDescPtr->c;
        offsetted_output = d_output + dstDescPtr->offsetInBytes;
        Rpp16f *temp_output = (Rpp16f *)pSrcDst;
	    for (int k = 0; k < oSizeROI.height; k++)
        {
            hipMemcpy(temp_output, offsetted_output, elementsInRowout * sizeof(Rpp16f),hipMemcpyDeviceToDevice);
            temp_output += elementsInRowout;
            offsetted_output += elementsInRowMax;
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        double gpu_time_used;

        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_P3R_Ctx(const Npp32f *const pSrc[3], int nSrcStep, Npp32f *const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


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

        Rpp32f *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_P3R(const Npp32f *const pSrc[3], int nSrcStep, Npp32f *const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


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

        Rpp32f *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
		for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrc[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_IP3R_Ctx(Npp32f *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


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

        Rpp32f *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiColorTwist_32f_IP3R(Npp32f *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4])
{
        int noOfImages = 1;
        int ip_channel = 3;
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
		
        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

        srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
        srcDescPtr->strides.hStride = srcDescPtr->w;
        srcDescPtr->strides.wStride = 1;


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

        Rpp32f *temp_in;
        hipMalloc(&temp_in, oSizeROI.height * oSizeROI.width * ip_channel);
	    for (int i = 0; i < ip_channel; i++)
        {
            hipMemcpy(temp_in + (i * oSizeROI.height * oSizeROI.width), pSrcDst[i], oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        Rpp32u elementsInRow = oSizeROI.width * srcDescPtr->c;

        for (int j = 0; j < oSizeROI.height; j++)
        {
            hipMemcpy(offsetted_input, temp_in, elementsInRow * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
            temp_in += elementsInRow;
            offsetted_input += srcDescPtr->w * srcDescPtr->c;
        }

        //RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
        Rpp32f brightness = aTwist[0][0];
        Rpp32f contrast = aTwist[0][1];
        Rpp32f hue = aTwist[0][2];
        Rpp32f saturation = aTwist[0][3];

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
		
        RppStatus status;
        status = rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, &brightness, &contrast, &hue, &saturation, roiTensorPtrSrc, roiTypeSrc, handle);
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
            hipMemcpy(pSrcDst[m],temp_output + (m * oSizeROI.height * oSizeROI.width), oSizeROI.height * oSizeROI.width * sizeof (Rpp32f), hipMemcpyDeviceToDevice);
        }

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(roiTensorPtrSrc);
        hipFree(temp_in);
        hipFree(temp_output);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** color_jitter ********************/

RppStatus rppt_color_jitter_host(RppPtr_t srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 RppPtr_t dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f *brightnessTensor,
                                 Rpp32f *contrastTensor,
                                 Rpp32f *hueTensor,
                                 Rpp32f *saturationTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_jitter_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       brightnessTensor,
                                       contrastTensor,
                                       hueTensor,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_jitter_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         brightnessTensor,
                                         contrastTensor,
                                         hueTensor,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_jitter_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         brightnessTensor,
                                         contrastTensor,
                                         hueTensor,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_jitter_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       brightnessTensor,
                                       contrastTensor,
                                       hueTensor,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

RppStatus rppt_color_cast_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptRGB *rgbTensor,
                               Rpp32f *alphaTensor,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_cast_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     rgbTensor,
                                     alphaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_cast_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_cast_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbTensor,
                                       alphaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams,
                                       rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_cast_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     rgbTensor,
                                     alphaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** exposure ********************/

RppStatus rppt_exposure_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *exposureFactorTensor,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        exposure_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   exposureFactorTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        exposure_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        exposure_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     exposureFactorTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        exposure_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   exposureFactorTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** contrast ********************/

RppStatus rppt_contrast_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *contrastFactorTensor,
                             Rpp32f *contrastCenterTensor,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        contrast_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   contrastFactorTensor,
                                   contrastCenterTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        contrast_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        contrast_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     contrastFactorTensor,
                                     contrastCenterTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams,
                                     rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        contrast_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   contrastFactorTensor,
                                   contrastCenterTensor,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

/******************** lut ********************/

RppStatus rppt_lut_host(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        RppPtr_t lutPtr,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
    if (srcDescPtr->dataType != RpptDataType::U8 && srcDescPtr->dataType != RpptDataType::I8)
        return RPP_ERROR_INVALID_SRC_DATATYPE;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        lut_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              static_cast<Rpp8u*>(lutPtr),
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        lut_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp16f*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               static_cast<Rpp16f*>(lutPtr),
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        lut_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp32f*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               static_cast<Rpp32f*>(lutPtr),
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        lut_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              static_cast<Rpp8s*>(lutPtr),
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** color_temperature ********************/

RppStatus rppt_color_temperature_host(RppPtr_t srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      RppPtr_t dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *adjustmentValueTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_temperature_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            adjustmentValueTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_temperature_f16_f16_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_temperature_f32_f32_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              adjustmentValueTensor,
                                              roiTensorPtrSrc,
                                              roiType,
                                              layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_temperature_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                            srcDescPtr,
                                            static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                            dstDescPtr,
                                            adjustmentValueTensor,
                                            roiTensorPtrSrc,
                                            roiType,
                                            layoutParams);
    }

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** brightness ********************/

RppStatus rppt_brightness_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              Rpp32f *alphaTensor,
                              Rpp32f *betaTensor,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(betaTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_brightness_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_brightness_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_brightness_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_brightness_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** gamma_correction ********************/

RppStatus rppt_gamma_correction_gpu(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *gammaTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(gammaTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_gamma_correction_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_gamma_correction_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_gamma_correction_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_gamma_correction_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         roiTensorPtrSrc,
                                         roiType,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** blend ********************/

RppStatus rppt_blend_gpu(RppPtr_t srcPtr1,
                         RppPtr_t srcPtr2,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32f *alphaTensor,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_blend_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                              static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_blend_tensor((half*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              (half*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_blend_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                              (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                              srcDescPtr,
                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_blend_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                              static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                              dstDescPtr,
                              roiTensorPtrSrc,
                              roiType,
                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** color_twist ********************/

RppStatus rppt_color_twist_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *brightnessTensor,
                               Rpp32f *contrastTensor,
                               Rpp32f *hueTensor,
                               Rpp32f *saturationTensor,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    Rpp32u paramIndex = 0;
    copy_param_float(brightnessTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(contrastTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(hueTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(saturationTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_twist_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_twist_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_twist_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_twist_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

RppStatus rppt_color_twist_gpu_c1r(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *brightnessTensor,
                               Rpp32f *contrastTensor,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c != 1 || dstDescPtr->c != 1)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    Rpp32u paramIndex = 0;
    copy_param_float(brightnessTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(contrastTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_twist_tensor_c1r(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_twist_tensor_c1r((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_twist_tensor_c1r((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_twist_tensor_c1r(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}



/******************** color_cast ********************/

RppStatus rppt_color_cast_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              RpptRGB *rgbTensor,
                              Rpp32f *alphaTensor,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_RpptRGB(rgbTensor, rpp::deref(rppHandle));

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_cast_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_cast_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_cast_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_cast_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** exposure ********************/

RppStatus rppt_exposure_gpu(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32f *exposureFactorTensor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(exposureFactorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_exposure_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_exposure_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_exposure_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_exposure_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** contrast ********************/

RppStatus rppt_contrast_gpu(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32f *contrastFactorTensor,
                            Rpp32f *contrastCenterTensor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(contrastFactorTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(contrastCenterTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_contrast_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_contrast_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_contrast_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_contrast_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** lut ********************/

RppStatus rppt_lut_gpu(RppPtr_t srcPtr,
                       RpptDescPtr srcDescPtr,
                       RppPtr_t dstPtr,
                       RpptDescPtr dstDescPtr,
                       RppPtr_t lutPtr,
                       RpptROIPtr roiTensorPtrSrc,
                       RpptRoiType roiType,
                       rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->dataType != RpptDataType::U8 && srcDescPtr->dataType != RpptDataType::I8)
        return RPP_ERROR_INVALID_SRC_DATATYPE;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                            dstDescPtr,
                            static_cast<Rpp8u*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                            dstDescPtr,
                            static_cast<half*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_lut_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                            dstDescPtr,
                            static_cast<Rpp32f*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_lut_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                            dstDescPtr,
                            static_cast<Rpp8s*>(lutPtr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** color_temperature ********************/

RppStatus rppt_color_temperature_gpu(RppPtr_t srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32s *adjustmentValueTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c != 3)
    {
        return RPP_ERROR_INVALID_CHANNELS;
    }

    Rpp32u paramIndex = 0;
    copy_param_int(adjustmentValueTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_color_temperature_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                          srcDescPtr,
                                          static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                          dstDescPtr,
                                          roiTensorPtrSrc,
                                          roiType,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_color_temperature_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          roiTensorPtrSrc,
                                          roiType,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_color_temperature_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          roiTensorPtrSrc,
                                          roiType,
                                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_color_temperature_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                          srcDescPtr,
                                          static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                          dstDescPtr,
                                          roiTensorPtrSrc,
                                          roiType,
                                          rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
