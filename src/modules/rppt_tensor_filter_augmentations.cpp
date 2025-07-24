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
#include "rppt_tensor_filter_augmentations.h"

#ifdef HIP_COMPILE
#include <hip/hip_fp16.h>
#include "hip/hip_tensor_filter_augmentations.hpp"
#endif // HIP_COMPILE

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** box_filter ********************/

RppStatus rppt_box_filter_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr,
                              Rpp32u kernelSize,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2))
        return RPP_ERROR_LOW_OFFSET;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_box_filter_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   kernelSize,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_box_filter_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   kernelSize,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_box_filter_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                   srcDescPtr,
                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                   dstDescPtr,
                                   kernelSize,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_box_filter_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   kernelSize,
                                   roiTensorPtrSrc,
                                   roiType,
                                   rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

NppStatus nppiFilterBox_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, 
				NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor)
{
        //pkd
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
	    Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
	    //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
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

/*NppStatus nppiFilterBox_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
{
        //pln1
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
        //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiFilterBox_8u_C1R_Ctx is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - FilterBox_8u : " << gpu_time_used;
        printf("\n");

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
}*/

/*NppStatus nppiFilterBox_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor)
{
        //pln1
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
        roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        RppStatus status;
        start = clock();
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
        //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiFilterBox_8u_C1R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU Time - FilterBox_8u : " << gpu_time_used;
        printf("\n");

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
}*/

NppStatus nppiFilterBox_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
{
        //pkd
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
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
        //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
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
	
NppStatus nppiFilterBox_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
{
        //pln1
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
	    roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
        //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
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

NppStatus nppiFilterBox_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor)
{
        //pln1
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
        RpptROI *roiTensorPtrSrc;
        hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));

        RpptRoiType roiTypeSrc;
	    roiTypeSrc = RpptRoiType::XYWH;

        roiTensorPtrSrc[0].xywhROI.xy.x = 0;
        roiTensorPtrSrc[0].xywhROI.xy.y = 0;
        roiTensorPtrSrc[0].xywhROI.roiWidth = oSizeROI.width;
        roiTensorPtrSrc[0].xywhROI.roiHeight = oSizeROI.height;

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
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
        //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
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

NppStatus nppiFilterBox_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx)
{
        //pkd
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
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
        //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
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

NppStatus nppiFilterBox_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor)
{
        //pkd
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
        Rpp32u kernelSize = oMaskSize.width;

        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);

        RppStatus status;
        status = rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
        //rppi_box_filter_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, kernelSize, noOfImages, handle);
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

/******************** gaussian_filter ********************/

RppStatus rppt_gaussian_filter_gpu(RppPtr_t srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   RppPtr_t dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *stdDevTensor,
                                   Rpp32u kernelSize,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcDescPtr->offsetInBytes < 12 * (kernelSize / 2))
        return RPP_ERROR_LOW_OFFSET;

    Rpp32u paramIndex = 0;
    copy_param_float(stdDevTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_gaussian_filter_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        kernelSize,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_gaussian_filter_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        kernelSize,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_gaussian_filter_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        kernelSize,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_gaussian_filter_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        kernelSize,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/*NppStatus nppiFilterGauss_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, 
				NppiMaskSize eMaskSize)
{
        int noOfImages = 1;
        int ip_channel = 3;//pkd_3
        RppiSize *srcSize = (RppiSize *)calloc(noOfImages, sizeof(RppiSize));
        RppiSize maxSize;
        srcSize->width  = oSizeROI.width;
        srcSize->height = oSizeROI.height;
        maxSize.width  = oSizeROI.width;
        maxSize.height = oSizeROI.height;
	unsigned int kernelSize;
	float stdDev;
	RppStatus status;
		
	if((eMaskSize == NPP_MASK_SIZE_1_X_3) || (eMaskSize == NPP_MASK_SIZE_1_X_5) || (eMaskSize == NPP_MASK_SIZE_3_X_1) || (eMaskSize == NPP_MASK_SIZE_5_X_1))
	{
		status = RPP_ERROR_INVALID_ARGUMENTS;
		return(hipRppStatusTocudaNppStatus(status));
	}else if(eMaskSize ==  NPP_MASK_SIZE_3_X_3)
	{
		kernelSize = 3;
		stdDev = 0.4 + (kernelSize/2) * 0.6;
	}else if(eMaskSize == NPP_MASK_SIZE_5_X_5)
	{
		kernelSize = 5;
		stdDev = 0.4 + (kernelSize/2) * 0.6;
	}else if(eMaskSize == NPP_MASK_SIZE_7_X_7)
	{
		kernelSize = 7;
		stdDev = 0.4 + (kernelSize/2) * 0.6;
	}else if(eMaskSize == NPP_MASK_SIZE_9_X_9)
	{
		kernelSize = 9;
		stdDev = 0.4 + (kernelSize/2) * 0.6;
	}else if(eMaskSize == NPP_MASK_SIZE_11_X_11)
	{
		kernelSize = 11;
		stdDev = 0.4 + (kernelSize/2) * 0.6;
	}else if(eMaskSize == NPP_MASK_SIZE_13_X_13)
	{
		kernelSize = 13;
		stdDev = 0.4 + (kernelSize/2) * 0.6;
	}else if(eMaskSize == NPP_MASK_SIZE_15_X_15)
	{
		kernelSize = 15;
		stdDev = 0.4 + (kernelSize/2) * 0.6;
	}
		
        rppHandle_t handle;
        hipStream_t stream;
        hipStreamCreate(&stream);
        rppCreateWithStreamAndBatchSize(&handle, stream, noOfImages);
        clock_t start, end;
        double gpu_time_used;

        start = clock();
	status = rppi_gaussian_filter_u8_pkd3_batchPD_gpu((RppPtr_t)pSrc, srcSize, maxSize, (RppPtr_t)pDst, &stdDev, &kernelSize, noOfImages, handle);
        hipDeviceSynchronize();
        end = clock();

        printf("\nnppiFilterGauss_8u_C3R is %d\n", status);
        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        std::cout << "\nGPU FilterGauss_8u -  : " << gpu_time_used;
        printf("\n");
        rppDestroyGPU(handle);
        free(srcSize);

        return(hipRppStatusTocudaNppStatus(status));
}*/										

#endif // GPU_SUPPORT
