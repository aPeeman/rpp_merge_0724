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
#include "rppt_tensor_statistical_operations.h"
#include "cpu/host_tensor_statistical_operations.hpp"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_statistical_operations.hpp"
#endif // HIP_COMPILE

/******************** tensor_sum ********************/

RppStatus rppt_tensor_sum_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t tensorSumArr,
                               Rpp32u tensorSumArrLength,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (tensorSumArrLength < srcDescPtr->n)      // sum of single channel
            return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }
    else if (srcDescPtr->c == 3)
    {
        if (tensorSumArrLength < srcDescPtr->n * 4)  // sum of each channel, and total sum of all 3 channels
            return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        tensor_sum_u8_u64_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp64u*>(tensorSumArr),
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        tensor_sum_f16_f32_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<Rpp32f*>(tensorSumArr),
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        tensor_sum_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                static_cast<Rpp32f*>(tensorSumArr),
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        tensor_sum_i8_i64_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp64s*>(tensorSumArr),
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** tensor_min ********************/

RppStatus rppt_tensor_min_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t minArr,
                               Rpp32u minArrLength,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (minArrLength < srcDescPtr->n)      // 1 min for each image
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (minArrLength < srcDescPtr->n * 4)  // min of each channel, and min of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        tensor_min_u8_u8_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(minArr),
                              minArrLength,
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        tensor_min_f16_f16_host((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 static_cast<Rpp16f*>(minArr),
                                 minArrLength,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        tensor_min_f32_f32_host((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 static_cast<Rpp32f*>(minArr),
                                 minArrLength,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        tensor_min_i8_i8_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(minArr),
                              minArrLength,
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** tensor_max ********************/

RppStatus rppt_tensor_max_host(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t maxArr,
                               Rpp32u maxArrLength,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (maxArrLength < srcDescPtr->n)      // 1 min for each image
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (maxArrLength < srcDescPtr->n * 4)  // min of each channel, and min of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        tensor_max_u8_u8_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8u*>(maxArr),
                              maxArrLength,
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        tensor_max_f16_f16_host((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 static_cast<Rpp16f*>(maxArr),
                                 maxArrLength,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        tensor_max_f32_f32_host((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 static_cast<Rpp32f*>(maxArr),
                                 maxArrLength,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        tensor_max_i8_i8_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                              srcDescPtr,
                              static_cast<Rpp8s*>(maxArr),
                              maxArrLength,
                              roiTensorPtrSrc,
                              roiType,
                              layoutParams);
    }

    return RPP_SUCCESS;
}


/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

/******************** tensor_sum ********************/
#ifdef HIP_COMPILE
RppStatus rppt_tensor_sum_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t tensorSumArr,
                              Rpp32u tensorSumArrLength,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (tensorSumArrLength < srcDescPtr->n)      // sum of single channel
            return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }
    else if (srcDescPtr->c == 3)
    {
        if (tensorSumArrLength < srcDescPtr->n * 4)  // sum of each channel, and total sum of all 3 channels
            return RPP_ERROR_NOT_ENOUGH_MEMORY;
    }
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_tensor_sum(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp64u*>(tensorSumArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        hip_exec_tensor_sum(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                            srcDescPtr,
                            static_cast<Rpp32f*>(tensorSumArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_tensor_sum(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                            srcDescPtr,
                            static_cast<Rpp32f*>(tensorSumArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        hip_exec_tensor_sum(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp64s*>(tensorSumArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

NppStatus nppiSum_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f aSum[3])
{
        //pkd
	    //bool reductionTypeCase = 1;
	    int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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
		
        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
	    Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
	    Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
	    Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n
			
	    int bitDepthByteSize = 0;
	    if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
		    bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
	    else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
		    bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
	    hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
	    Rpp8u *d_input,*d_output;
	    hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
	
        aSum[0] = (double)reductionFuncResultArr[0];
	    aSum[1] = (double)reductionFuncResultArr[1];
	    aSum[2] = (double)reductionFuncResultArr[2];
	  
        rppDestroyGPU(handle);
        
        hipFree(d_input);
        hipFree(d_output);
	    hipHostFree(reductionFuncResultArr);
	    hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pSum, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
		
	    int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
		
	//reductionFuncResultArrLength = srcDescPtr->n;
        RppStatus status;
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        pSum[0] = (double)reductionFuncResultArr[0];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pSum)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
		
	    int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
		
	//reductionFuncResultArrLength = srcDescPtr->n;
        RppStatus status;
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        pSum[0] = (double)reductionFuncResultArr[0];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f aSum[3], NppStreamContext nppStreamCtx)
{
        //pkd
        //bool reductionTypeCase = 1;
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n
		
	    int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        aSum[0] = (double)reductionFuncResultArr[0];
        aSum[1] = (double)reductionFuncResultArr[1];
        aSum[2] = (double)reductionFuncResultArr[2];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** tensor_min ********************/

RppStatus rppt_tensor_min_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t imageMinArr,
                              Rpp32u imageMinArrLength,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (imageMinArrLength < srcDescPtr->n)   // min of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageMinArrLength < srcDescPtr->n * 4)   // min of each channel, and overall min of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_tensor_min(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8u*>(imageMinArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        hip_exec_tensor_min((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                            srcDescPtr,
                            static_cast<half*>(imageMinArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_tensor_min((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                            srcDescPtr,
                            static_cast<Rpp32f*>(imageMinArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        hip_exec_tensor_min(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8s*>(imageMinArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

NppStatus nppiMin_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u aMin[3])
{
        //pkd
	    //bool reductionTypeCase = 1;
	    int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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
		
        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
	    Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
	    Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
	    Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n
			
	    int bitDepthByteSize = 0;
	    if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
		    bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
	    else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
		    bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
	    hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
	    Rpp8u *d_input,*d_output;
	    hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
      
	    hipMemcpy(aMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);
	  
        rppDestroyGPU(handle);
        
        hipFree(d_input);
        hipFree(d_output);
	    hipHostFree(reductionFuncResultArr);
	    hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMin_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pMin, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
	    int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMin_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pMin)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
	    int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMin_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u aMin[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n
	    int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(aMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

/******************** tensor_max ********************/

RppStatus rppt_tensor_max_gpu(RppPtr_t srcPtr,
                              RpptDescPtr srcDescPtr,
                              RppPtr_t imageMaxArr,
                              Rpp32u imageMaxArrLength,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (imageMaxArrLength < srcDescPtr->n)   // max of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageMaxArrLength < srcDescPtr->n * 4)   // max of each channel, and overall max of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_tensor_max(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8u*>(imageMaxArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        hip_exec_tensor_max((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                            srcDescPtr,
                            static_cast<half*>(imageMaxArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_tensor_max((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                            srcDescPtr,
                            static_cast<Rpp32f*>(imageMaxArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        hip_exec_tensor_max(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                            srcDescPtr,
                            static_cast<Rpp8s*>(imageMaxArr),
                            roiTensorPtrSrc,
                            roiType,
                            rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}

NppStatus nppiMax_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u aMax[3])
{
	    int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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
		
        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
	    Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
	    Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
	    Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n
			
	    int bitDepthByteSize = 0;
	    if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
		    bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
	    else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
		    bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
	    hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
	    Rpp8u *d_input,*d_output;
	    hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();
      
	    hipMemcpy(aMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);
	  
        rppDestroyGPU(handle);
        
        hipFree(d_input);
        hipFree(d_output);
	    hipHostFree(reductionFuncResultArr);
	    hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMax_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pMax, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
	    if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMax_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pMax)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
	    if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMax_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u aMax[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
	    if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(aMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_8u64s_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64s *pSum, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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

        //reductionFuncResultArrLength = srcDescPtr->n;
        RppStatus status;
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        pSum[0] = (long long)reductionFuncResultArr[0];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_8u64s_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64s *pSum)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * sizeof(Rpp8u) + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * sizeof(Rpp8u) + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp8u *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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

        //reductionFuncResultArrLength = srcDescPtr->n;
        RppStatus status;
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        pSum[0] = (long long)reductionFuncResultArr[0];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pSum, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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

        //reductionFuncResultArrLength = srcDescPtr->n;
        RppStatus status;
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        pSum[0] = (double)reductionFuncResultArr[0];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f *pSum)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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

        //reductionFuncResultArrLength = srcDescPtr->n;
        RppStatus status;
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        pSum[0] = (double)reductionFuncResultArr[0];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f aSum[3], NppStreamContext nppStreamCtx)
{
        //pkd
        //bool reductionTypeCase = 1;
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
		        bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        aSum[0] = (double)reductionFuncResultArr[0];
        aSum[1] = (double)reductionFuncResultArr[1];
        aSum[2] = (double)reductionFuncResultArr[2];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiSum_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp64f aSum[3])
{
        //pkd
        //bool reductionTypeCase = 1;
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        Rpp64u *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp64u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
		        bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);

        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        aSum[0] = (double)reductionFuncResultArr[0];
        aSum[1] = (double)reductionFuncResultArr[1];
        aSum[2] = (double)reductionFuncResultArr[2];

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}


NppStatus nppiMin_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f *pMin, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_f32 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_f32 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMin_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f *pMin)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_f32 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_f32 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMin_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f aMin[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_f32 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_f32 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(aMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMin_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f aMin[3])
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_f32 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_f32 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n
        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(aMin,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}


NppStatus nppiMax_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f *pMax, NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMax_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f *pMax)
{
        int noOfImages = 1;
        int ip_channel = 1;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NCHW;//NCHW
        dstDescPtr->layout = RpptLayout::NCHW;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(pMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMax_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f aMax[3], NppStreamContext nppStreamCtx)
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(aMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

NppStatus nppiMax_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp32f aMax[3])
{
        int noOfImages = 1;
        int ip_channel = 3;
        RpptDesc srcDesc, dstDesc;
        RpptDescPtr srcDescPtr, dstDescPtr;
        srcDescPtr = &srcDesc;
        dstDescPtr = &dstDesc;

        srcDescPtr->layout = RpptLayout::NHWC;//NCHW
        dstDescPtr->layout = RpptLayout::NHWC;//NCHW
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

        Rpp64u ioBufferSize = 0;
        Rpp64u oBufferSize = 0;

        ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

        Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u oBufferSizeInBytes_u8 = oBufferSize * 4 + dstDescPtr->offsetInBytes;
        Rpp64u inputBufferSize = ioBufferSize * 4 + srcDescPtr->offsetInBytes;
        Rpp64u outputBufferSize = oBufferSize * 4 + dstDescPtr->offsetInBytes;

        void *reductionFuncResultArr;
        Rpp32u reductionFuncResultArrLength = srcDescPtr->n;//c1 is srcDescPtr->n

        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
                bitDepthByteSize = sizeof(Rpp8u);//sum is Rpp64u
        else if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32))
                bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f
        hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize);
		
        Rpp32f *d_input,*d_output;
        hipMalloc(&d_input,inputBufferSize);
        hipMalloc(&d_output,outputBufferSize);
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
        status = rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
        hipDeviceSynchronize();

        hipMemcpy(aMax,reductionFuncResultArr,ip_channel*bitDepthByteSize,hipMemcpyHostToDevice);

        rppDestroyGPU(handle);

        hipFree(d_input);
        hipFree(d_output);
        hipHostFree(reductionFuncResultArr);
        hipHostFree(roiTensorPtrSrc);

        return(hipRppStatusTocudaNppStatus(status));
}

#endif // backend
