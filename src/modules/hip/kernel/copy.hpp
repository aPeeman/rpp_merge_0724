#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void copy_pkd3_pln3_hip_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void copy_pln3_pkd3_hip_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 pix_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_copy_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               rpp::Handle& handle)
{
    if (srcDescPtr->layout == dstDescPtr->layout)
    {
        hipMemcpy(dstPtr, srcPtr, dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(T), hipMemcpyDeviceToDevice);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(copy_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(srcDescPtr->w, srcDescPtr->h));
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(copy_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(srcDescPtr->w, srcDescPtr->h));
        }
    }

    return RPP_SUCCESS;
}

extern "C" __global__ void copyborder_batch(const unsigned char *input,
					    unsigned char *output,
					    unsigned int *xroi_begin,
					    unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end, 
					    unsigned int *height,
					    unsigned int *width,
					    unsigned int *max_width,
					    unsigned int *max_height,
					    int oDstwidth,
					    int oDstheight,
					    int nTopBorderHeight,
					    int nLeftBorderWidth,  
					    unsigned char nValue,
					    unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex) // use 1 pln 3 for pkd
{  
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
	
    int indextmp = 0;
    long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y <= oDstheight) && (id_x <= oDstwidth)) {
	// top
	if (id_y < nTopBorderHeight && id_x < oDstwidth) {  
		output[pixIdx] = nValue;  
	}  
	// left  
	if (id_y >= nTopBorderHeight && id_y < oDstheight - nTopBorderHeight && id_x < nLeftBorderWidth) {  
		output[pixIdx] = nValue;  
	}  
	// right  
	if (id_y >= nTopBorderHeight && id_y < oDstheight - nTopBorderHeight && id_x >= oDstwidth - nLeftBorderWidth) {  
		output[pixIdx] = nValue;  
	}  
	// top 
	if (id_y >= oDstheight - nTopBorderHeight && id_x < oDstwidth) {  
		output[pixIdx] = nValue;  
	}  

	if (id_y >= nTopBorderHeight && id_y < oDstheight - nTopBorderHeight &&   
             id_x >= nLeftBorderWidth && id_x < oDstwidth - nLeftBorderWidth) {
		if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
			int srcX = id_x - nLeftBorderWidth;  
			int srcY = id_y - nTopBorderHeight;  
			output[pixIdx] = input[batch_index[id_z] + (srcX  + srcY * max_width[id_z]) * plnpkdindex];
			pixIdx += inc[id_z];
		}
		else if((id_x < width[id_z]) && (id_y < height[id_z]))
		{
			for(indextmp = 0; indextmp < channel; indextmp++)
			{
				output[pixIdx] = input[pixIdx];
				pixIdx += inc[id_z];
			}
		}
	}  
    }   
}

RppStatus npp_exec_copyborder_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, int oDstwidth, int oDstheight, int nTopBorderHeight, int nLeftBorderWidth, Rpp8u nValue)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(copyborder_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,    
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
		       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
		       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
		       handle.GetInitHandle()->mem.mgpu.maxSrcSize.height,
		       oDstwidth,
		       oDstheight,
                       nTopBorderHeight,
                       nLeftBorderWidth,
                       nValue,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}  
