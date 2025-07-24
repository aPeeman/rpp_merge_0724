#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
#define max(a,b)  (((a)>(b))?(a):(b))
#define min(a,b)  (((a)<(b))?(a):(b))

extern "C" __global__ void box_filter_batch(unsigned char *input,
                                            unsigned char *output,
                                            unsigned int *kernelSize,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    unsigned char valuer, valuer1, valueg, valueg1, valueb, valueb1;
    int kernelSizeTemp = kernelSize[id_z];

    int bound = (kernelSizeTemp - 1) / 2;
    /*if(id_x < width[id_z] && id_y < height[id_z])
    {
        long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
        if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            int r = 0, g = 0, b = 0;
            for(int i = -bound; i <= bound; i++)
            {
                for(int j = -bound; j <= bound; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] - 1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        r += input[index];
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            g += input[index];
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            b += input[index];
                        }
                    }
                    else
                    {
                        r = 0;
                        if(channel == 3)
                        {
                            g = 0;
                            b = 0;
                        }
                        break;
                    }
                }
            }

            if(id_x >= bound && id_x <= width[id_z] - bound - 1 && id_y >= bound && id_y <= height[id_z] - bound - 1 )
            {
                int temp = (int)(r / (kernelSizeTemp * kernelSizeTemp));
                output[pixIdx] = saturate_8u(temp);
                if(channel == 3)
                {
                    temp = (int)(g / (kernelSizeTemp * kernelSizeTemp));
                    output[pixIdx + inc[id_z]] = saturate_8u(temp);
                    temp = (int)(b / (kernelSizeTemp * kernelSizeTemp));
                    output[pixIdx + inc[id_z] * 2] = saturate_8u(temp);
                }
            }
            else
            {
                for(int indextmp = 0; indextmp < channel; indextmp++)
                {
                    output[pixIdx] = input[pixIdx];
                    pixIdx += inc[id_z];
                }
            }
        }
        else if((id_x < width[id_z]) && (id_y < height[id_z]))
        {
            for(int indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }*/
    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        long pixIdx = batch_index[id_z] + (id_x + id_y * width[id_z]) * plnpkdindex;
        int startX = id_x - bound;
        int startY = id_y - bound;
        int endX = id_x + bound;
        int endY = id_y + bound;

        int sum = 0;
        int count = 0;

        for (int i = -bound; i <= bound; ++i) {
            for (int j = -bound; j <= bound; ++j) {
                int sampleX = startX + j;
                int sampleY = startY + i;

                if (sampleX >= 0 && sampleX < xroi_end[id_z] &&
                    sampleY >= 0 && sampleY < yroi_end[id_z]) {

                    sum += input[sampleY * max_width[id_z] + sampleX];
                    count++;
                }
            }
        }

        if (count > 0) {
            int average = sum / count;
            output[pixIdx] = saturate_8u(average);
        } else {
            output[pixIdx] = input[pixIdx];
        }
    }
}

/*extern "C" __global__ void npp_box_filter_batch(unsigned char *input,
                                            unsigned char *output,
                                            unsigned int *kernelSize,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex, // use 1 pln 3 for pkd
					    RppiPoint maskAnchor,
					    RppiBorderType borderMode) 
{  
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    unsigned char valuer, valuer1, valueg, valueg1, valueb, valueb1;
    int kernelSizeTemp = kernelSize[id_z];

    int bound = (kernelSizeTemp - 1) / 2;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
        if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            int r = 0, g = 0, b = 0;
            for(int i = -bound; i <= bound; i++)
            {
                for(int j = -bound; j <= bound; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] - 1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        r += input[index];
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            g += input[index];
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            b += input[index];
                        }
                    }
                    else
                    {
                        r = 0;
                        if(channel == 3)
                        {
                            g = 0;
                            b = 0;
                        }
                        break;
                    }
                }
            }

            if(id_x >= bound && id_x <= width[id_z] - bound - 1 && id_y >= bound && id_y <= height[id_z] - bound - 1 )
            {
                int temp = (int)(r / (kernelSizeTemp * kernelSizeTemp));
                output[pixIdx] = saturate_8u(temp);
                if(channel == 3)
                {
                    temp = (int)(g / (kernelSizeTemp * kernelSizeTemp));
                    output[pixIdx + inc[id_z]] = saturate_8u(temp);
                    temp = (int)(b / (kernelSizeTemp * kernelSizeTemp));
                    output[pixIdx + inc[id_z] * 2] = saturate_8u(temp);
                }
            }
            else
            {
                for(int indextmp = 0; indextmp < channel; indextmp++)
                {
                    output[pixIdx] = input[pixIdx];
                    pixIdx += inc[id_z];
                }
            }
        }
        else if((id_x < width[id_z]) && (id_y < height[id_z]))
        {
            for(int indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    } 
} */

extern "C" __global__ void npp_box_filter_batch(unsigned char *input,
                                            unsigned char *output,
                                            unsigned int *kernelSize,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            const unsigned int channel,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex, // use 1 pln 3 for pkd
                                            RppiPoint maskAnchor,
                                            RppiBorderType borderMode)
{  
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;  
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;  

    unsigned char valuer, valueg, valueb;  
    int kernelSizeTemp = kernelSize[id_z];  
    int bound = (kernelSizeTemp - 1) / 2;  

    if (id_x < width[id_z] && id_y < height[id_z])  
    {  
        long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;  
        if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))  
        {  
            int r = 0, g = 0, b = 0;  

            for (int i = -bound; i <= bound; i++)  
            {  
                for (int j = -bound; j <= bound; j++)  
                {  
                    int srcX = id_x + j;  
                    int srcY = id_y + i;  

                    // 边界处理  
                    if (borderMode == 0)  
                    {  
                        if (srcX < 0 || srcX >= width[id_z] || srcY < 0 || srcY >= height[id_z])  
                            continue; // 在边界外，不处理  
                    }  
                    else if (borderMode == 1)  
                    {  
                        if (srcX < 0 || srcX >= width[id_z] || srcY < 0 || srcY >= height[id_z])  
                        {  
                            r += 0; // 填充值为0  
                            g += 0;  
                            b += 0;  
                            continue;  
                        }  
                    }  
                    else if (borderMode == 2)  
                    {  
                        srcX = max(0, min(srcX, width[id_z] - 1));  
                        srcY = max(0, min(srcY, height[id_z] - 1));  
                    }  
                    else if (borderMode == 3)  
                    {  
                        srcX = (srcX + width[id_z]) % width[id_z];  
                        srcY = (srcY + height[id_z]) % height[id_z];  
                    }  
                    else if (borderMode == 4)  
                    {  
                        if (srcX < 0)  
                            srcX = -srcX;  
                        else if (srcX >= width[id_z])  
                            srcX = 2 * width[id_z] - srcX - 1;  

                        if (srcY < 0)  
                            srcY = -srcY;  
                        else if (srcY >= height[id_z])  
                            srcY = 2 * height[id_z] - srcY - 1;  
                    }  

                    // 计算实际访问的像素索引  
                    unsigned int index = batch_index[id_z] + (srcX + (srcY * max_width[id_z])) * plnpkdindex;  
                    r += input[index];  
                    if (channel == 3)  
                    {  
                        g += input[index + inc[id_z]];  
                        b += input[index + inc[id_z] * 2];  
                    }  
                }  
            }  

            if (id_x >= bound && id_x < width[id_z] - bound && id_y >= bound && id_y < height[id_z] - bound)  
            {  
                int temp = (int)(r / (kernelSizeTemp * kernelSizeTemp));  
                output[pixIdx] = saturate_8u(temp);  
                if (channel == 3)  
                {  
                    temp = (int)(g / (kernelSizeTemp * kernelSizeTemp));  
                    output[pixIdx + inc[id_z]] = saturate_8u(temp);  
                    temp = (int)(b / (kernelSizeTemp * kernelSizeTemp));  
                    output[pixIdx + inc[id_z] * 2] = saturate_8u(temp);  
                }  
            }  
            else  
            {  
                for (int indextmp = 0; indextmp < channel; indextmp++)  
                {  
                    output[pixIdx + indextmp * inc[id_z]] = input[pixIdx + indextmp * inc[id_z]];  
                }  
            }  
        }  
        else if ((id_x < width[id_z]) && (id_y < height[id_z]))  
        {  
            for (int indextmp = 0; indextmp < channel; indextmp++)  
            {  
                output[pixIdx + indextmp * inc[id_z]] = input[pixIdx + indextmp * inc[id_z]];  
            }  
        }  
    }  
}   

RppStatus hip_exec_box_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(box_filter_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus npp_exec_box_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiPoint maskAnchor, RppiBorderType rBorderType)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(npp_box_filter_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind,
		       maskAnchor,
		       rBorderType);

    return RPP_SUCCESS;
}

extern "C" __global__ void labelmarkers_batch(unsigned char *input,
                                         unsigned int *output,
                                         unsigned int *xroi_begin,
                                         unsigned int *xroi_end,
                                         unsigned int *yroi_begin,
                                         unsigned int *yroi_end,
                                         unsigned int *height,
                                         unsigned int *width,
                                         unsigned int *max_width,
                                         RppiNorm eNorm,
                                         unsigned long long *batch_index,
                                         const int plnpkdindex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x < width[id_z] && id_y < height[id_z]) {
        long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
                if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
                {
                        if (input[pixIdx] > 0)
                        {
                                if (eNorm == 0) {
                                        bool hasNeighbor = false;

                                        for (int dy = -1; dy <= 1; dy++) {
                                                for (int dx = -1; dx <= 1; dx++) {
                                                        if (dy == 0 && dx == 0) continue;

                                                        int neighborX = id_x + dx;
                                                        int neighborY = id_y + dy;

                                                        if (neighborX >= 0 && neighborX < xroi_end[id_z] &&
                                                                neighborY >= 0 && neighborY < yroi_end[id_z]) {
                                                                unsigned char pNeighborPixel = input[batch_index[id_z] + (neighborX + neighborY * max_width[id_z]) * plnpkdindex];
                                                                if (pNeighborPixel > 0) {
                                                                        hasNeighbor = true;
                                                                        break;
                                                                }
                                                        }
                                                }
                                                if (hasNeighbor) break;
                                        }
                                        output[pixIdx] = hasNeighbor ? 1 : 0;
                                } else if (eNorm == 1) {
                                        bool hasNeighbor = false;

                                        int neighbors[][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} };
                                        for (int i = 0; i < 4; i++) {
                                                int neighborX = id_x + neighbors[i][0];
                                                int neighborY = id_y + neighbors[i][1];

                                                if (neighborX >= 0 && neighborX < xroi_end[id_z] &&
                                                        neighborY >= 0 && neighborY < yroi_end[id_z]) {
                                                        unsigned char pNeighborPixel = input[batch_index[id_z] + (neighborX + neighborY * max_width[id_z]) * plnpkdindex];
                                                        if (pNeighborPixel > 0) {
                                                                hasNeighbor = true;
                                                                break;
                                                        }
                                                }
                                        }
                                        output[pixIdx] = hasNeighbor ? 1 : 0;
                                } else {
                                        output[pixIdx] = 0;
                                }
                        } else {
                                output[pixIdx] = 0;
                        }
                } else if((id_x < width[id_z]) && (id_y < height[id_z])) {
                        output[pixIdx] = input[pixIdx];
                }
        }
}

RppStatus npp_exec_labelmarkers_batch_batch(Rpp8u *srcPtr, Rpp32u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiNorm eNorm)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(labelmarkers_batch,
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
		       eNorm,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       plnpkdind);

    return RPP_SUCCESS;
}

/*extern "C" __global__ void labelmarkers_batch(unsigned char *input,
					 unsigned int *output,
					 unsigned int *xroi_begin,
					 unsigned int *xroi_end,
                                         unsigned int *yroi_begin,
                                         unsigned int *yroi_end,
                                         unsigned int *height,
                                         unsigned int *width,
                                         unsigned int *max_width,  
                                         RppiNorm eNorm,
					 unsigned long long *batch_index,
                                         const int plnpkdindex) 
{  
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;  

    if (id_x < width[id_z] && id_y < height[id_z]) {   
        long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex; 
		if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        	{  
			if (input[pixIdx] > 0) 
			{  
				if (eNorm == 0) {  
					bool hasNeighbor = false;  
  
					for (int dy = -1; dy <= 1; dy++) {  
						for (int dx = -1; dx <= 1; dx++) {   
							if (dy == 0 && dx == 0) continue;  

							int neighborX = id_x + dx;  
							int neighborY = id_y + dy;  
 
							if (neighborX >= 0 && neighborX < xroi_end[id_z] &&  
								neighborY >= 0 && neighborY < yroi_end[id_z]) {  
								unsigned char pNeighborPixel = input[batch_index[id_z] + (neighborX + neighborY * max_width[id_z]) * plnpkdindex];  
								if (pNeighborPixel > 0) {  
									hasNeighbor = true;
									break;  
								}  
							}  
						}  
						if (hasNeighbor) break;  
					}  
					output[pixIdx] = hasNeighbor ? 1 : 0; 
				} else if (eNorm == 1) {  
					bool hasNeighbor = false;  
 
					int neighbors[][2] = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} }; 
					for (int i = 0; i < 4; i++) {  
						int neighborX = id_x + neighbors[i][0];  
						int neighborY = id_y + neighbors[i][1];  

						if (neighborX >= 0 && neighborX < xroi_end[id_z] &&  
							neighborY >= 0 && neighborY < yroi_end[id_z]) {  
							unsigned char pNeighborPixel = input[batch_index[id_z] + (neighborX + neighborY * max_width[id_z]) * plnpkdindex];  
							if (pNeighborPixel > 0) {  
								hasNeighbor = true; 
								break;  
							}  
						}  
					}   
					output[pixIdx] = hasNeighbor ? 1 : 0;  
				} else {  
					output[pixIdx] = 0;
				}  
			} else {
				output[pixIdx] = 0;
			}
		} else if((id_x < width[id_z]) && (id_y < height[id_z])) {
			output[pixIdx] = input[pixIdx];
		}
	}
}*/ 

extern "C" __global__ void compressmarkers_batch(unsigned int *input,
                                         unsigned int *xroi_begin,
                                         unsigned int *xroi_end,
                                         unsigned int *yroi_begin,
                                         unsigned int *yroi_end,
                                         unsigned int *height,
                                         unsigned int *width,
                                         unsigned int *max_width,
                                         RppBufferDescriptor* pBufferBatch,
                                         unsigned int* pNewMaxLabelID,
                                         int nPerImageBufferSize,
                                         unsigned long long *batch_index,
                                         const int plnpkdindex)
{  
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x < width[id_z] && id_y < height[id_z]) {
    long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
    	if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    	{
                 unsigned int label = input[pixIdx];
                 if (label > 0) {
                       unsigned int newLabel = atomicAdd(&pNewMaxLabelID[batch_index[id_z]], 1);
                       input[pixIdx] = newLabel + 1;
		       unsigned int* pBuffer = (unsigned int*)pBufferBatch[batch_index[id_z]].pData;  
                       pBuffer[newLabel] = label; 
                }
        }
    }
}

RppStatus npp_exec_compressmarkers_batch_batch(Rpp32u *srcPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, RppBufferDescriptor *pBufferBatch, Rpp32u *pNewMaxLabelID, Rpp32s nPerImageBufferSize)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(compressmarkers_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
		       pBufferBatch,
		       pNewMaxLabelID,
		       nPerImageBufferSize,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       plnpkdind);

    return RPP_SUCCESS;
}

extern "C" __global__ void compressmarkers(unsigned int *input,
					 unsigned int *xroi_begin,
					 unsigned int *xroi_end,
                                         unsigned int *yroi_begin,
                                         unsigned int *yroi_end,
                                         unsigned int *height,
                                         unsigned int *width,
                                         unsigned int *max_width,  
                                         int* pNewNumber,
					 int nStartingNumber,
					 unsigned long long *batch_index,
                                         const int plnpkdindex)
{  
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
		
	if (id_x < width[id_z] && id_y < height[id_z]) {   
        long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex; 
	if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
		unsigned int label = input[pixIdx];
		if (label > 0) {
			input[pixIdx] = label + nStartingNumber;
			atomicMax(pNewNumber, label + nStartingNumber);
		}
	}
    }
}

RppStatus npp_exec_compressmarkers(Rpp32u *srcPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, Rpp32s *pNewNumber, Rpp32s nStartingNumber)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(compressmarkers,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       pNewNumber,
                       nStartingNumber,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       plnpkdind);

    return RPP_SUCCESS;
}
