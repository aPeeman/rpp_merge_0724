#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"
#define round(value) ( (value - (int)(value)) >=0.5 ? (value + 1) : (value))
#define MAX_SIZE 64

__device__ unsigned int get_pkd_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, 
                        unsigned int height, unsigned channel)
                         {
 return (id_z + id_x * channel + id_y * width * channel);
}
extern "C" __global__
void partial_histogram_pln( unsigned char *input,
                         unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int local_size = (int)hipBlockDim_x * (int)hipBlockDim_y;
    int group_indx = (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x) * 256 ;
    unsigned int pixId;
    __shared__ uint tmp_histogram [256];
    int tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    int j = 256 ;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    __syncthreads();

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x  + id_y * width ;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + width * height];
        unsigned char pixelB = input[pixId + 2 * width * height];
        atomicAdd(&tmp_histogram[pixelR], 1);
        atomicAdd(&tmp_histogram[pixelG], 1);
        atomicAdd(&tmp_histogram[pixelB], 1);
    }
    __syncthreads();
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}

extern "C" __global__
void partial_histogram_pkd( unsigned char *input,
                            unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int local_size = (int)hipBlockDim_x * (int)hipBlockDim_y;
    int group_indx = (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x) * 256 ;
    unsigned int pixId;
    __shared__ uint tmp_histogram [256];
    int tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    int j = 256 ;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    __syncthreads();

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x * channel + id_y * width * channel;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + 1];
        unsigned char pixelB = input[pixId + 2];
        atomicAdd(&tmp_histogram[pixelR], 1);
        atomicAdd(&tmp_histogram[pixelG], 1);
        atomicAdd(&tmp_histogram[pixelB], 1);
    }
    __syncthreads();
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}

extern "C" __global__
void partial_histogram_batch( unsigned char* input,
                                    unsigned int *histogramPartial,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long *batch_index,
                                    const unsigned int num_groups,// For partial histogram indexing// try out a better way
                                    const unsigned int channel,
                                    const unsigned int batch_size,
                                    unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    ){

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int hist_index = num_groups * id_z * 256;
    int local_size = (int)hipBlockDim_x * (int)hipBlockDim_y;
    int group_indx = (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x) * 256 ;
    unsigned int pixId;
    __shared__ uint tmp_histogram [256];
    int tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    int j = 256 ;
    int indx = 0;
    //printf("%d",id_z);
    int temp_index = id_z * 256;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    __syncthreads();
    if ((id_z < batch_size) && (id_x < width[id_z]) && (id_y < height[id_z]))
    {
        pixId =  batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex  ;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + inc[id_z]];
        unsigned char pixelB = input[pixId + 2 * inc[id_z]];
        atomicAdd(&tmp_histogram[pixelR], 1);
        atomicAdd(&tmp_histogram[pixelG], 1);
        atomicAdd(&tmp_histogram[pixelB], 1);
    }
    __syncthreads();
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[hist_index + group_indx + tid] = tmp_histogram[temp_index + tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[hist_index + group_indx + indx + tid] = tmp_histogram[ temp_index + indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
     // printf("tmp hist%d", histogramPartial[hist_index + group_indx + tid]);

}

extern "C" __global__
void partial_histogram_semibatch( unsigned char* input,
                                     unsigned int *histogramPartial,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int max_width,
                                     const unsigned long batch_index,
                                     const unsigned  int hist_index,// For partial histogram indexing// try out a better way
                                    const unsigned int channel,
                                     const unsigned int inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    ){

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int local_size = (int)hipBlockDim_x * (int)hipBlockDim_y;
    int group_indx = (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x) * 256 ;
    unsigned int pixId;
    __shared__ uint tmp_histogram [256];
    int tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    int j = 256 ;
    int indx = 0;
    //printf("%d",id_z);
    do
    {
        if (tid < j)
        tmp_histogram[ indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    __syncthreads();
    if ( (id_x < width) && (id_y < height))
    {
        pixId =  batch_index + (id_x  + id_y * max_width ) * plnpkdindex  ;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + inc];
        unsigned char pixelB = input[pixId + 2 * inc];
       atomicAdd(&tmp_histogram[pixelR], 1);
        atomicAdd(&tmp_histogram[pixelG], 1);
        atomicAdd(&tmp_histogram[pixelB], 1);
    }
    __syncthreads();
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[hist_index + group_indx + tid] = tmp_histogram[ tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[hist_index + group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
      //printf("tmp hist%d", histogramPartial[hist_index + group_indx + tid]);
      //printf("batch index %lu", batch_index);
}



extern "C" __global__ void
histogram_sum_partial( unsigned int *histogramPartial,
                       unsigned int *histogram,
                      const unsigned int num_groups)
{
     int tid = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
    int  group_indx;
    int  n = num_groups;

    __shared__ uint tmp_histogram[256];
    tmp_histogram[tid] = histogramPartial[tid];  
  
    group_indx = 256;
    while (--n > 1)
    {
        tmp_histogram[tid] = tmp_histogram[tid] +  histogramPartial[group_indx + tid];
        group_indx += 256; 
    }
    histogram[tid] = tmp_histogram[tid];

}

extern "C" __global__ void
histogram_sum_partial_batch( unsigned int *histogramPartial,
                       unsigned int *histogram,
                      const unsigned int batch_size,
                      const unsigned int num_groups,
                      const unsigned int channel)
{
    int  tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int  bid = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int  group_indx;
    int  n = num_groups;
   __shared__ uint tmp_histogram[256];
    unsigned int hist_index = num_groups * bid * 256;
    group_indx = 256;
    if(bid < batch_size){
        tmp_histogram[tid] = histogramPartial[hist_index + tid];    
         while (--n > 1)
            {
                tmp_histogram[tid] = tmp_histogram[tid] +  histogramPartial[hist_index + group_indx + tid];
                group_indx += 256; 
            }
            histogram[256 * bid + tid] = tmp_histogram[tid];
    }

}

extern "C" __global__ void
histogram_equalize_pln( unsigned char *input,
                    unsigned char *output,
                    unsigned int *cum_histogram,
                   const unsigned int width,
                   const unsigned int height,
                   const unsigned int channel
                   )
{
    float normalize_factor = 255.0 / (height * width * channel);
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int local_size = (int)hipBlockDim_x * (int)hipBlockDim_y;
    unsigned pixId;
    pixId = id_x  + id_y * width + id_z * height * width;
    output[pixId] = cum_histogram[input[pixId]] * (normalize_factor);
}

extern "C" __global__ void
histogram_equalize_pkd( unsigned char *input,
                    unsigned char *output,
                    unsigned int *cum_histogram,
                   const unsigned int width,
                   const unsigned int height,
                   const unsigned int channel
                   )
{
    float normalize_factor = 255.0 / (height * width * channel);
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int local_size = (int)hipBlockDim_x * (int)hipBlockDim_y;
    unsigned pixId;
    pixId = id_x * channel + id_y * width * channel + id_z;
    output[pixId] = round(cum_histogram[input[pixId]] * (normalize_factor));
}


extern "C" __global__ void
histogram_equalize_batch( unsigned char* input,
                                     unsigned char* output,
                                     unsigned int *cum_histogram,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long *batch_index,
                                    const unsigned int channel,
                                    const unsigned int batch_size,
                                     unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                   )
{
    unsigned int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    unsigned int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    float normalize_factor = 255.0 / (height[id_z] * width[id_z] * channel);
    int indextmp=0;
    unsigned long pixIdx = 0;
    
    if(id_x < width[id_z] && id_y < height[id_z])
    {   
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        for(indextmp = 0; indextmp < channel; indextmp++){
                //output[pixIdx] = cum_histogram[ 256 * id_z + input[pixIdx]] * (normalize_factor);
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
    }
}

extern "C" __global__ void histogram_even_batch(unsigned char *input,
						unsigned int *xroi_begin,
						unsigned int *xroi_end,
						unsigned int *yroi_begin,
						unsigned int *yroi_end,
						unsigned int *height,
						unsigned int *width,
						unsigned int *max_width,
						unsigned long long *batch_index,
						int *pHist,
						int nLevels,
						int nLowerLevel,
						int nUpperLevel,
						const unsigned int channel,
						unsigned int *inc, // use width * height for pln and 1 for pkd
						const int plnpkdindex) // use 1 pln 3 for pkd
{ 
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
	
    unsigned long pixIdx = 0;
	
    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z] ) * plnpkdindex;
 
    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) { 
        for(int indextmp = 0; indextmp < channel; indextmp++)
	{
		unsigned char pixelValue = input[pixIdx]; 
 
		int histIndex = ((pixelValue - nLowerLevel) * nLevels) / (nUpperLevel - nLowerLevel + 1); 
 
		if (histIndex >= 0 && histIndex < nLevels) {
			atomicAdd(&pHist[histIndex], 1); 
		}
		pixIdx += inc[id_z];
	}
    }
    /*else if((id_x < width[id_z] ) && (id_y < height[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[pixIdx] = input[pixIdx];
            pixIdx += inc[id_z];
        }
    }*/
} 

RppStatus npp_exec_histogram_even_batch(Rpp8u *srcPtr, rpp::Handle& handle, Rpp32u channel, Rpp32s plnpkdind, 
					Rpp32u max_height, Rpp32u max_width, int *pHist, int nLevels, int nLowerLevel, int nUpperLevel)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(histogram_even_batch,
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
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
		       pHist,
		       nLevels,
		       nLowerLevel,
		       nUpperLevel,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}
