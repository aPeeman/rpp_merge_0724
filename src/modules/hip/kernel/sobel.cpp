#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
#define saturate_16s(value) ((value) > 32767 ? 32767 : ((value) < -32768 ? -32768 : (value)))
#define abs(value) ((value) < 0 ? (-value) : value)

__device__ unsigned int power_sobel(unsigned int a, unsigned int b)
{
    unsigned int sum = 1;
    for(int i = 0; i < b; i++)
    {
        sum += sum * a;
    }
    return sum;
}

__device__ int calcSobelx(int a[3][3])
{
    int gx[3][3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sum = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            sum += a[i][j] * gx[i][j];
        }
    }
    return sum;
}

__device__ int calcSobely(int a[3][3])
{
    int gy[3][3]={-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int sum = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            sum += a[i][j] * gy[i][j];
        }
    }
    return sum;
}

__device__ int calcPrewittx(int a[3][3])
{
    int gx[3][3] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
    int sum = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            sum += a[i][j] * gx[i][j];
        }
    }
    return sum;
}

__device__ int calcPrewitty(int a[3][3])
{
    int gy[3][3]={-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int sum = 0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            sum += a[i][j] * gy[i][j];
        }
    }
    return sum;
}

extern "C" __global__ void sobel_pkd(unsigned char *input,
                                     unsigned char *output,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int channel,
                                     const unsigned int sobelType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int value = 0;
    int value1 = 0;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int a[3][3];

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height - 1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                a[i+1][j+1] = input[index];
            }
            else
            {
                a[i+1][j+1] = 0;
            }
        }
    }
    if(sobelType == 2)
    {
        value = calcSobelx(a);
        value1 = calcSobely(a);
        value = power_sobel(value, 2);
        value1 = power_sobel(value1, 2);
        value = sqrt((float)(value + value1));
        output[pixIdx] = saturate_8u(value);
    }

    if(sobelType == 1)
    {
        value = calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}

extern "C" __global__ void sobel_pln(unsigned char *input,
                                     unsigned char *output,
                                     const unsigned int height,
                                     const unsigned int width,
                                     const unsigned int channel,
                                     const unsigned int sobelType)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int value = 0;
    int value1 = 0;
    int a[3][3];

    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height - 1)
            {
                unsigned int index = pixIdx + j + (i * width);
                a[i+1][j+1] = input[index];
            }
            else
            {
                a[i+1][j+1] = 0;
            }
        }
    }
    if(sobelType == 2)
    {
        value = calcSobelx(a);
        value1 = calcSobely(a);
        value = power_sobel(value, 2);
        value1 = power_sobel(value1, 2);
        value = sqrt((float)(value + value1));
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 1)
    {
        value = calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}


extern "C" __global__ void sobel_batch(unsigned char *input,
                                       unsigned char *output,
                                       unsigned int *sobelType,
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
    int sobelTypeTemp = sobelType[id_z];
    int indextmp = 0;
    long pixIdx = 0;
    // printf("%d", id_x);
    int value = 0;
    int value1 = 0;
    int r[3][3],g[3][3],b[3][3];

    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(int i = -1; i <= 1; i++)
        {
            for(int j = -1; j <= 1; j++)
            {
                if(id_x != 0 && id_x != width[id_z] - 1 && id_y != 0 && id_y != height[id_z] - 1)
                {
                    unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                    r[i+1][j+1] = input[index];
                    if(channel == 3)
                    {
                        index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                        g[i+1][j+1] = input[index];
                        index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                        b[i+1][j+1] = input[index];
                    }
                }
                else
                {
                    r[i+1][j+1] = 0;
                    if(channel == 3)
                    {
                        g[i+1][j+1] = 0;
                        b[i+1][j+1] = 0;
                    }
                }
            }
        }

        if(sobelType[id_z] == 2)
        {
            value = calcSobelx(r);
            value1 = calcSobely(r);
            value = power_sobel(value, 2);
            value1 = power_sobel(value1, 2);
            value = sqrt((float)(value + value1));
            output[pixIdx] = saturate_8u(value);
            if(channel == 3)
            {
                value = calcSobelx(g);
                value1 = calcSobely(g);
                value = power_sobel(value, 2);
                value1 = power_sobel(value1, 2);
                value = sqrt((float)(value + value1));
                output[pixIdx + inc[id_z]] = saturate_8u(value);
                value = calcSobelx(b);
                value1 = calcSobely(b);
                value = power_sobel(value, 2);
                value1 = power_sobel(value1, 2);
                value = sqrt((float)(value + value1));
                output[pixIdx + inc[id_z] * 2] = saturate_8u(value);
            }
        }
        if(sobelType[id_z] == 1)
        {
            value = calcSobely(r);
            output[pixIdx] = saturate_8u(value);
            if(channel == 3)
            {
                value = calcSobely(g);
                output[pixIdx + inc[id_z]] = saturate_8u(value);
                value = calcSobely(g);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(value);
            }
        }
        if(sobelType[id_z] == 0)
        {
            value = calcSobelx(r);
            output[pixIdx] = saturate_8u(value);
            if(channel == 3)
            {
                value = calcSobelx(g);
                output[pixIdx + inc[id_z]] = saturate_8u(value);
                value = calcSobelx(g);
                output[pixIdx + inc[id_z] * 2] = saturate_8u(value);
            }
        }
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

extern "C" __global__ void prewitt_batch(unsigned char *input,  
                                       short *outputx,
				       short *outputy,
				       short *pDstMag,
				       float *pDstAngle,                        
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
                                       const int plnpkdindex, // use 1 for pln, 3 for pkd
				       RppiBorderType borderType, 
				       RppiNorm norm)
{  
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;  
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;  

    //unsigned char valuer, valuer1, valueg, valueg1, valueb, valueb1;
    int value = 0;
    int value1 = 0;
    int valuer = 0;
    int valuer1 = 0;
    
    long pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;  

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&   
        (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))  
    {  
        int r[3][3] = {0}, g[3][3] = {0}, b[3][3] = {0};  

        for (int i = -1; i <= 1; i++)  
        {  
            for (int j = -1; j <= 1; j++)  
            {   
                int current_x = id_x + j;  
                int current_y = id_y + i;  
 
                if (current_x < 0 || current_x >= width[id_z] ||   
                    current_y < 0 || current_y >= height[id_z])  
                {  
                    switch (borderType)  
                    {  
                        case 0:  
                            r[i + 1][j + 1] = 0; // or skip, it's undefined  
                            if (channel == 3) {  
                                g[i + 1][j + 1] = 0;  
                                b[i + 1][j + 1] = 0;  
                            }  
                            break;  

                        case 1:  
                            r[i + 1][j + 1] = 0; // assuming a constant border value of 0  
                            if (channel == 3) {  
                                g[i + 1][j + 1] = 0;  
                                b[i + 1][j + 1] = 0;  
                            }  
                            break;  
                        
                        case 2:  
                            //r[i + 1][j + 1] = (current_x < 0 || current_x >= width[id_z]) ? r[1][1] : input[pixIdx + (j + (i * max_width[id_z])) * plnpkdindex];  
                            //if (channel == 3) {  
                            //    g[i + 1][j + 1] = (current_x < 0 || current_x >= width[id_z]) ? g[1][1] : input[pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z]];  
                            //    b[i + 1][j + 1] = (current_x < 0 || current_x >= width[id_z]) ? b[1][1] : input[pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2];  
                            //}
                            current_x = max(0, min(current_x, width[id_z] - 1));  
                	    current_y = max(0, min(current_y, height[id_z] - 1));
			    r[i + 1][j + 1] = input[batch_index[id_z] + (current_x + (current_y * max_width[id_z])) * plnpkdindex];
			    if (channel == 3) {
				g[i + 1][j + 1] = input[batch_index[id_z] + (current_x + (current_y * max_width[id_z])) * plnpkdindex + inc[id_z]];
				b[i + 1][j + 1] = input[batch_index[id_z] + (current_x + (current_y * max_width[id_z])) * plnpkdindex + inc[id_z] * 2];
			    }
                            break;  

                        case 3:  
                            current_x = (current_x + width[id_z]) % width[id_z];  
                            current_y = (current_y + height[id_z]) % height[id_z];  
                            r[i + 1][j + 1] = input[batch_index[id_z] + (current_x + current_y * max_width[id_z]) * plnpkdindex];  
                            if (channel == 3) {  
                                g[i + 1][j + 1] = input[batch_index[id_z] + (current_x + current_y * max_width[id_z]) * plnpkdindex + inc[id_z]];  
                                b[i + 1][j + 1] = input[batch_index[id_z] + (current_x + current_y * max_width[id_z]) * plnpkdindex + inc[id_z] * 2];  
                            }  
                            break;  

                        case 4:  
                            current_x = (current_x < 0) ? -current_x : (current_x >= width[id_z]) ? 2 * width[id_z] - current_x - 1 : current_x;  
                            current_y = (current_y < 0) ? -current_y : (current_y >= height[id_z]) ? 2 * height[id_z] - current_y - 1 : current_y;  
                            r[i + 1][j + 1] = input[batch_index[id_z] + (current_x + current_y * max_width[id_z]) * plnpkdindex];  
                            if (channel == 3) {  
                                g[i + 1][j + 1] = input[batch_index[id_z] + (current_x + current_y * max_width[id_z]) * plnpkdindex + inc[id_z]];  
                                b[i + 1][j + 1] = input[batch_index[id_z] + (current_x + current_y * max_width[id_z]) * plnpkdindex + inc[id_z] * 2];  
                            }  
                            break;  
                    }  
                }  
                else  
                {  
                    unsigned int index = batch_index[id_z] + (current_x + (current_y * max_width[id_z])) * plnpkdindex;  
                    r[i + 1][j + 1] = input[index];  
                    if (channel == 3)  
                    {  
                        index = batch_index[id_z] + (current_x + (current_y * max_width[id_z])) * plnpkdindex + inc[id_z];  
                        g[i + 1][j + 1] = input[index];  
                        index = batch_index[id_z] + (current_x + (current_y * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;  
                        b[i + 1][j + 1] = input[index];  
                    }  
                }  
            }  
        }
	if(norm == 1) {
                value = calcPrewittx(r);
                value1 = calcPrewitty(r);
                outputx[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = saturate_16s(value);
                outputy[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = saturate_16s(value1);
                if (pDstMag != 0){
                	valuer = (abs(value) + abs(value1));
                	//outputx[pixIdx] = saturate_16s(value);
                	//outputy[pixIdx] = saturate_16s(value1);
                	pDstMag[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = saturate_16s(valuer);
		}
                /*if(channel == 3)
                {
                        value = calcPrewittx(g);
                        value1 = calcPrewitty(g);
                        outputx[pixIdx + inc[id_z]] = saturate_8u(value);
                        outputy[pixIdx + inc[id_z]] = saturate_8u(value1);
                        valueg = power_sobel(value, 2);
                        valueg1 = power_sobel(value1, 2);
                        valueg = sqrt((float)(valueg + valueg1));
                        //outputx[pixIdx + inc[id_z]] = saturate_8u(value);
                        //outputy[pixIdx + inc[id_z]] = saturate_8u(value1);
                        pDstMag[pixIdx + inc[id_z]] = saturate_16s(valueg);
                        value = calcPrewittx(b);
                        value1 = calcPrewitty(b);
                        outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value);
                        outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value1);
                        valueb = power_sobel(value, 2);
                        valueb1 = power_sobel(value1, 2);
                        valueb = sqrt((float)(valueb + valueb1));
                        //outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value);
                        //outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value1);
                        pDstMag[pixIdx + inc[id_z] * 2] = saturate_16s(valueb);
                }*/	
		if(pDstAngle != 0 && (value1 != 0 && value != 0)){
			pDstAngle[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = atan2f(static_cast<float>(value1),static_cast<float>(value));
		} 
	} else if(norm == 2) {
		value = calcPrewittx(r);
		value1 = calcPrewitty(r);
		outputx[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = saturate_16s(value);
                outputy[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = saturate_16s(value1);
		if (pDstMag != 0){
			valuer = power_sobel(value, 2);
			valuer1 = power_sobel(value1, 2);
			valuer = sqrt((float)(valuer + valuer1));
		//outputx[pixIdx] = saturate_16s(value);
		//outputy[pixIdx] = saturate_16s(value1);
			pDstMag[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = saturate_16s(valuer);
		}
		/*if(channel == 3)
		{
			value = calcPrewittx(g);
			value1 = calcPrewitty(g);
			outputx[pixIdx + inc[id_z]] = saturate_8u(value);
                        outputy[pixIdx + inc[id_z]] = saturate_8u(value1);
			valueg = power_sobel(value, 2);
			valueg1 = power_sobel(value1, 2);
			valueg = sqrt((float)(valueg + valueg1));
			//outputx[pixIdx + inc[id_z]] = saturate_8u(value);
			//outputy[pixIdx + inc[id_z]] = saturate_8u(value1);
			pDstMag[pixIdx + inc[id_z]] = saturate_16s(valueg);
			value = calcPrewittx(b);
			value1 = calcPrewitty(b);
			outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value);
                        outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value1);
			valueb = power_sobel(value, 2);
			valueb1 = power_sobel(value1, 2);
			valueb = sqrt((float)(valueb + valueb1));
			//outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value);
			//outputx[pixIdx + inc[id_z] * 2] = saturate_8u(value1);
			pDstMag[pixIdx + inc[id_z] * 2] = saturate_16s(valueb);
		}*/
		if(pDstAngle != 0 && (value1 != 0 && value != 0)){
                        pDstAngle[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = atan2f(static_cast<float>(value1),static_cast<float>(value));
                } 
	}
    }
    else if((id_x < width[id_z]) && (id_y < height[id_z]))
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            outputx[batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex] = input[pixIdx];
            pixIdx += inc[id_z];
        }
    }
}

RppStatus hip_exec_sobel_pln(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32u sobelType)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(sobel_pln,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       height,
                       width,
                       channel,
                       sobelType);

    return RPP_SUCCESS;
}

RppStatus hip_exec_sobel_pkd(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32u sobelType)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = width;
    int globalThreads_y = height;
    int globalThreads_z = 1;

    hipLaunchKernelGGL(sobel_pkd,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       height,
                       width,
                       channel,
                       sobelType);

    return RPP_SUCCESS;
}

RppStatus hip_exec_sobel_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(sobel_batch,
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

RppStatus npp_exec_prewitt_filter_batch(Rpp8u *srcPtr, Rpp16s *dstPtrx, Rpp16s *dstPtry, Rpp16s *pDstMag, Rpp32f *pDstAngle, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, RppiNorm eNorm, RppiBorderType rBorderType)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(prewitt_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtrx,
		       dstPtry,
		       pDstMag,
		       pDstAngle,
                       //handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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
		       rBorderType,
		       eNorm);

    return RPP_SUCCESS;
}
