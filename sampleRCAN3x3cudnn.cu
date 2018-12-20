#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <curand.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <chrono>

#include <memory> // mt add shared_ptr

#include <cublas_v2.h>

#include <cudnn.h>

//#include "NvInfer.h"
//#include "NvOnnxParser.h"
#include "common.h"
using namespace nvinfer1;

static const bool VERBOSE = false;

#define DEBUG false

static const int SCALE = 2;

static const int INPUT_BATCH = 4;
static const int INPUT_CH = 3;
static const int INPUT_H = 320;
static const int INPUT_W = 240;
static const int OUTPUT_SIZE = 3 * 320 * 240;
//static Logger gLogger;
//static int gUseDLACore{-1};
static const int N_FEAT = 64;

static const int iterations = 10;
static const int avgRuns = 10;
static const int pct = 99;

#define BLOCK_NUM 512
#define THREAD_NUM 512

// TODO(mt): trying to use cudnnGetConvolutionForwardAlgorithm or cudnnGetConvolutionForwardAlgorithm_v7 to get best performance?
static const cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
//CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
//CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
//CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
//CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
//CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
//CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
//CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
//CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
//CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8

static const cudnnBatchNormMode_t bn_mode = CUDNN_BATCHNORM_SPATIAL;
///* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) */
//CUDNN_BATCHNORM_PER_ACTIVATION = 0,
//
///* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors) */
//CUDNN_BATCHNORM_SPATIAL = 1,
//
///*
// * bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors).
// * May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values
// */
//CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,

cudnnDataType_t cudnn_data_type = CUDNN_DATA_FLOAT;
// TODO(mt): this variable is too big now, fix it afterwards. originally 8 * 1024 * 1024 in caffe, but I wonder if it's useful.
size_t workspace_limit_bytes = 512 * 1024 * 1024;

static const std::string DELIMIT('=', 89);

const std::vector<std::string> directories{"/home/mt/cur_work/RCAN-master/RCAN_TestCode/code_testonnx/onnx_model/", "/home/mt/cur_work/RCAN-master/RCAN_TestCode/code/onnx_model/", "/home/mt/cur_work/dataset/DIV2K/DIV2K_train_LR_bicubic/X2/"};
std::string locateFile(const std::string& input)
{
    return locateFile(input, directories);
}

float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    if(status != CUDNN_STATUS_SUCCESS) \
	  std::cout << __FILE__ << "," << __LINE__ <<" " \
		  << cudnnGetErrorString(status) << std::endl; \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if(status != CUBLAS_STATUS_SUCCESS) \
	  std::cout << __FILE__ << "," << __LINE__ <<" " \
		  << "CUBLAS ERROR" << std::endl; \
  } while (0)

#define CHECK_EQ(a, b)                            \
    do                                            \
    {                                             \
        if ((a) != (b))							  \
        {                                         \
            std::cout << __FILE__ << ", "		  \
			<< __LINE__ << " "					  \
			<< "CHECK FAILURE " << #a			  \
			<< "(" << (a) << ")"				  \
			<< " != " << #b <<					  \
			"(" << (b) << ")" << std::endl;		  \
            abort();                              \
        }                                         \
    } while (0)

#define CHECK_GE(a, b)                            \
    do                                            \
    {                                             \
        if ((a) < (b))							  \
        {                                         \
            std::cout << __FILE__ << ", "		  \
			<< __LINE__ << " "					  \
			<< "CHECK FAILURE " << #a			  \
			<< "(" << (a) << ")"				  \
			<< " < " << #b <<					  \
			"(" << (b) << ")" << std::endl;		  \
            abort();                              \
        }                                         \
    } while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}

// simple image reader
void readOpenCVFile(const std::string& fileName, uint8_t buffer[INPUT_CH * INPUT_H * INPUT_W])
{
    readOpenCVFile(fileName, buffer, INPUT_CH, INPUT_H, INPUT_W);
}

#define CUDA_KERNEL_LOOP(i, n) \
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
				i < (n); \
				i += blockDim.x * gridDim.x)

// c is c not c * s * s, while numel and fmel considers c * r * r as channel.
__global__ void PixelShuffleKernel(float* src, float* dst, int n, int c, int h, int w, int s, int numel, int fmel, int spel)
{
	CUDA_KERNEL_LOOP(i, numel)
	{
		int id_n = i / fmel;
		int id_cr2 = i / spel - id_n * c * s * s;
		int id_g = id_cr2 % (s * s);
		int id_gh = id_g / s;
		int id_gw = id_g % s;
		int id_h = i / w - i / spel * h;
		int id_w = i % w;
		
		int dst_id_n = id_n;
		int dst_id_c = id_cr2 / (s * s);
		int offset_dst = id_n * fmel + id_cr2 / (s * s) * s * s * spel + (s * id_h + id_gh) * w * s + id_w * s + id_gw;
		dst[offset_dst] = src[i];
	}
}

typedef struct _cudnnConvInfo 
{
	cudnnHandle_t* handle;
	cublasHandle_t* cublas_handle;
	cudaStream_t* stream;

	cudnnConvolutionDescriptor_t conv_desc;
	cudnnTensorDescriptor_t input_desc, output_desc;

	cudnnTensorDescriptor_t bias_desc;
	cudnnFilterDescriptor_t filter_desc;

	size_t workspace_fwd_sizes;
	void* workspace_data;

	void* filter_data;
	void* bias_data;
} _cudnnConvInfo;

// Since all the convolution layers in RCAN have the same input and output feature map size, here we don't pass int the output size, conv stride and group parameters.
void InitConv(_cudnnConvInfo* conv_info, 
			int ni, int ci, int co, int hi, int wi, 
			int k_h, int k_w, int p_h, int p_w, 
			cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
{
	if(DEBUG)
	{
		std::cout << "Debug, ni = " << ni << ", ci = " << ci << ", co = " << co << ", hi = " << hi << ", wi = " << wi << std::endl
			<< ", k_h = " << k_h << ", k_w = " << k_w << ", p_h = " << p_h << ", p_w = " << p_w << std::endl;
	}
	conv_info->handle = handle;
	conv_info->cublas_handle = cublas_handle;
	conv_info->stream = stream;

	CUDNN_CHECK(cudnnCreateFilterDescriptor(&(conv_info->filter_desc) ) );
	// filter desc must be set before conv desc
	CUDNN_CHECK(cudnnSetFilter4dDescriptor(conv_info->filter_desc, cudnn_data_type,
					CUDNN_TENSOR_NCHW, co, ci, k_h, k_w));

	CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&(conv_info->conv_desc) ) );
	
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(conv_info->input_desc) ) );
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(conv_info->output_desc) ) );
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(conv_info->bias_desc) ) );

	// Set
	CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_info->conv_desc,
					p_h, p_w, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, cudnn_data_type));
					//p_h, p_w, 1, 1, 1, 1, CUDNN_CONVOLUTION, cudnn_data_type));
					
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv_info->input_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
					ni, ci, hi, wi));
	int no_postval, co_postval, ho_postval, wo_postval;
	CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
					conv_info->conv_desc,
					conv_info->input_desc,
					conv_info->filter_desc,
					&no_postval, &co_postval, &ho_postval, &wo_postval) );
	if(DEBUG)
		std::cout << "Debug, no_postval = " << no_postval << ", co_postval = " << co_postval << ", ho_postval = " << ho_postval << ", wo_postval = " << wo_postval << std::endl;
	CHECK_EQ(ni, no_postval);
	CHECK_EQ(co, co_postval);
	CHECK_EQ(hi, ho_postval);
	CHECK_EQ(wi, wo_postval);
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv_info->output_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
					no_postval, co, ho_postval, wo_postval));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(conv_info->bias_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
					1, co, 1, 1));
	//CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(conv_info->input_desc, cudnn_data_type,
	//				n, c, h, w, c*h*w, h*w, w, 1));
	//CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(conv_info->output_desc, cudnn_data_type,
	//				n, c, h, w, c*h*w, h*w, w, 1));
	//CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(conv_info->bias_desc, cudnn_data_type,
	//				1, c, 1, 1, c, 1, 1, 1));


	// Malloc workspace
	CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(*handle,
					conv_info->input_desc,
					conv_info->filter_desc,
					conv_info->conv_desc,
					conv_info->output_desc,
					conv_algo,
					&(conv_info->workspace_fwd_sizes) ) );
	//conv_info->workspace_fwd_sizes = 32 * 1024 * 1024;
	if(DEBUG)
		std::cout << "workspace_fwd_sizes is " << conv_info->workspace_fwd_sizes << " in bytes" << std::endl;

    CHECK(cudaMalloc(&(conv_info->workspace_data), static_cast<size_t>(static_cast<float>(conv_info->workspace_fwd_sizes) * 1.05) ) );

	// Malloc data
	CHECK(cudaMalloc(&(conv_info->filter_data), co * ci * k_h * k_w * sizeof(float) ) );
	CHECK(cudaMalloc(&(conv_info->bias_data), co * sizeof(float) ) );

	// TODO(mt): may delete this? Besides, the memset function shall be used to float, but now the variable passed in is void*.
	CHECK(cudaMemset(conv_info->filter_data, 1., sizeof(float) * co * ci * k_h * k_w) );
	CHECK(cudaMemset(conv_info->bias_data, 1., sizeof(float) * co) );
}

void DestroyConv(_cudnnConvInfo* conv_info)
{
	cudnnDestroyTensorDescriptor(conv_info->input_desc);
	cudnnDestroyTensorDescriptor(conv_info->output_desc);
	cudnnDestroyTensorDescriptor(conv_info->bias_desc);

	cudnnDestroyConvolutionDescriptor(conv_info->conv_desc);

	cudnnDestroyFilterDescriptor(conv_info->filter_desc);

	cudaFree(conv_info->workspace_data);

	cudaFree(conv_info->filter_data);
	cudaFree(conv_info->bias_data);
}

typedef struct _cudnnGAPInfo 
{
	cudnnHandle_t* handle;
	cublasHandle_t* cublas_handle;
	cudaStream_t* stream;

	cudnnPoolingDescriptor_t pool_desc;
	cudnnTensorDescriptor_t input_desc, output_desc;
} _cudnnGAPInfo;

// API consistent with InitConv 
void InitGAP(_cudnnGAPInfo* gap_info, 
			int ni, int ci, int co, int hi, int wi, 
			int k_h, int k_w, int p_h, int p_w, 
			cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
{
	CHECK_EQ(ci, co);
	if(DEBUG)
	{
		std::cout << "Debug, ni = " << ni << ", ci = " << ci << ", co = " << co << ", hi = " << hi << ", wi = " << wi << std::endl
			<< ", k_h = " << k_h << ", k_w = " << k_w << ", p_h = " << p_h << ", p_w = " << p_w << std::endl;
	}
	gap_info->handle = handle;
	gap_info->cublas_handle = cublas_handle;
	gap_info->stream = stream;

	CUDNN_CHECK(cudnnCreatePoolingDescriptor(&(gap_info->pool_desc) ) );
	
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(gap_info->input_desc) ) );
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(gap_info->output_desc) ) );

	// Set
	CUDNN_CHECK(cudnnSetPooling2dDescriptor(gap_info->pool_desc,
					CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_PROPAGATE_NAN,
					hi, wi, 0, 0, 1, 1));
					
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(gap_info->input_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
					ni, ci, hi, wi));
	int no_postval, co_postval, ho_postval, wo_postval;
	CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(
					gap_info->pool_desc,
					gap_info->input_desc,
					&no_postval, &co_postval, &ho_postval, &wo_postval) );
	if(DEBUG)
		std::cout << "Debug, no_postval = " << no_postval << ", co_postval = " << co_postval << ", ho_postval = " << ho_postval << ", wo_postval = " << wo_postval << std::endl;
	CHECK_EQ(ni, no_postval);
	CHECK_EQ(co, co_postval);
	CHECK_EQ(ho_postval, 1);
	CHECK_EQ(wo_postval, 1);
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(gap_info->output_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
					no_postval, co, ho_postval, wo_postval));
	//CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(conv_info->input_desc, cudnn_data_type,
	//				n, c, h, w, c*h*w, h*w, w, 1));
	//CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(conv_info->output_desc, cudnn_data_type,
	//				n, c, h, w, c*h*w, h*w, w, 1));
	//CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(conv_info->bias_desc, cudnn_data_type,
	//				1, c, 1, 1, c, 1, 1, 1));

	// Seemingly no extra workspace is demanded by Pooling2d
}

void DestroyGAP(_cudnnGAPInfo* gap_info)
{
	cudnnDestroyPoolingDescriptor(gap_info->pool_desc);

	cudnnDestroyTensorDescriptor(gap_info->input_desc);
	cudnnDestroyTensorDescriptor(gap_info->output_desc);
}

typedef struct _cudnnActivationInfo 
{
	cudnnHandle_t* handle;
	cublasHandle_t* cublas_handle;
	cudaStream_t* stream;

	cudnnActivationDescriptor_t act_desc;
	cudnnTensorDescriptor_t input_desc, output_desc;
} _cudnnActivationInfo;

// CUDNN_ACTIVATION_SIGMOID
void InitActivation(_cudnnActivationInfo* act_info, 
			int ni, int ci, int hi, int wi,
			cudnnActivationMode_t act_mode, 
			cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
{
	act_info->handle = handle;
	act_info->cublas_handle = cublas_handle;
	act_info->stream = stream;

	CUDNN_CHECK(cudnnCreateActivationDescriptor(&(act_info->act_desc) ) );
	
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(act_info->input_desc) ) );
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(act_info->output_desc) ) );

	// Set
	CUDNN_CHECK(cudnnSetActivationDescriptor(act_info->act_desc,
					act_mode, CUDNN_PROPAGATE_NAN,
					1.0) );
					
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(act_info->input_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
					ni, ci, hi, wi));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(act_info->output_desc, CUDNN_TENSOR_NCHW, cudnn_data_type,
					ni, ci, hi, wi));
}

void DestroyActivation(_cudnnActivationInfo* act_info)
{
	cudnnDestroyActivationDescriptor(act_info->act_desc);

	cudnnDestroyTensorDescriptor(act_info->input_desc);
	cudnnDestroyTensorDescriptor(act_info->output_desc);
}

typedef struct _cudnnBNInfo 
{
	cudnnHandle_t* handle;
	cublasHandle_t* cublas_handle;
	cudaStream_t* stream;

	cudnnTensorDescriptor_t input_desc, output_desc;
	cudnnTensorDescriptor_t bn_param_desc;
} _cudnnBNInfo;

void InitBN(_cudnnBNInfo* bn_info, 
			int ni, int ci, int hi, int wi, 
			cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
{
	bn_info->handle = handle;
	bn_info->cublas_handle = cublas_handle;
	bn_info->stream = stream;

	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(bn_info->input_desc) ) );
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(bn_info->output_desc) ) );
	CUDNN_CHECK(cudnnCreateTensorDescriptor(&(bn_info->bn_param_desc) ) );

	// Set
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(bn_info->input_desc, 
					CUDNN_TENSOR_NCHW, cudnn_data_type, 1, ci, hi, wi));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(bn_info->output_desc, 
					CUDNN_TENSOR_NCHW, cudnn_data_type, 1, ci, hi, wi));

	CUDNN_CHECK(cudnnSetTensor4dDescriptor(bn_info->bn_param_desc, 
					CUDNN_TENSOR_NCHW, cudnn_data_type, 1, ci, 1, 1));
}

void DestroyBN(_cudnnBNInfo* bn_info)
{
	cudnnDestroyTensorDescriptor(bn_info->input_desc);
	cudnnDestroyTensorDescriptor(bn_info->output_desc);
}

class Module
{
public:
	Module() {}
	~Module() {}

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream) = 0;

	// TODO(mt): only one input and one output supported now.
	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory) = 0;
};

typedef std::shared_ptr<Module> pModule;

class Conv2d : public Module
{
public:
	Conv2d(int ni, int ci, int co, int hi, int wi, int k_h, int k_w, int p_h, int p_w, float alpha=1.0, float beta=0.0, bool bias=true)
		: conv_info(), NI(ni), CI(ci), CO(co), HI(hi), WI(wi),
		K_H(k_h), K_W(k_w), P_H(p_h), P_W(p_w), 
		a(alpha), b(beta), one(1.), bias_(bias) {}

	~Conv2d() { DestroyConv(&(this->conv_info) ); }

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
	{
		InitConv(&(this->conv_info),
			NI, CI, CO, HI, WI,
			K_H, K_W, P_H, P_W,
			handle, cublas_handle, stream);
	}

	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		if(DEBUG)
		{
			std::cout << "x = " << x << std::endl
				<< "conv_info.filter = " << conv_info.filter_data << std::endl
				<< "conv_info.workspace_data = " << conv_info.workspace_data << std::endl
				<< "&a = " << &a << ", a = " << a << std::endl
				<< "&b = " << &b << ", b = " << a << std::endl
				<< "y = " << y << std::endl;
		}
		CUDNN_CHECK(cudnnConvolutionForward(*(conv_info.handle),
						&a,
						conv_info.input_desc, x,
						conv_info.filter_desc, conv_info.filter_data,
						conv_info.conv_desc, conv_algo, 
						conv_info.workspace_data, conv_info.workspace_fwd_sizes,
						&b,
						conv_info.output_desc, y) );
		if(this->bias_)
		    CUDNN_CHECK(cudnnAddTensor(*(conv_info.handle), 
		    				&one,
		    				conv_info.bias_desc, conv_info.bias_data,
		    				&one,
		    				conv_info.output_desc, y) );
	}

public:
	_cudnnConvInfo conv_info;
	int NI, CI, HI, WI; // input size
	int CO; // output channel
	int K_H, K_W; // kernel
	int P_H, P_W; // padding
	// TODO(mt): fix it using typedef
	float a, b;
	float one;
private:
	bool bias_;
};

class GAP : public Module
{
public:
	GAP(int ni, int ci, int co, int hi, int wi, int k_h, int k_w, int p_h, int p_w, float alpha=1.0, float beta=0.0)
		: gap_info(), NI(ni), CI(ci), CO(co), HI(hi), WI(wi),
		K_H(k_h), K_W(k_w), P_H(p_h), P_W(p_w), 
		a(alpha), b(beta), one(1.) {}

	~GAP() { DestroyGAP(&(this->gap_info) ); }

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
	{
		InitGAP(&(this->gap_info),
			NI, CI, CO, HI, WI,
			K_H, K_W, P_H, P_W,
			handle, cublas_handle, stream);
	}

	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		if(DEBUG)
		{
			std::cout << "x = " << x << std::endl
				<< "&a = " << &a << ", a = " << a << std::endl
				<< "&b = " << &b << ", b = " << a << std::endl
				<< "y = " << y << std::endl;
		}
		CUDNN_CHECK(cudnnPoolingForward(*(gap_info.handle),
						gap_info.pool_desc,
						&a,
						gap_info.input_desc, x,
						&b,
						gap_info.output_desc, y) );
	}

public:
	_cudnnGAPInfo gap_info;
	int NI, CI, HI, WI; // input size
	int CO; // output channel
	int K_H, K_W; // kernel
	int P_H, P_W; // padding
	// TODO(mt): fix it using typedef
	float a, b;
	float one;
};

class SIGMOID : public Module
{
public:
	SIGMOID(int ni, int ci, int hi, int wi, float alpha=1.0, float beta=0.0)
		: act_info(), NI(ni), CI(ci), HI(hi), WI(wi),
		a(alpha), b(beta), one(1.) {}

	~SIGMOID() { DestroyActivation(&(this->act_info) ); }

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
	{
		InitActivation(&(this->act_info), 
			NI, CI, HI, WI,
			CUDNN_ACTIVATION_SIGMOID,
			handle, cublas_handle, stream);
	}

	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		CUDNN_CHECK(cudnnActivationForward(*(act_info.handle),
						act_info.act_desc,
						&a,
						act_info.input_desc, x,
						&b,
						act_info.output_desc, y) );
	}

public:
	_cudnnActivationInfo act_info;
	int NI, CI, HI, WI; // input size
	int K_H, K_W; // kernel
	int P_H, P_W; // padding
	// TODO(mt): fix it using typedef
	float a, b;
	float one;
};

class RELU : public Module
{
public:
	RELU(int ni, int ci, int hi, int wi, float alpha=1.0, float beta=0.0)
		: act_info(), NI(ni), CI(ci), HI(hi), WI(wi),
		a(alpha), b(beta), one(1.) {}

	~RELU() { DestroyActivation(&(this->act_info) ); }

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
	{
		InitActivation(&(this->act_info), 
			NI, CI, HI, WI,
			CUDNN_ACTIVATION_RELU,
			handle, cublas_handle, stream);
	}

	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		CUDNN_CHECK(cudnnActivationForward(*(act_info.handle),
						act_info.act_desc,
						&a,
						act_info.input_desc, x,
						&b,
						act_info.output_desc, y) );
	}

public:
	_cudnnActivationInfo act_info;
	int NI, CI, HI, WI; // input size
	int K_H, K_W; // kernel
	int P_H, P_W; // padding
	// TODO(mt): fix it using typedef
	float a, b;
	float one;
};

// This BN is intended to be used as elementwise multiplication.
class BN : public Module
{
public:
	BN(int ni, int ci, int hi, int wi, float alpha=1.0, float beta=1.0, double eps=2e-5)
		: bn_info(), NI(ni), CI(ci), HI(hi), WI(wi), bn_scale(), alpha(alpha), beta(beta), eps(eps)
	{
		CHECK(cudaMalloc(&est_mean, 1 * INPUT_CH * 1 * 1 * sizeof(float)));
		CHECK(cudaMalloc(&est_var, 1 * INPUT_CH * 1 * 1 * sizeof(float)));
		CHECK(cudaMemset(est_mean, 0, 1 * INPUT_CH * 1 * 1 * sizeof(float)));
		CHECK(cudaMemset(est_var, 1, 1 * INPUT_CH * 1 * 1 * sizeof(float)));

		CHECK(cudaMalloc(&bn_bias, 1 * INPUT_CH * 1 * 1 * sizeof(float)));
		CHECK(cudaMemset(bn_bias, 0, 1 * INPUT_CH * 1 * 1 * sizeof(float)));

		step = ci * hi * wi;
	}

	~BN()
	{
		cudaFree(this->est_mean);
		cudaFree(this->est_var);
		cudaFree(this->bn_bias);
		DestroyBN(&(this->bn_info) );
	}

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
	{
		//std::cout << "bn" << std::endl;
		InitBN(&(this->bn_info), 
			NI, CI, HI, WI,
			handle, cublas_handle, stream);
	}

	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		for(long i = 0; i < NI; i++)
		{
			float* x_new = static_cast<float*>(x) + i * step;
			float* y_new = static_cast<float*>(y) + i * step;
			float* channel_value = static_cast<float*>((*reusable_memory)[2]) + i * CI;
#define P(x) std::cout << #x << " = " << x << std::endl;
			//P(x_new);
			//P(y_new);
			//P(alpha);
			//P(beta);
			//P(channel_value);
			//P(bn_bias);
			//P(est_mean);
			//P(est_var);
			//P(i);
			CUDNN_CHECK(cudnnBatchNormalizationForwardInference(*(bn_info.handle),
							bn_mode,
							&alpha,
							&beta,
							bn_info.input_desc,
							static_cast<void*>(x_new),
							bn_info.output_desc,
							static_cast<void*>(y_new),
							bn_info.bn_param_desc,
							static_cast<void*>(channel_value),
							bn_bias,
							est_mean,
							est_var,
							eps) );
		}
	}

public:
	_cudnnBNInfo bn_info;
	int NI, CI, HI, WI; // input size
	long step;
	void* est_mean;
	void* est_var;
	void* bn_scale;
	void* bn_bias;
	float alpha, beta;
	float eps;
};

class RCAB : public Module
{
public:
	RCAB(int ni, int ci, int co, int hi, int wi, int k_h, int k_w, int p_h, int p_w, float alpha=1.0, float beta=0.0, bool bias=true, int reduction=16)
		: reduction(reduction)
	{
		main_conv.push_back(std::shared_ptr<Module>(new Conv2d(ni, ci, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, bias) ) );
		main_conv.push_back(std::shared_ptr<Module>(new Conv2d(ni, co, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, bias) ) );

		fc_conv.push_back(std::shared_ptr<Module>(new Conv2d(ni, co, co / reduction, 1, 1, 1, 1, 0, 0, 1.0, 0.0, true) ) );
		fc_conv.push_back(std::shared_ptr<Module>(new Conv2d(ni, co / reduction, co, 1, 1, 1, 1, 0, 0, 1.0, 0.0, true) ) );

		main_acts.push_back(std::shared_ptr<Module>(new RELU(ni, co, hi, wi, 1.0, 0.0) ) );

		gap.push_back(std::shared_ptr<Module>(new GAP(ni, co, co, hi, wi, hi, wi, 0, 0, alpha=1.0, beta=0.0) ) );

		ca_acts.push_back(std::shared_ptr<Module>(new RELU(ni, co / reduction, 1, 1, 1.0, 0.0) ) );
		ca_acts.push_back(std::shared_ptr<Module>(new SIGMOID(ni, co, 1, 1, 1.0, 0.0) ) );

		// here we use BN to perform multiplication.
		ca_acts.push_back(std::shared_ptr<Module>(new BN(ni, co, hi, wi, 1.0, 1.0, 2e-5) ) );
	}

	~RCAB() {}

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
	{
		main_conv[0]->init(handle, cublas_handle, stream);
		main_conv[1]->init(handle, cublas_handle, stream);

		fc_conv[0]->init(handle, cublas_handle, stream);
		fc_conv[1]->init(handle, cublas_handle, stream);

		main_acts[0]->init(handle, cublas_handle, stream);

		gap[0]->init(handle, cublas_handle, stream);

		ca_acts[0]->init(handle, cublas_handle, stream);
		ca_acts[1]->init(handle, cublas_handle, stream);
		ca_acts[2]->init(handle, cublas_handle, stream);
	}

	// NOTE that for RCAB, y is dummy to give a compatiable api but use memory best.
	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		int main_conv_len = reusable_memory->size();
		CHECK_GE(main_conv_len, 4);
		std::vector<void*> dummy_reusable_memory;

		main_conv[0]->forward(x, (*reusable_memory)[0], &dummy_reusable_memory);
		main_acts[0]->forward((*reusable_memory)[0], (*reusable_memory)[0], &dummy_reusable_memory);
		main_conv[1]->forward((*reusable_memory)[0], (*reusable_memory)[1], &dummy_reusable_memory);

		gap[0]->forward((*reusable_memory)[1], (*reusable_memory)[2], &dummy_reusable_memory);

		fc_conv[0]->forward((*reusable_memory)[2], (*reusable_memory)[3], &dummy_reusable_memory);
		ca_acts[0]->forward((*reusable_memory)[3], (*reusable_memory)[3], &dummy_reusable_memory);
		fc_conv[1]->forward((*reusable_memory)[3], (*reusable_memory)[2], &dummy_reusable_memory);
		ca_acts[1]->forward((*reusable_memory)[2], (*reusable_memory)[2], &dummy_reusable_memory);

		// this will inplace append to original x
		ca_acts[2]->forward((*reusable_memory)[1], x, reusable_memory);
	}

public:
	std::vector<pModule> main_conv;
	std::vector<pModule> fc_conv;
	std::vector<pModule> main_acts;
	std::vector<pModule> ca_acts;
	std::vector<pModule> gap;
	int reduction;
};

class RG : public Module
{
public:
	RG(int num_resblocks, int ni, int ci, int co, int hi, int wi, int k_h, int k_w, int p_h, int p_w, float alpha=1.0, float beta=0.0, bool bias=true, int reduction=16)
		: num_resblocks(num_resblocks)
	{
		for(int i = 0; i < this->num_resblocks; i++)
		{
			if(i == 0)
				rcab.push_back(std::shared_ptr<Module>(new RCAB(ni, ci, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, true, reduction) ) );
			else	
				rcab.push_back(std::shared_ptr<Module>(new RCAB(ni, co, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, true, reduction) ) );
		}
		conv.push_back(std::shared_ptr<Module>(new Conv2d(ni, co, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, bias) ) );
	}

	~RG() {}

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream)
	{
		for(int i = 0; i < num_resblocks; i++)
			rcab[i]->init(handle, cublas_handle, stream);
		conv[0]->init(handle, cublas_handle, stream);
	}

	// NOTE that for RCAB, y could be the same as x to maximize memory usage.
	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		for(int i = 0; i < num_resblocks; i++)
			rcab[i]->forward(x, x, reusable_memory);
		conv[0]->forward(x, y, reusable_memory);
	}

public:
	std::vector<pModule> rcab;
	std::vector<pModule> conv;
	int num_resblocks;
};

class PixelShuffle : public Module
{
public:
	// ci is cout not cout * scale * scale
	PixelShuffle(int scale, int ni, int ci, int hi, int wi)
		: NI(ni), CI(ci), SI(scale), HI(hi), WI(wi) {}

	~PixelShuffle() {}

	virtual void init(cudnnHandle_t* handle, cublasHandle_t* cublas_handle, cudaStream_t* stream) {}

	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		PixelShuffleKernel<<<BLOCK_NUM, THREAD_NUM>>>(
					static_cast<float*>(x), 
					static_cast<float*>(y), 
					NI, CI, HI, WI, SI, 
					NI * CI * SI * SI * HI * WI, 
					CI * SI * SI * HI * WI, 
					HI * WI);
	}
public:
	int NI, CI, SI, HI, WI;
};

void DoInference(std::vector<pModule>& network, void* input_device, std::vector<void*>& infer_memory, std::vector<void*>& reusable_memory, bool verbose=false)
{
	if(verbose)
	{
	    std::cout << DELIMIT << std::endl 
	    	<< "Module id " << 0 << " forward" << std::endl;
	}
	network[0]->forward(input_device, infer_memory[0], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
		std::cout << DELIMIT << std::endl 
			<< "Module id " << 1 << " forward" << std::endl;
	}
	network[1]->forward(infer_memory[0], infer_memory[1], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
		std::cout << DELIMIT << std::endl 
			<< "Module id " << 2 << " forward" << std::endl;
	}
	network[2]->forward(infer_memory[1], infer_memory[2], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
		std::cout << DELIMIT << std::endl 
			<< "Module id " << 3 << " forward" << std::endl;
	}
	network[3]->forward(infer_memory[2], infer_memory[1], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
		std::cout << DELIMIT << std::endl 
			<< "Module id " << 4 << " forward" << std::endl;
	}
	network[4]->forward(infer_memory[1], infer_memory[2], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
		std::cout << DELIMIT << std::endl 
			<< "Module id " << 5 << " forward" << std::endl;
	}
	// This layer is for scale**2 times channel enlargement.
	network[5]->forward(infer_memory[2], infer_memory[3], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
		std::cout << DELIMIT << std::endl 
			<< "Module id " << 6 << " forward" << std::endl;
	}
	network[6]->forward(infer_memory[3], infer_memory[4], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
		std::cout << DELIMIT << std::endl 
			<< "Module id " << 7 << " forward" << std::endl;
	}
	network[7]->forward(infer_memory[4], infer_memory[5], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
	}
	network[8]->forward(infer_memory[5], infer_memory[6], &reusable_memory);
	if(verbose)
	{
		std::cout << DELIMIT << std::endl;
	}
}

int main(int argc, char** argv)
{
	//TODO(mt): batch size is manually set to 1
    uint8_t fileData[INPUT_CH * INPUT_H * INPUT_W];
    readOpenCVFile(locateFile("0427x2.png", directories), fileData);
	
	// convert to WriteCombined memory to enable faster PCI transferring.
	void* input_host;
	CHECK(cudaHostAlloc(&input_host, INPUT_BATCH * INPUT_CH * INPUT_H * INPUT_W * sizeof(float), cudaHostAllocWriteCombined));
	// TODO(mt): accelerate this vanilla copy operations.
	for(int ch = 0; ch < INPUT_CH; ch++)
		for(int h = 0; h < INPUT_H; h++)
			for(int w = 0; w < INPUT_W; w++)
				for(int b = 0; b < INPUT_BATCH; b++)
			    	static_cast<float*>(input_host)[(b*INPUT_CH+ch)*INPUT_H*INPUT_W+h*INPUT_W+w] = float(fileData[h*INPUT_W*INPUT_CH+w*INPUT_CH+ch]);

	void* input_device;
    CHECK(cudaMalloc(&input_device, INPUT_BATCH * INPUT_CH * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMemcpy(input_device, input_host, INPUT_BATCH * INPUT_CH * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice) );
	// TODO(mt): this should have done nothing.
	CHECK(cudaDeviceSynchronize() );
	
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

	cudnnHandle_t cudnn_handle;
	CUDNN_CHECK(cudnnCreate(&cudnn_handle));

	cublasHandle_t cublas_handle;
	CUBLAS_CHECK(cublasCreate(&cublas_handle));

	CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));
	CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

	// build your network
	// TODO(mt): only network with no branch is supported now.
	std::vector<pModule> network;
	network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, INPUT_CH, INPUT_CH, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 0.0, true) ) );
	network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, INPUT_CH, N_FEAT, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 0.0, true) ) );
	//network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W,
	//						3, 3, 1, 1, 1.0, 0.0, true) ) );
	//network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W,
	//						3, 3, 1, 1, 1.0, 0.0, true) ) );
	//network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W,
	//						3, 3, 1, 1, 1.0, 0.0, true) ) );
	network.push_back(std::shared_ptr<Module>(new RG(3, INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W, 
							3, 3, 1, 1, 1.0, 0.0, true, 16) ) );
	network.push_back(std::shared_ptr<Module>(new RG(3, INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W, 
							3, 3, 1, 1, 1.0, 0.0, true, 16) ) );
	network.push_back(std::shared_ptr<Module>(new RG(3, INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W, 
							3, 3, 1, 1, 1.0, 0.0, true, 16) ) );
	network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, N_FEAT, SCALE * SCALE * N_FEAT, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 0.0, true) ) );
	network.push_back(std::shared_ptr<Module>(new PixelShuffle(SCALE, INPUT_BATCH, N_FEAT, INPUT_H, INPUT_W) ) );
	network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, N_FEAT, INPUT_CH, INPUT_H * SCALE, INPUT_W * SCALE,
							3, 3, 1, 1, 1.0, 0.0, true) ) );
	network.push_back(std::shared_ptr<Module>(new Conv2d(INPUT_BATCH, INPUT_CH, INPUT_CH, INPUT_H * SCALE, INPUT_W * SCALE,
							3, 3, 1, 1, 1.0, 0.0, true) ) );

	// auto init
	int network_depth = network.size();
	for(int i = 0; i < network_depth; i++)
	{
		std::cout << DELIMIT << std::endl
			<< "Module id " << i << " init" << std::endl;
		network[i]->init(&cudnn_handle, &cublas_handle, &stream);
		std::cout << DELIMIT << std::endl;
	}

	// Malloc inference memory
	std::vector<void*> infer_memory(7);
    CHECK(cudaMalloc(&(infer_memory[0]), INPUT_BATCH * INPUT_CH * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[1]), INPUT_BATCH * N_FEAT * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[2]), INPUT_BATCH * N_FEAT * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[3]), INPUT_BATCH * N_FEAT * SCALE * SCALE * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[4]), INPUT_BATCH * N_FEAT * SCALE * SCALE * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[5]), INPUT_BATCH * INPUT_CH * INPUT_H * SCALE * INPUT_W * SCALE * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[6]), INPUT_BATCH * INPUT_CH * INPUT_H * SCALE * INPUT_W * SCALE * sizeof(float)));

	std::vector<void*> reusable_memory(4);
    CHECK(cudaMalloc(&(reusable_memory[0]), INPUT_BATCH * N_FEAT * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(reusable_memory[1]), INPUT_BATCH * N_FEAT * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(reusable_memory[2]), INPUT_BATCH * N_FEAT * 1 * 1 * sizeof(float)));
    CHECK(cudaMalloc(&(reusable_memory[3]), INPUT_BATCH * N_FEAT * 1 * 1 * sizeof(float)));

	// TODO(mt): this should have done nothing.
	CHECK(cudaStreamSynchronize(stream) );

	// Do inference here
	// remember to alloc device memory before creating cuda stream.
	
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    std::vector<float> times(avgRuns);
    for (int j = 0; j < iterations; j++)
    {
        float totalGpu{0}, totalHost{0}; // GPU and Host timers
        for (int i = 0; i < avgRuns; i++)
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, stream);
			DoInference(network, input_device, infer_memory, reusable_memory, VERBOSE);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);

            auto tEnd = std::chrono::high_resolution_clock::now();
            totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
            totalGpu += ms;
        }
        totalGpu /= avgRuns;
        totalHost /= avgRuns;
        std::cout << "Average over " << avgRuns << " runs is " << totalGpu << " ms (host walltime is " << totalHost
                  << " ms, " << static_cast<int>(pct) << "\% percentile time is " << percentile(pct, times) << ")." << std::endl;
    }
	
	CHECK(cudaStreamSynchronize(stream) );

	// TODO(mt): get network output back?
	
	// destroy infer memory
	int memory_vec_len = infer_memory.size();
	for(int i = 0; i < memory_vec_len; i++)
	{
		CHECK(cudaFree(infer_memory[i]) );
	}
	
	cudaStreamDestroy(stream);
	
	cudnnDestroy(cudnn_handle);
	cublasDestroy(cublas_handle);

	CHECK(cudaFreeHost(input_host) );
	CHECK(cudaFree(input_device) );

    return 0;
}
