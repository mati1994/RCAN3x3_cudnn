#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
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

static const int INPUT_BATCH = 1;
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

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	;
}

typedef struct _cudnnConvInfo 
{
	cudnnHandle_t* handle;
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
			cudnnHandle_t* handle, cudaStream_t* stream)
{
	if(DEBUG)
	{
		std::cout << "Debug, ni = " << ni << ", ci = " << ci << ", co = " << co << ", hi = " << hi << ", wi = " << wi << std::endl
			<< ", k_h = " << k_h << ", k_w = " << k_w << ", p_h = " << p_h << ", p_w = " << p_w << std::endl;
	}
	conv_info->handle = handle;
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
	cudaStream_t* stream;

	cudnnPoolingDescriptor_t pool_desc;
	cudnnTensorDescriptor_t input_desc, output_desc;
} _cudnnGAPInfo;

// API consistent with InitConv 
void InitGAP(_cudnnGAPInfo* gap_info, 
			int ni, int ci, int co, int hi, int wi, 
			int k_h, int k_w, int p_h, int p_w, 
			cudnnHandle_t* handle, cudaStream_t* stream)
{
	if(DEBUG)
	{
		std::cout << "Debug, ni = " << ni << ", ci = " << ci << ", co = " << co << ", hi = " << hi << ", wi = " << wi << std::endl
			<< ", k_h = " << k_h << ", k_w = " << k_w << ", p_h = " << p_h << ", p_w = " << p_w << std::endl;
	}
	gap_info->handle = handle;
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
	CHECK_EQ(hi, 1);
	CHECK_EQ(wi, 1);
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

class Module
{
public:
	Module() {}
	~Module() {}

	virtual void init(cudnnHandle_t* handle, cudaStream_t* stream) = 0;

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

	virtual void init(cudnnHandle_t* handle, cudaStream_t* stream)
	{
		InitConv(&(this->conv_info),
			NI, CI, CO, HI, WI,
			K_H, K_W, P_H, P_W,
			handle, stream);
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

	virtual void init(cudnnHandle_t* handle, cudaStream_t* stream)
	{
		InitGAP(&(this->gap_info),
			NI, CI, CO, HI, WI,
			K_H, K_W, P_H, P_W,
			handle, stream);
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

class RCAB : public Module
{
public:
	RCAB(int ni, int ci, int co, int hi, int wi, int k_h, int k_w, int p_h, int p_w, float alpha=1.0, float beta=0.0, bool bias=true)
	{
		main_conv.push_back(std::shared_ptr<Conv2d>(new Conv2d(ni, ci, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, true) ) );
		main_conv.push_back(std::shared_ptr<Conv2d>(new Conv2d(ni, ci, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, true) ) );

		//main_conv.push_back(std::make_shared(new Conv2d(ni, ci, co, hi, wi, k_h, k_w, p_h, p_w, 1.0, 0.0, true) ) );
	}

	~RCAB() {}

	virtual void init(cudnnHandle_t* handle, cudaStream_t* stream)
	{
		main_conv[0]->init(handle, stream);
		main_conv[1]->init(handle, stream);
	}

	// NOTE that for RCAB, y could be the same as x to maximize memory usage.
	virtual void forward(void* x, void* y, std::vector<void*>* reusable_memory)
	{
		int main_conv_len = reusable_memory->size();
		CHECK_GE(main_conv_len, 2);
		std::vector<void*> dummy_reusable_memory;
		main_conv[0]->forward(x, (*reusable_memory)[0], &dummy_reusable_memory);
		main_conv[1]->forward((*reusable_memory)[0], (*reusable_memory)[1], &dummy_reusable_memory);

	}

public:
	std::vector<pModule> main_conv;
	std::vector<pModule> fc_conv;
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

	CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));

	// build your network
	// TODO(mt): only network with no branch is supported now.
	std::vector<pModule> network;
	network.push_back(std::shared_ptr<Conv2d>(new Conv2d(INPUT_BATCH, INPUT_CH, INPUT_CH, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 1.0, true) ) );
	network.push_back(std::shared_ptr<Conv2d>(new Conv2d(INPUT_BATCH, INPUT_CH, N_FEAT, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 1.0, true) ) );
	network.push_back(std::shared_ptr<Conv2d>(new Conv2d(INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 1.0, true) ) );
	network.push_back(std::shared_ptr<Conv2d>(new Conv2d(INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 1.0, true) ) );
	network.push_back(std::shared_ptr<Conv2d>(new Conv2d(INPUT_BATCH, N_FEAT, N_FEAT, INPUT_H, INPUT_W,
							3, 3, 1, 1, 1.0, 1.0, true) ) );

	// auto init
	int network_depth = network.size();
	for(int i = 0; i < network_depth; i++)
	{
		std::cout << DELIMIT << std::endl
			<< "Module id " << i << " init" << std::endl;
		network[i]->init(&cudnn_handle, &stream);
		std::cout << DELIMIT << std::endl;
	}

	// Malloc inference memory
	std::vector<void*> infer_memory(3);
    CHECK(cudaMalloc(&(infer_memory[0]), INPUT_BATCH * INPUT_CH * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[1]), INPUT_BATCH * N_FEAT * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(infer_memory[2]), INPUT_BATCH * N_FEAT * INPUT_H * INPUT_W * sizeof(float)));

	std::vector<void*> reusable_memory(2);
    CHECK(cudaMalloc(&(reusable_memory[0]), INPUT_BATCH * INPUT_CH * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&(reusable_memory[1]), INPUT_BATCH * INPUT_CH * INPUT_H * INPUT_W * sizeof(float)));

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
	CHECK(cudaFreeHost(input_host) );
	CHECK(cudaFree(input_device) );

    return 0;
}
