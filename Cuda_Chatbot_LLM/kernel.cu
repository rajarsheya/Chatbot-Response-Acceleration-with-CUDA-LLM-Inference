//--------------------------------------------------------------------------------------------------------------------------------------------------
// Project : Chatbot-Response-Acceleration-with-CUDA-LLM-Inference
//
// Chatbot-Response-Acceleration-with-CUDA-LLM-Inference leverages CUDA-powered ONNX Runtime to accelerate large language model inference, enabling 
// faster, real-time chatbot responses for enhanced user interactions.
//
// Author: Arsheya Raj
// Date: 15th April 2025
//--------------------------------------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

// CUDA kernel to scale output logits
__global__ void scale_output_kernel(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}

const wchar_t* model_path = L"gpt2-medium.onnx";

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "chatbot");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        Ort::Session session(env, model_path, session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<int64_t> input_ids = { 50256, 9906, 318, 617 };  // "Hi, how are"
        std::vector<int64_t> input_shape = { 1, static_cast<int64_t>(input_ids.size()) };

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = { "input_ids" };
        const char* output_names[] = { "output" };

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run ONNX inference
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

        // End timer
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Print only inference time
        std::cout << "ONNX inference completed in " << duration_ms << " ms." << std::endl;

        // CUDA post-processing (commented out)
        /*
        float* host_output = output_tensors[0].GetTensorMutableData<float>();
        int output_size = 768;  // Adjust based on model

        float* device_output;
        cudaMalloc(&device_output, output_size * sizeof(float));
        cudaMemcpy(device_output, host_output, output_size * sizeof(float), cudaMemcpyHostToDevice);

        float scale = 0.5f;
        int threads_per_block = 256;
        int blocks = (output_size + threads_per_block - 1) / threads_per_block;
        scale_output_kernel << <blocks, threads_per_block >> > (device_output, output_size, scale);
        cudaDeviceSynchronize();

        cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Scaled Output[0]: " << host_output[0] << std::endl;

        cudaFree(device_output);
        */
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
