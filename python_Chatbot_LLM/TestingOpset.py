#--------------------------------------------------------------------------------------------------------------------------------------------------
# Project : Chatbot-Response-Acceleration-with-CUDA-LLM-Inference
#
# Chatbot-Response-Acceleration-with-CUDA-LLM-Inference leverages CUDA-powered ONNX Runtime to accelerate large language model inference, enabling
# faster, real-time chatbot responses for enhanced user interactions.
#
# Author: Arsheya Raj
# Date: 15th April 2025
#--------------------------------------------------------------------------------------------------------------------------------------------------
#
#	Optimized the speed of chatbot responses by using CUDA to accelerate the inference of a language model. This allowed for faster customer
#  service interactions on local servers.
#
#--------------------------------------------------------------------------------------------------------------------------------------------------

import onnx
model = onnx.load("gpt2-medium.onnx")
print(f"Exported model opset version: {model.opset_import[0].version}")
