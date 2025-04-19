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

import warnings
warnings.filterwarnings("ignore", message="Unsupported Windows version")
warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")  # Suppress the TracerWarning related to boolean conversion

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.onnx import export
import onnx

# Load pre-trained model and tokenizer from Hugging Face
# You can change this to any other model (e.g., "gpt2", "gpt-neo", "distilgpt2", "gpt2-medium" etc.)
#model_name = "distilgpt2"
#model_name = "gpt2"
model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define dummy input for the model
dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")

# Specify the ONNX output file
onnx_model_path = f"{model_name}.onnx"

# Export the model to ONNX format with Opset version 14
torch.onnx.export(
    model,  # Model to export
    (dummy_input['input_ids'],),  # Input tuple (matching input signature)
    onnx_model_path,  # Path to save the ONNX model
    input_names=['input_ids'],  # Input name(s)
    output_names=['output'],  # Output name(s)
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},  # Dynamic axes
                  'output': {0: 'batch_size', 1: 'sequence_length'}},
    opset_version=17  # Update to opset version 17
)

# Verify the model is successfully converted to ONNX format
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

print(f"Model successfully converted to ONNX and saved to {onnx_model_path}")
