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
#   Optimized the speed of chatbot responses by using CUDA to accelerate the inference of a language model. This allowed for faster customer
#  service interactions on local servers.
#
#--------------------------------------------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", message="Unsupported Windows version")

import time
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Function to run inference and measure time
def run_inference_with_timing(model_path, prompt, max_length=50):
    # Start the timer
    start_time = time.time()

    # Load the tokenizer (same as the model you used)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs['input_ids'].astype(np.int64)  # Convert to int64

    # Initialize the ONNX Runtime session
    session = ort.InferenceSession(model_path)

    # Start with the initial input tokens (like in the CUDA code)
    generated_ids = input_ids

    # Loop to generate up to max_length tokens (similar to the CUDA kernel generation)
    for _ in range(max_length):
        # Run inference
        ort_inputs = {session.get_inputs()[0].name: generated_ids}
        ort_output = session.run(None, ort_inputs)

        # Get the logits from the output (same as CUDA code's output processing)
        logits = ort_output[0][:, -1, :]  # Last token's logits

        # Pick the most probable next token (like np.argmax in CUDA code)
        next_token_id = np.argmax(logits, axis=-1)

        # Append the next token to the sequence
        generated_ids = np.concatenate([generated_ids, next_token_id.reshape(1, 1)], axis=-1)

        # Stop early if EOS token is generated (like in the CUDA approach)
        if next_token_id == tokenizer.eos_token_id:
            break

    # Measure the inference time
    inference_time = time.time() - start_time

    # Return the inference time (no need to print the scaled output)
    return inference_time


# Example usage
if __name__ == "__main__":
    model_path = "gpt2-medium.onnx"  # Path to your ONNX model
    prompt = "What is the capital of France?"

    # Call the inference function and get the inference time
    inference_time = run_inference_with_timing(model_path, prompt)

    # Print the inference time (in milliseconds)
    print(f"ONNX inference completed in {inference_time * 1000:.2f} ms.")
