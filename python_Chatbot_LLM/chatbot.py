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

import onnx
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Load the ONNX model
#onnx_model_path = "distilgpt2"
#onnx_model_path = "gpt2"
onnx_model_path = "gpt2-medium.onnx"
onnx_model = onnx.load(onnx_model_path)

# Check the model
onnx.checker.check_model(onnx_model)

# Set up the ONNX Runtime session
ort_session = ort.InferenceSession(onnx_model_path)

# Load the tokenizer (same as the model you used)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to generate a response
def generate_response(prompt, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs['input_ids'].astype(np.int64)  # Convert to int64

    # Start with the initial input tokens
    generated_ids = input_ids

    # Loop to generate up to max_length tokens
    for _ in range(max_length):
        # Perform inference using ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: generated_ids}
        ort_output = ort_session.run(None, ort_inputs)

        # Get the logits from the model's output
        logits = ort_output[0][:, -1, :]  # We need the last token's logits

        # Use np.argmax for greedy decoding (pick the most likely token)
        next_token_id = np.argmax(logits, axis=-1)

        # Reshape the next token ID to have the same dimensions as generated_ids
        next_token_id = next_token_id.reshape(1, 1)

        # Append the next token to the generated sequence
        generated_ids = np.concatenate([generated_ids, next_token_id], axis=-1)

        # If the model generates the end token, stop early
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode the generated tokens back to text
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return response

# Chat loop
print("Chatbot is ready! (type 'exit' to stop)")

while True:
    # Get user input
    user_input = input("You: ")

    # Exit the loop if the user types 'exit'
    if user_input.lower() == 'exit':
        break

    # Generate a response from the chatbot
    response = generate_response(user_input)

    # Print the response
    print(f"Chatbot: {response}")
