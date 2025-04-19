# Chatbot-Response-Acceleration-with-CUDA-LLM-Inference

🚀 **Accelerated Chatbot Inference using CUDA and ONNX Runtime**

This project demonstrates how to convert Hugging Face transformer models to ONNX format and leverage GPU acceleration via CUDA and ONNX Runtime to significantly boost chatbot response time on local servers. It enables faster real-time interactions in customer service and conversational AI applications.

---

## Demo Video - https://youtu.be/_y9JjciFu6E

## Project Highlights

- Converts GPT-based language models (e.g., `gpt2`, `gpt2-medium`, `distilgpt2`, etc.) to ONNX format  
- Uses `CUDA` backend through ONNX Runtime for GPU-accelerated inference  
- Optimized for real-time chatbot interactions on local machines  
- Suppresses unnecessary tracing warnings for clean conversion  
- Fully compatible with Windows + Visual Studio + ONNX Runtime GPU 1.21.0  

---

## Models Supported

You can choose from several Hugging Face models:
- `distilgpt2`
- `gpt2`
- `gpt2-medium`