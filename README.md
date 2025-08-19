# 🤖 AI Craftsman Portfolio

Welcome to my portfolio! I'm an AI developer with a passion for building intelligent systems from the ground up. This repository showcases a collection of projects that reflect a deep understanding of foundational model architectures and their application to solve real-world problems in natural language processing and computer vision.

My approach combines theoretical knowledge with hands-on implementation, focusing on creating clean, efficient, and well-documented code. Below are some highlights of my work.

---

## 🚀 Projects

### 1. A GPT-2 Style Transformer From Scratch in PyTorch

This project is a deep dive into the architecture that powers modern large language models. I implemented a decoder-only transformer, the foundational design of GPT models, entirely from scratch using PyTorch. The goal was to build a comprehensive understanding of the mechanics behind text generation.



* **Architecture**: A decoder-only transformer built by stacking multiple blocks, each containing:
    * **Causal Self-Attention**: A multi-head attention mechanism with a causal mask to ensure the model cannot "see" future tokens during training. It allows each token to gather context from previous tokens in the sequence.
    * **Multi-Layer Perceptron (MLP)**: A two-layer feed-forward network with a GELU activation, used to process information from the attention layer.
* **Key Components**: Implemented core building blocks like **Layer Normalization**, **token and positional embeddings**, and **residual connections** to ensure stable training and robust performance.
* **Technology**: Python, PyTorch.
* **Status**: Complete. The implementation successfully demonstrates the core principles of generative pre-trained transformers.

> **[Link to the GPT-2 Project Repository &raquo;](https://github.com/omhome16/My-GPT-2-from-scratch)**

---

### 2. 💹 FinEdge RAG Chatbot

**FinEdge** is an intelligent, conversational AI assistant designed for financial analysis. It leverages a Retrieval-Augmented Generation (RAG) pipeline to answer questions based *only* on a provided set of financial documents. This chatbot provides precise answers, cites the exact source page for verification, and maintains a professional, analytical tone throughout the conversation.

## 🌟 Key Features

* **Conversational AI with Memory**: Remembers previous parts of the conversation for context-aware follow-up questions.
* **Source Verification**: Displays an image of the exact page from the source PDF document, allowing users to instantly verify the information.
* **Secure & Private**: Operates exclusively on the documents you provide, ensuring your data remains private and the AI doesn't use external knowledge.
* **Professional Persona**: The AI is prompted to act as a highly knowledgeable financial analyst, ensuring professional and relevant responses.
* **Interactive UI**: A clean, user-friendly chat interface built with Streamlit.

## 🛠️ Technology Stack

* **Backend**: Python
* **AI/ML Framework**: LangChain
* **LLM**: Google Gemini (`gemini-1.5-flash-latest`)
* **Embeddings**: Hugging Face `sentence-transformers`
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **Frontend**: Streamlit
* **PDF Processing**: PyMuPDF (`fitz`)
* **Environment Management**: `python-dotenv`

> **[Link to the FineEdge-RAG-Chatbot repository &raquo;](https://github.com/omhome16/FinEdge-RAG-chatbot)**

---

### 3. Automated Segmentation of LGG Brain Tumors

This project applies deep learning to the critical medical task of brain tumor segmentation. I developed and compared two powerful convolutional neural network architectures, **U-NET** and **Attention U-NET**, to automatically identify Low-Grade Glioma (LGG) regions in multi-modal MRI scans.



* **Objective**: To build a robust pipeline for segmenting LGG tumors from the BraTS dataset (T1, T1ce, T2, FLAIR modalities).
* **Models Compared**:
    * **U-NET**: A classic encoder-decoder architecture for biomedical image segmentation.
    * **Attention U-NET**: An enhanced U-NET that incorporates attention gates to help the model focus on more relevant feature regions.
* **Key Achievement**: Through rigorous hyperparameter tuning (exploring 24 combinations of optimizers, learning rates, and schedulers), the standard **U-NET** architecture significantly outperformed the Attention U-NET.
* **Best Result**: Achieved a **Dice Score of 0.8726** with the U-NET model after 50 epochs, demonstrating its effectiveness for this specific task.
* **Technology**: Python, PyTorch, U-NET, Attention U-NET, Adam/AdamW, CosineAnnealingLR.

> **[Link to the Brain Tumor Segmentation Project Repository &raquo;](https://github.com/omhome16/lgg-MRI-medical-image-segmentation)**

---

### 4. Crop Disease Segmentation using U-Net & Nested U-Net (U-Net++)

This project addresses a key challenge in agriculture: the early and accurate detection of crop diseases. I built a flexible deep learning pipeline for pixel-wise segmentation of diseased areas on plant leaves, implementing both U-Net and the more advanced Nested U-Net (U-Net++).



* **Objective**: To create a modular and configurable system for identifying crop diseases from high-resolution leaf images.
* **Models Implemented**:
    * **U-Net**: With optional attention gates and configurable upsampling methods (Bilinear or Transposed Convolutions).
    * **Nested U-Net (U-Net++)**: Features dense skip connections and support for deep supervision to improve gradient flow and feature fusion.
* **Key Features**:
    * The entire training pipeline is configurable via a single `config.yaml` file.
    * Includes robust checkpointing and logging for tracking experiments.
* **Best Result**: Achieved a **Dice Score of 0.6906** after 50 epochs of training.
* **Technology**: Python, PyTorch, U-Net, Nested U-Net (U-Net++), Attention Gates.

> **[Link to the Crop Disease Segmentation Project Repository &raquo;](https://github.com/omhome16/crop-disease-segmentation)**

---

## 📫 Connect with Me

I am always open to discussing new projects, sharing ideas, or collaborating on innovative AI solutions.

* **LinkedIn**: `[Your LinkedIn Profile]([https://www.linkedin.com/in//](https://www.linkedin.com/in/om-nawale-7b8722289/))`
* **GitHub**: `[omhome16](https://github.com/omhome16)`


