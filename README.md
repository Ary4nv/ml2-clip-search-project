# Multimodal Semantic Image Search with CLIP + FAISS

**CSCI 4052U – Machine Learning II Final Project**  
**Group ID: 11**

---

## 👥 Team Members

| Name        | Student ID | Email                       |
| ----------- | ---------- | --------------------------- |
| Arian Vares | 100882708  | Arian.vares@ontariotechu.ca |
| Member 2    | XXXXXXXX   | email2@ontariotechu.ca      |
| Member 3    | XXXXXXXX   | email3@ontariotechu.ca      |

---

## 📌 Project Overview

This project implements a **multimodal semantic image search system** that retrieves relevant images using natural language queries.
The system leverages:

- **CLIP (Contrastive Language-Image Pretraining)** for joint image-text embeddings
- **FAISS** for efficient similarity search
- **Gradio** for an interactive user interface
  Given a query such as `"a red car"` or `"dog on grass"`, the system returns the top-k most semantically similar images from a dataset.

---

## ❓ Problem Formulation

Traditional image retrieval methods rely on:

- Keywords and tags
- Manual annotations
- Metadata matching
  These approaches struggle with:
- Understanding semantic meaning
- Generalizing to unseen queries
- Scaling without labeled data

### 🔥 Key Challenge

Mapping images and text into a shared semantic space is difficult using traditional approaches.

### 🤖 Neural Network Solution

We use **CLIP**, a pretrained neural network that:

- Learns joint embeddings for images and text
- Enables direct comparison between text and images
- Supports zero-shot learning without task-specific training

---

## 🧠 Neural Network Component

### Model Used

- **CLIP (ViT-B/32)** from OpenAI
- Source: https://huggingface.co/openai/clip-vit-base-patch32

### Architecture

CLIP consists of:

- **Vision Transformer (ViT)** for images
- **Transformer encoder** for text
- Projection layers mapping both into a shared embedding space

### Training

- Trained on **400M+ image-text pairs**
- Uses contrastive learning

### Usage in This Project

- No additional training or fine-tuning
- Pretrained model used for inference only

---

## ⚙️ End-to-End Application Pipeline

CIFAR-10 images ──► CLIP image encoder ──► L2 normalize ──► FAISS IndexFlatIP
▲
“a red car” ──────► CLIP text encoder ──► L2 normalize ───────────┘
│
top-k similar

### Steps

1. Load CIFAR-10 dataset (500 images)
2. Encode images using CLIP image encoder
3. Normalize embeddings
4. Store embeddings in FAISS index
5. Encode text query using CLIP text encoder
6. Retrieve nearest neighbors using FAISS
7. Display results in Gradio UI

---

## 🔌 Software Architecture

User Query → CLIP Text Encoder → Embedding
↓
FAISS Search Index
↓
Retrieved Image Embeddings → Image Paths
↓
Gradio UI Display

---

## 📦 Deployment

### Environment Options

- **Google Colab (recommended)**
  - T4 GPU (16GB VRAM)
  - Works with notebook + Gradio
- **Local Machine**
  - CPU or GPU supported

---

## 🚀 How to Run

### Option 1: Google Colab

1. Open `clip_faiss_search.ipynb`
2. Enable GPU (Runtime → T4)
3. Run all cells
4. Run:

!python app.py

5. Click the Gradio public link

⸻

Option 2: Local Machine

Install dependencies:

pip install transformers faiss-cpu gradio torch torchvision pillow numpy matplotlib

Run:

python app.py

⸻

📂 Project Structure

ML2_final/
├── app.py # Gradio UI application
├── clip_faiss_search.ipynb # Builds dataset + embeddings + FAISS index
├── project_data/
│ ├── images/ # CIFAR-10 images
│ ├── image_embeddings.npy # Cached embeddings
│ ├── image_paths.pkl # Paths aligned with embeddings
│ └── faiss_index.bin # FAISS index
└── README.md

⸻

🖼️ Application Features

- Text-to-image search using natural language
- Top-k nearest neighbor retrieval
- Similarity scores displayed
- Interactive UI via Gradio
- Example queries for quick testing

⸻

🔍 Example Queries

- a red car
- dog on grass
- a ship on the ocean
- small airplane in the sky
- horse in a field

⸻

📸 Screenshots

⸻

🎥 Video Demonstration

⸻

🧾 Design Decisions

- FAISS IndexFlatIP used for exact search (efficient for small datasets)
- L2 normalization ensures cosine similarity
- 512-dimensional embeddings from CLIP
- Text caching added for faster repeated queries

⸻

⚠️ Limitations

- CIFAR-10 images are low resolution (32×32)
- No fine-tuning of CLIP model
- Exact search does not scale to very large datasets
- No advanced ranking or reranking

⸻

📊 Conclusion

This project demonstrates how pretrained neural networks can be integrated into a full AI system. By combining CLIP embeddings with FAISS, we achieve efficient and scalable semantic image retrieval.

⸻

```

```
