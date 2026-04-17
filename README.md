# Multimodal Semantic Image Search with CLIP + FAISS

A text-to-image retrieval system that finds semantically relevant images from natural language queries. Built with OpenAI's CLIP model for embedding, FAISS for efficient similarity search, and Gradio for the web interface.

## Overview

Given a query like `"a red car"` or `"dog on grass"`, the system returns the top-k most similar images from an indexed collection. It works by projecting both text and images into CLIP's shared embedding space, then using FAISS inner-product search over L2-normalized vectors (equivalent to cosine similarity).

The demo indexes the first 500 images from the CIFAR-10 test split, but the pipeline works with any image collection.

## How It Works

```
 CIFAR-10 images ──► CLIP image encoder ──► L2 normalize ──► FAISS IndexFlatIP
                                                                    ▲
 "a red car" ──────► CLIP text encoder  ──► L2 normalize ───────────┘
                                                                    │
                                                              top-k similar
```

1. Each image is encoded with `CLIPModel.get_image_features` and L2-normalized.
2. Normalized vectors are stored in a FAISS `IndexFlatIP` (inner product = cosine similarity on unit vectors).
3. At query time the text is encoded the same way and the index returns the nearest neighbors.

## Project Structure

```
ML2_final/
├── app.py                      # Gradio app serving the search UI
├── clip_faiss_search.ipynb     # Notebook that builds the dataset, embeddings, and FAISS index
├── project_data/
│   ├── images/                 # 500 CIFAR-10 PNGs (created by the notebook)
│   ├── image_embeddings.npy    # Cached (N, 512) float32 embeddings
│   ├── image_paths.pkl         # Ordered list of image paths aligned with embeddings
│   └── faiss_index.bin         # Serialized FAISS index
└── README.md
```

The `ml2/` directory is a local Python virtual environment and is not required to be committed.

## Setup

Requires Python 3.10+.

```bash
python -m venv ml2
source ml2/Scripts/activate        # Windows (bash)
# source ml2/bin/activate          # Linux/macOS

pip install transformers faiss-cpu gradio torch torchvision pillow numpy matplotlib
```

## Usage

### 1. Build the index (first run only)

Open `clip_faiss_search.ipynb` and run all cells. This will:

- Download CIFAR-10 into `project_data/`
- Save 500 PNGs to `project_data/images/`
- Load `openai/clip-vit-base-patch32`
- Compute and cache image embeddings
- Build and persist the FAISS index

Subsequent runs detect the cached artifacts and skip the encoding pass.

### 2. Launch the search app

```bash
python app.py
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`). Open it, enter a query, and adjust `k` to control how many results are returned.

The app refuses to start if the cached artifacts are missing — run the notebook first.

## Example Queries

- `a red car`
- `dog on grass`
- `a ship on the ocean`
- `small airplane in the sky`
- `horse in a field`

CIFAR-10 is 32×32, so expect coarse matches rather than fine-grained retrieval.

## Tech Stack

| Component | Choice |
|-----------|--------|
| Embedding model | `openai/clip-vit-base-patch32` via `transformers` |
| Vector search | FAISS `IndexFlatIP` (exact, cosine via inner product) |
| Dataset | CIFAR-10 test split (first 500 images) |
| UI | Gradio |
| Compute | PyTorch (CUDA if available, otherwise CPU) |

## Design Notes

- **Why `IndexFlatIP`?** At 500 vectors exact search is effectively free. For larger corpora, swap in `IndexIVFFlat` or `IndexHNSWFlat` without changing the query code.
- **Why L2-normalize?** Makes inner product equal to cosine similarity, which is the similarity CLIP was trained under.
- **Embedding dimension:** 512 (CLIP ViT-B/32).
- **Device handling:** `app.py` and the notebook both auto-select CUDA if available and fall back to CPU.

## Limitations

- Hardcoded to 500 CIFAR-10 images; extending to a new corpus requires re-running the embedding step.
- Exact search only — will not scale past ~100k vectors without switching index type.
- No query caching; each text query triggers a CLIP forward pass.
