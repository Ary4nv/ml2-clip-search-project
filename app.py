import os
import pickle

import faiss
import gradio as gr
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = "project_data"
PATHS_FILE = os.path.join(DATA_DIR, "image_paths.pkl")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "image_embeddings.npy")
MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

if not os.path.exists(PATHS_FILE):
    raise FileNotFoundError(
        f"{PATHS_FILE} not found. Run clip_faiss_search.ipynb first to build cached artifacts."
    )

with open(PATHS_FILE, "rb") as f:
    image_paths = pickle.load(f)

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    print("Loaded FAISS index from disk. ntotal =", index.ntotal)
elif os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print("Rebuilt FAISS index from embeddings. ntotal =", index.ntotal)
else:
    raise FileNotFoundError(
        f"Neither {INDEX_FILE} nor {EMBEDDINGS_FILE} found. "
        "Run clip_faiss_search.ipynb first to build cached artifacts."
    )

text_cache = {}
def get_text_embedding(text):
    if text in text_cache:
        return text_cache[text]

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    features = features / torch.norm(features, dim=-1, keepdim=True)
    features = features.cpu().numpy().astype("float32")

    text_cache[text] = features
    return features


def search(query, k=5):
    vec = get_text_embedding(query)
    scores, idxs = index.search(vec, k)
    return [(image_paths[i], float(s)) for i, s in zip(idxs[0], scores[0])]


def gradio_search(query, k):
    if not query or not query.strip():
        return []
    results = search(query, int(k))
    from PIL import Image
    return [(Image.open(path), f"score: {score:.4f}") for path, score in results]


with gr.Blocks(title="CLIP + FAISS Image Search") as demo:
    gr.Markdown("# CLIP + FAISS Image Search")
    gr.Markdown("Text-to-image retrieval over 500 CIFAR-10 images.")

    with gr.Row():
        query_box = gr.Textbox(
            label="Query",
            placeholder="e.g. 'a red car' or 'dog on grass'",
        )
        k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="k")

    search_btn = gr.Button("Search")
    gallery = gr.Gallery(label="Results", columns=5, height="auto")

    search_btn.click(fn=gradio_search, inputs=[query_box, k_slider], outputs=gallery)
    query_box.submit(fn=gradio_search, inputs=[query_box, k_slider], outputs=gallery)


if __name__ == "__main__":
    demo.launch(share=True)
