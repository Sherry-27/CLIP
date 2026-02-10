%%writefile hf_space_files/app.py
import gradio as gr
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path

# =====================================
# CONFIG
# =====================================
TITLE = "CLIP Semantic Image Search"
DESCRIPTION = """Search ~13–14k images (COCO val2017 + Flickr30k) using natural language.
Powered by OpenAI CLIP (ViT-B/32) + FAISS."""

INDEX_DIR = "./clip_search_index"
IMAGES_DIR = "./images"

# =====================================
# Load model & index
# =====================================
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading FAISS index...")
index = faiss.read_index(os.path.join(INDEX_DIR, "image_index.faiss"))

with open(os.path.join(INDEX_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)
    image_paths = metadata["image_paths"]

# Fix paths to be relative to current working directory
base_image_dir = os.path.abspath(IMAGES_DIR)
image_paths_fixed = []
for p in image_paths:
    fname = os.path.basename(p)
    full_path = os.path.join(base_image_dir, fname)
    image_paths_fixed.append(full_path)

print(f"Loaded {len(image_paths_fixed)} image references")

# =====================================
# Search function
# =====================================
def search(query, top_k=12):
    if not query.strip():
        return [], "Please enter a search query"

    try:
        # Encode text
        tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features.cpu().numpy().astype("float32")

        # Search
        distances, indices = index.search(query_vec, int(top_k))

        results = []
        captions = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(image_paths_fixed):
                continue
            img_path = image_paths_fixed[idx]
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                results.append(img)
                captions.append(f"Score: {dist:.4f}")
            else:
                # Skip missing images silently in production
                continue

        status = f"**{len(results)} results** for: *{query}*"
        return results, status

    except Exception as e:
        return [], f"Error: {str(e)}"


# =====================================
# Gradio Interface
# =====================================
with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        query_box = gr.Textbox(
            label="What are you looking for?",
            placeholder="a cute dog playing in the park, delicious sushi plate, woman in red dress...",
            lines=2,
            scale=4
        )
        k_slider = gr.Slider(
            4, 24, value=12, step=1,
            label="Number of results"
        )

    search_button = gr.Button("Search", variant="primary")

    gallery = gr.Gallery(
        label="Results",
        columns=4,
        height="auto",
        object_fit="contain",
        show_label=False
    )

    status_text = gr.Markdown()

    # Examples
    gr.Examples(
        examples=[
            ["a cute dog running in the park"],
            ["person riding a bicycle on the street"],
            ["beautiful mountain landscape with snow"],
            ["plate of delicious sushi"],
            ["woman wearing elegant red dress"]
        ],
        inputs=query_box
    )

    # Actions
    search_button.click(
        fn=search,
        inputs=[query_box, k_slider],
        outputs=[gallery, status_text]
    )

demo.launch()