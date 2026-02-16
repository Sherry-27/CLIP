import gradio as gr
import torch
import open_clip
import faiss
import numpy as np
import os
import json
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

TITLE = "CLIP Semantic Image Search"
DESCRIPTION = """Search images using natural language. Powered by OpenCLIP + FAISS + AWS S3."""

INDEX_DIR = "./clip_search_index"
IMAGES_DIR = "./images"  # only used for optional local check/logging

# ────────────────────────────────────────────────
# AWS S3 Setup
# ────────────────────────────────────────────────
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-south-1"  # ← CHANGE to your bucket region if different
)

BUCKET_NAME = "shaheerkhan-clip-search-images"  # ← your bucket name
IMAGE_PREFIX = "images/"  # ← folder prefix in S3 (change if different)

def get_presigned_url(key: str, expiration=3600):  # 1 hour
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"S3 presigned URL error for {key}: {e}")
        return None

# ────────────────────────────────────────────────
# Load OpenCLIP model (ViT-B/32)
# ────────────────────────────────────────────────
print("Loading OpenCLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
model.eval()
print(f"OpenCLIP loaded on {device}")

# ────────────────────────────────────────────────
# Load FAISS index
# ────────────────────────────────────────────────
print("Loading FAISS index...")
try:
    index = faiss.read_index(os.path.join(INDEX_DIR, "image_index.faiss"))
    print(f"Index loaded – {index.ntotal} vectors")
except Exception as e:
    print(f"FAISS load failed: {e}")
    index = None

# ────────────────────────────────────────────────
# Load metadata & filter (optional – for logging)
# ────────────────────────────────────────────────
print(f"Loading metadata from {INDEX_DIR}/metadata.json...")
image_paths = []
try:
    with open(os.path.join(INDEX_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
        filenames = metadata.get("image_paths", [])
    
    # Optional: check local folder (for debug only)
    existing = set(f.name for f in Path(IMAGES_DIR).glob("*.jpg"))
    for fname in filenames:
        if fname in existing:
            image_paths.append(os.path.join(IMAGES_DIR, fname))
    
    print(f"Metadata has {len(filenames)} filenames")
    print(f"Local usable images (for debug): {len(image_paths)}")
except Exception as e:
    print(f"Metadata load failed: {e}")
    image_paths = []

# ────────────────────────────────────────────────
# Search function – returns S3 presigned URLs
# ────────────────────────────────────────────────
def search(query, top_k=12):
    if not query.strip():
        return [], [], "Enter a search query"

    if index is None:
        return [], [], "FAISS index failed to load"

    try:
        tokens = open_clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features.cpu().numpy().astype("float32")

        distances, indices = index.search(query_vec, int(top_k))

        result_urls = []
        captions = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(image_paths):
                continue

            fname = os.path.basename(image_paths[idx]) if image_paths else "unknown.jpg"
            s3_key = IMAGE_PREFIX + fname

            url = get_presigned_url(s3_key)
            if url:
                result_urls.append(url)
                captions.append(f"Score: {dist:.4f} | {fname}")
            else:
                print(f"No presigned URL for {s3_key}")

        status = f"**{len(result_urls)} results** for: *{query}*"
        return result_urls, captions, status

    except Exception as e:
        return [], [], f"Search error: {str(e)}"

# ────────────────────────────────────────────────
# Gradio Interface
# ────────────────────────────────────────────────
with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        query_box = gr.Textbox(
            label="What are you looking for?",
            placeholder="a cute dog running in the park, plate of sushi...",
            lines=2,
            scale=4
        )
        k_slider = gr.Slider(4, 24, value=12, step=1, label="Number of results")

    search_button = gr.Button("Search", variant="primary")

    gallery = gr.Gallery(
        columns=4,
        height="auto",
        object_fit="contain",
        show_label=False,
        format="url"  # Required for S3 URLs
    )

    status_text = gr.Markdown()

    search_button.click(
        fn=search,
        inputs=[query_box, k_slider],
        outputs=[gallery, gallery, status_text]
    )

demo.launch()
