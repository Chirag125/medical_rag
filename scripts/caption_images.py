"""
scripts/caption_images.py — BLIP-2 captioning for the figures document_loader.py
already extracts and saves to data/processed/images/.

Why this needs no new retrieval code: a caption is just text, so it goes
through the exact same BGE embedder and ChromaDB collection as everything
else. It's stored as a child chunk like any other, with metadata pointing
back at the image_id it describes — the reranker and Retriever class don't
need to know the difference between a caption chunk and a text chunk.

This does NOT touch your existing text chunks or collections — it ADDS
caption chunks to the same child_chunks/parent_chunks collections, so run
this AFTER your normal ingestion (scripts/ingest.py), not instead of it.

Usage:
    python scripts/caption_images.py
    python scripts/caption_images.py --image-dir data/processed/images --chroma-path data/processed/chroma
"""

import argparse
import os
import sys
import uuid

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.append(".")

from pathlib import Path


def get_captioner():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch

    model_name = "Salesforce/blip2-opt-2.7b"
    print(f"Loading {model_name} (this is a large model — first run downloads several GB) ...")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to("cpu")
    return processor, model


def caption_image(image_path, processor, model):
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=40)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def main():
    parser = argparse.ArgumentParser(description="Caption extracted images and add them as searchable chunks")
    parser.add_argument("--image-dir", default="data/processed/images")
    parser.add_argument("--chroma-path", default="data/processed/chroma")
    args = parser.parse_args()

    image_paths = sorted(Path(args.image_dir).glob("*.png"))
    if not image_paths:
        print(f"No images found in {args.image_dir} — nothing to caption. "
              f"(This is expected if your ingested PDFs so far had no extractable images.)")
        return

    print(f"Found {len(image_paths)} images to caption")
    processor, model = get_captioner()

    from src.ingestion.embedder import TextEmbedder
    embedder = TextEmbedder(chroma_path=args.chroma_path)

    caption_parents, caption_children = [], []
    for image_path in image_paths:
        image_id = image_path.stem  # e.g. "doc_p3_img0"
        # image_id format from document_loader.py is "{doc_name}_p{page_num}_img{n}"
        try:
            doc_part, page_part, _ = image_id.rsplit("_", 2)
            page_num = int(page_part.lstrip("p"))
        except (ValueError, IndexError):
            doc_part, page_num = image_id, -1

        caption = caption_image(image_path, processor, model)
        print(f"  {image_id}: {caption}")

        chunk_id = str(uuid.uuid4())
        text = f"[Figure caption, {doc_part} page {page_num}]: {caption}"
        chunk = {
            "id": chunk_id,
            "parent_id": chunk_id,
            "text": text,
            "doc_name": doc_part,
            "page_num": page_num,
            "token_count": len(caption.split()),
            "image_ids": image_id,  # ties this caption chunk back to the exact image
        }
        caption_parents.append(chunk)
        caption_children.append(chunk)

    embedder.store_parents(caption_parents)
    embedder.store_children(caption_children)
    embedder.get_collection_stats()
    print(f"\nAdded {len(caption_children)} caption chunks. "
          f"Queries about figures should now be able to surface them alongside text chunks.")


if __name__ == "__main__":
    main()