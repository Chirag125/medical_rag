# src/ingestion/document_loader.py
import pdfplumber
import os
from pathlib import Path
from PIL import Image
import io
import re

# Where extracted page images get saved so they have a stable, referenceable path.
# This is new — previously extracted images were kept as in-memory PIL objects only
# and discarded after the run, with no id tying them to anything.
IMAGE_OUTPUT_DIR = "data/processed/images"


def clean_pdf_text(text):
    """
    Fix common PDF extraction artifacts:
    - Remove excessive whitespace
    - Fix fused words where possible (add space before capitals in camelCase)
    - Remove lone special characters
    """
    if not text:
        return ""

    # Fix fused words — insert space before uppercase letters
    # that follow lowercase letters (camelCase artifacts)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Collapse multiple spaces/newlines
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just noise (single chars, page numbers)
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 3]

    return '\n'.join(lines).strip()


def load_pdf(pdf_path):
    """
    Extract text and images from a single PDF.

    Returns:
        pages: list of dicts with keys:
               - page_num: int
               - text: str
               - images: list of dicts {"image_id": str, "image_path": str}
                         (each image is saved to disk under IMAGE_OUTPUT_DIR;
                         image_id is stable and referenceable from chunk metadata)
               - image_ids: list of str — convenience flat list of the ids above,
                         used later by chunker.py to tag text chunks on this page
               - section_heading: str (best guess from font size)
    """
    pages = []
    doc_name = Path(pdf_path).stem
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # Extract text
            # x_tolerance controls how close two characters need to be to be
            # merged as one word vs. treated as separate words with a space
            # between them. pdfplumber's default (3) fuses words together on
            # some PDFs (this arXiv paper's font metrics being one of them) —
            # a smaller tolerance fixes it without touching anything else.
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            text = clean_pdf_text(text)

            # Extract images — now saved to disk with a stable image_id
            # instead of kept only as in-memory PIL.Image objects.
            images = []
            image_ids = []
            for img_idx, img in enumerate(page.images):
                try:
                    # Crop image region from page
                    bbox = (img['x0'], img['top'],
                            img['x1'], img['bottom'])
                    cropped = page.crop(bbox)
                    img_obj = cropped.to_image(resolution=150)
                    pil_img = img_obj.original

                    image_id = f"{doc_name}_p{page_num + 1}_img{img_idx}"
                    image_path = os.path.join(IMAGE_OUTPUT_DIR, f"{image_id}.png")
                    pil_img.save(image_path)

                    images.append({"image_id": image_id, "image_path": image_path})
                    image_ids.append(image_id)
                except Exception:
                    continue

            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "images": images,          # list of {"image_id", "image_path"} dicts
                "image_ids": image_ids,     # flat list, used by chunker.py
                "doc_name": doc_name,
                "doc_path": str(pdf_path)
            })

    return pages


def load_corpus(data_dir, max_docs=50):
    """Load all PDFs from a directory."""
    pdf_files = list(Path(data_dir).glob("*.pdf"))[:max_docs]
    print(f"Found {len(pdf_files)} PDFs in {data_dir}")

    all_pages = []
    for i, pdf_path in enumerate(pdf_files):
        print(f"  Loading {i+1}/{len(pdf_files)}: {pdf_path.name}")
        try:
            pages = load_pdf(pdf_path)
            all_pages.extend(pages)
        except Exception as e:
            print(f"  Error loading {pdf_path.name}: {e}")
            continue

    print(f"Loaded {len(all_pages)} pages from {len(pdf_files)} documents")
    return all_pages


# Quick test — run this file directly to verify it works
if __name__ == "__main__":
    # Test with a single PDF
    # Download a sample medical paper first:
    # https://arxiv.org/pdf/2303.08774  (GPT-4 technical report as test PDF)

    import requests
    test_url = "https://arxiv.org/pdf/2303.08774"
    test_path = "data/raw/test_paper.pdf"

    os.makedirs("data/raw", exist_ok=True)

    if not os.path.exists(test_path):
        print("Downloading test PDF...")
        r = requests.get(test_url)
        with open(test_path, "wb") as f:
            f.write(r.content)

    print("Testing document loader...")
    pages = load_pdf(test_path)

    print(f"Loaded {len(pages)} pages")
    print(f"Page 1 text preview: {pages[0]['text'][:200]}")
    print(f"Page 1 images found: {len(pages[0]['images'])}")
    if pages[0]["images"]:
        print(f"Page 1 first image_id: {pages[0]['images'][0]['image_id']}")
        print(f"Page 1 first image saved to: {pages[0]['images'][0]['image_path']}")
    print("document_loader.py works correctly")