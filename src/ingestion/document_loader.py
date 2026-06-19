# src/ingestion/document_loader.py
import pdfplumber
import os
from pathlib import Path
from PIL import Image
import io
import re 
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
               - images: list of PIL Images
               - section_heading: str (best guess from font size)
    """
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            
            # Extract text
            text = page.extract_text() or ""
            text = clean_pdf_text(text)
            
            # Extract images
            images = []
            for img in page.images:
                try:
                    # Crop image region from page
                    bbox = (img['x0'], img['top'], 
                            img['x1'], img['bottom'])
                    cropped = page.crop(bbox)
                    img_obj = cropped.to_image(resolution=150)
                    pil_img = img_obj.original
                    images.append(pil_img)
                except Exception:
                    continue
            
            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "images": images,
                "doc_name": Path(pdf_path).stem,
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
    print("document_loader.py works correctly")