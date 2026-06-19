# src/ingestion/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import uuid

# Load BGE tokenizer to count tokens accurately
# (character-based splitting is imprecise — token-based is correct)
TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

def count_tokens(text):

    return len(TOKENIZER.encode(text, 
                                add_special_tokens=False,
                                truncation=True,
                                max_length=4096))

def truncate_to_token_limit(text, max_tokens=500):
    """
    For CHILDREN only — these go through BGE (512 token limit).
    Uses tokenizer encode/decode — only call this on child chunks.
    """
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    # Truncate at a sentence boundary to avoid mid-sentence cuts
    truncated = TOKENIZER.decode(tokens[:max_tokens], 
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
    return truncated


def truncate_parent_by_chars(text, max_chars=2000):
    """
    For PARENTS only — these go to the LLM generator, not BGE.
    Simple character truncation — preserves original text exactly.
    No tokenizer decode mangling.
    """
    if len(text) <= max_chars:
        return text
    # Cut at last sentence boundary before limit
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.8:  # only cut at sentence if not too far back
        return truncated[:last_period + 1]
    return truncated

def split_into_parent_child(pages, 
                             child_size=256, 
                             parent_size=512, 
                             overlap=32):
    """
    Parent-child chunking strategy.
    
    Children (256 tokens) → used for retrieval (precise matching)
    Parents (1024 tokens) → used for generation (full context)
    
    At retrieval time: find best child → return its parent to LLM
    This gives precision in retrieval + context in generation.
    """
    parents = []
    children = []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size * 5,   # *4 = approx chars from tokens
        chunk_overlap=overlap * 5,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size * 2,
        chunk_overlap=overlap * 2,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for page in pages:
        text = page["text"].strip()
        if not text or len(text) < 50:  # skip near-empty pages
            continue

        # Split page into parent chunks
        parent_texts = parent_splitter.split_text(text)

        for parent_text in parent_texts:
            parent_id = str(uuid.uuid4())

            # Build context header — prepended to every chunk before embedding
            # This is crucial: without it chunks lose document context
            context_header = (
                f"Document: {page['doc_name']} | "
                f"Page: {page['page_num']}\n\n"
            )

            enriched_parent = truncate_parent_by_chars(context_header + parent_text)
            parents.append({
                "id": parent_id,
                "text": enriched_parent,
                "raw_text": parent_text,
                "doc_name": page["doc_name"],
                "page_num": page["page_num"],
                "token_count": count_tokens(enriched_parent)
            })

            # Split parent into children
            child_texts = child_splitter.split_text(parent_text)

            for child_text in child_texts:
                child_id = str(uuid.uuid4())
                enriched_child = truncate_to_token_limit(context_header + child_text)

                children.append({
                    "id": child_id,
                    "parent_id": parent_id,  # pointer back to parent
                    "text": enriched_child,
                    "raw_text": child_text,
                    "doc_name": page["doc_name"],
                    "page_num": page["page_num"],
                    "token_count": count_tokens(enriched_child)
                })

    print(f"Created {len(parents)} parent chunks, {len(children)} child chunks")
    return parents, children


def generate_hypothetical_questions(chunk_text, llm_fn, n_questions=2):
    """
    HyDE: generate hypothetical questions this chunk would answer.
    
    Why: embedding space gap between questions and answers.
    User asks: "what is the recall@5 score?"
    Chunk says: "we achieved recall@5 of 0.84"
    These aren't close in embedding space — but the hypothetical
    question "what recall@5 score did the system achieve?" IS close
    to the user's actual question.
    
    llm_fn: a callable that takes a prompt and returns a string
            (pass your Phi-3 inference function here)
    """
    prompt = f"""Given this text, write {n_questions} questions that this text directly answers.
Return only the questions, one per line, no numbering.

Text: {chunk_text[:500]}

Questions:"""
    
    try:
        response = llm_fn(prompt)
        questions = [q.strip() for q in response.strip().split('\n') 
                    if q.strip() and len(q.strip()) > 10]
        return questions[:n_questions]
    except Exception as e:
        print(f"HyDE generation failed: {e}")
        return []


# Test
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.ingestion.document_loader import load_pdf

    print("Testing chunker...")
    pages = load_pdf("data/raw/test_paper.pdf")
    parents, children = split_into_parent_child(pages)

    print(f"\nSample parent chunk:")
    print(f"  ID: {parents[0]['id']}")
    print(f"  Tokens: {parents[0]['token_count']}")
    print(f"  Preview: {parents[0]['text'][:200]}")

    print(f"\nSample child chunk:")
    print(f"  ID: {children[0]['id']}")
    print(f"  Parent ID: {children[0]['parent_id']}")
    print(f"  Tokens: {children[0]['token_count']}")
    print(f"  Preview: {children[0]['text'][:200]}")

    # Verify parent-child link
    parent_ids = {p['id'] for p in parents}
    orphaned = [c for c in children if c['parent_id'] not in parent_ids]
    print(f"\nOrphaned children: {len(orphaned)} (should be 0)")
    print("chunker.py works correctly")
    # Add this to your test block to see the real distribution
    avg_children = len(children) / len(parents)
    print(f"Average children per parent: {avg_children:.1f}")

    # Show a parent that has multiple children
    for p in parents:
        p_children = [c for c in children if c['parent_id'] == p['id']]
        if len(p_children) > 2:
            print(f"\nParent with {len(p_children)} children:")
            print(f"  Parent tokens: {p['token_count']}")
            print(f"  Child tokens: {[c['token_count'] for c in p_children]}")
            print(f"  Parent preview: {p['text'][:100]}")
            print(f"  Child 1 preview: {p_children[0]['text'][:80]}")
            print(f"  Child 2 preview: {p_children[1]['text'][:80]}")
            break
    