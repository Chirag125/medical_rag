# Run this once from E:\medicalRAG
# Save as fix_init.py and run it

import os

folders = [
    "src",
    "src/ingestion", 
    "src/retrieval",
    "src/generation",
    "src/evaluation"
]

for folder in folders:
    init_path = os.path.join(folder, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write("")  # empty file, proper UTF-8
    print(f"Created {init_path}")