from utils import get_paper_dir
import json

def get_entry_discovery(arXiv_id):
    # Input: arXiv ID (đã được giao từ trước)
    # Output: dictionary - arxiv_id (string, non-null), title (string), authors (list of strings), 
    # submission_date (ISO string), revised_dates (list of strings), venue (string, nullable), num_versions, categories, abstract
    print("Not implemented yet")

def save_metadata(arXiv_id, metadata):
    # Input: arXiv ID (đã được giao từ trước), metadata (dictionary)
    # Output: None
    print("Not implemented yet")
    paper_dir = get_paper_dir(arXiv_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    with open(paper_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)