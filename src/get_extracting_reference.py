from utils import get_paper_dir
import json

def get_extracting_references(arXiv_id):
    # Input: arXiv ID (đã được giao từ trước)
	# Output: dictionary - arxiv_id (string, non-null, the citing paper), ref_arxiv_id (string, non-null), ref_title (string), 
    # ref_authors (list of strings), ref_submission_date (string), ref_s2_id (string) 

    # Giả sử api trả về một json - dictionary là result: mapping tương đương với các giá trị trả về từ Semantic Scholar API: arxiv_id = input, ref_arxiv_id = ref\["externalIds"]\["ArXiv"], ref_title = result\["title"], 
    # ref_authors = result\["authors"], ref_submission_date = result\["publicationDate"], ref_s2_id = result\["paperId"])

    print("Not implemented yet")

def save_extracting_references(arXiv_id, references):
    # Input: arXiv ID (đã được giao từ trước), references (dictionary)
    # Output: None
    print("Not implemented yet")
    paper_dir = get_paper_dir(arXiv_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    with open(paper_dir / "references.json", "w") as f:
        json.dump(references, f)