from utils import get_paper_dir

def get_source(arXiv_id, num_versions):
    # Input: arXiv ID (đã được giao từ trước)
    # Output:  **A source archive**, gzipped tar containing the `.tex` files, often several and linked through `\input` or `\include`, 
    # the `.bib` bibliography files, style and class files, occasionally a pre-compiled `.bbl`, and the figure binaries
    print("Not implemented yet")

def save_source(arXiv_id, version, source):
    # Input: arXiv ID (đã được giao từ trước), version (int), source (bytes)
    # Output: None
    print("Not implemented yet")
    # paper_dir = get_paper_dir(arXiv_id)
    # paper_dir.mkdir(parents=True, exist_ok=True)
    # with open(paper_dir / version, "wb") as f:
    #     f.write(source)