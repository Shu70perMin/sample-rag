import pdfplumber
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import requests


def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"
    return all_text


def split_into_chunks(text):
    chunks = []
    pattern = r"(?:^|\n)(SLD-\d+:|Issue\s+\d+:|Finding\s+\d+:)"
    parts = re.split(pattern, text)

    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            if i + 1 >= len(parts):
                print(f"[WARNING] Bỏ qua {title} vì không có nội dung đi kèm.")
                continue
            raw_content = parts[i + 1].strip()

            # Bỏ qua các mục như Summary, Disclaimer, Overview, hoặc mục lục ngắn
            if any(kw in raw_content for kw in ["Disclaimer", "Summary", "Overview", "Audit Scope", "Project Detail"]):
                continue
            if len(raw_content.split()) <= 20:
                continue

            labeled_content = f"Title: {title}\n"

            desc = re.search(r"(?i)Description\s*[:\-]?\s*\n(.+?)(Recommendation|Client Response|Category|Code Reference|Status|$)", raw_content, re.DOTALL)
            if desc:
                labeled_content += f"\nDescription:\n{desc.group(1).strip()}\n"

            mitigation = re.search(r"(?i)(Recommendation|Mitigation)\s*[:\-]?\s*\n?(.+?)(Client Response|Category|$)", raw_content, re.DOTALL)
            if mitigation:
                labeled_content += f"\nRecommendation:\n{mitigation.group(2).strip()}\n"

            code_ref = re.findall(r"^\d+:.*$", raw_content, re.MULTILINE)
            if code_ref:
                labeled_content += f"\nCode:\n" + "\n".join(code_ref) + "\n"

            category_match = re.search(r"(?i)Category\s+([^\n]+)", raw_content)
            if category_match:
                labeled_content += f"\nCategory: {category_match.group(1).strip()}\n"

            severity_match = re.search(r"(?i)Severity\s+([^\n]+)", raw_content)
            if severity_match:
                labeled_content += f"\nSeverity: {severity_match.group(1).strip()}\n"

            status_match = re.search(r"(?i)Status\s+([^\n]+)", raw_content)
            if status_match:
                labeled_content += f"\nStatus: {status_match.group(1).strip()}\n"

            chunks.append(labeled_content.strip())

    return chunks


def extract_structured_info(chunk):
    info = {
        "title": None,
        "description": None,
        "severity": None,
        "condition": None,
        "mitigation": None,
        "category": None,
        "code_reference": None,
        "status": None
    }

    lines = chunk.strip().split("\n")
    info["title"] = lines[0].replace("Title:", "").strip()

    severity_keywords = ["Critical", "Medium", "Informational", "High", "Low", "Major", "Minor"]
    for keyword in severity_keywords:
        if keyword.lower() in chunk.lower():
            info["severity"] = keyword
            break

    desc_match = re.search(r"(?i)Description\s*[:\-]?\s*\n(.+?)(Recommendation|Client Response|Category|Code Reference|Status|$)", chunk, re.DOTALL)
    if desc_match:
        info["description"] = desc_match.group(1).strip()

    code_ref_match = re.findall(r"^\d+:.*$", chunk, re.MULTILINE)
    if code_ref_match:
        info["code_reference"] = "\n".join(code_ref_match)

    mit_match = re.search(r"(?i)(Recommendation|Mitigation)\s*[:\-]?\s*\n?(.+?)(Client Response|Category|$)", chunk, re.DOTALL)
    if mit_match:
        info["mitigation"] = mit_match.group(2).strip()
    
    status_keywords = ["Fixed", "Acknowledged", "Pending", "Not Fixed", "In Progress"]
    for keyword in status_keywords:
        if keyword.lower() in chunk.lower():
            info["status"] = keyword
            break

    # Sử dụng LLM để chiết xuất condition và category
    try:
        prompt = f"""
        You are a smart contract security auditor. Read the following description and code snippet and extract:
        - The condition under which this bug can occur (as short sentence)
        - The vulnerability category (one of: Logical, Code Style, Language Specific, Access Control, Reentrancy, Oracle Manipulation, Gas Optimization, Business Logic, Arithmetic, Time-based)

        Reply in JSON with fields: condition and category.

        ### DESCRIPTION:
        {info['description']}

        ### CODE:
        {info['code_reference']}
        """

        res = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json",
                "HTTP-Referer": "your-site.com",
                "X-Title": "AuditExtractor"
            },
            json={
                "model": "deepseek/deepseek-r1-distill-llama-70b:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )

        if res.status_code == 200:
            data = res.json()
            content = data["choices"][0]["message"]["content"]
            match = re.search(r'"condition"\s*:\s*"(.*?)"\s*,\s*"category"\s*:\s*"(.*?)"', content, re.DOTALL)
            if match:
                info["condition"] = match.group(1).strip()
                info["category"] = match.group(2).strip()

    except Exception as e:
        print(f"[LLM Warning] Failed to retrieve condition/category: {e}")

    return info

    return info


def embed_chunks_from_text(chunks, model):
    return model.encode(chunks)


def save_to_faiss(embeddings, chunks, faiss_index_path="audit_index.index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, faiss_index_path)

    with open("audit_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")


def search_faiss(query, model, faiss_index_path="audit_index.index", top_k=3):
    query_embedding = model.encode([query])
    index = faiss.read_index(faiss_index_path)
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)

    with open("audit_chunks.txt", "r", encoding="utf-8") as f:
        all_chunks = f.read().split("\n---\n")

    return [all_chunks[i] for i in I[0]]


def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)

    chunks = split_into_chunks(text)

    # Lưu các chunk vào file TXT
    with open("audit_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")

    structured_data = [extract_structured_info(chunk) for chunk in chunks]

    # Lưu thông tin cấu trúc vào JSON
    with open("audit_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Đã lưu file audit_structured.json với {len(structured_data)} đoạn.")


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pdf_path = "Secure3_Shield_SSVault_security_audit_report.pdf"
    process_pdf(pdf_path)

    # 2. Đọc lại chunks từ file văn bản
    with open("audit_chunks.txt", "r", encoding="utf-8") as f:
        all_chunks = f.read().split("\n---\n")

    # 3. Encode & Save FAISS
    embeddings = embed_chunks_from_text(all_chunks, model)
    save_to_faiss(embeddings, all_chunks)

    # 4. Truy vấn thử
    query = "What reentrancy vulnerabilities are present in the deposit function?"
    results = search_faiss(query, model)
    for i, chunk in enumerate(results):
        print(f"\n=== Result {i+1} ===\n{chunk}")
