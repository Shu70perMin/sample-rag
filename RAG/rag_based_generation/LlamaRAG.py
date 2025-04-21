import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Đọc file CSV và gom nội dung từ cột 'content' và 'sol'
def read_and_process_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Gom nội dung từ 'content' và 'sol' thành một chuỗi duy nhất cho mỗi dòng
    df['combined_text'] = df['content'].fillna('') + " " + df['project_name'].fillna('')
    
    # Trả về danh sách các văn bản đã gom
    return df['combined_text'].tolist()


# Sinh embedding cho các chuỗi văn bản
def embed_chunks_from_text(chunks, model):
    return model.encode(chunks)


# Lưu vào FAISS
def save_to_faiss(embeddings, chunks, faiss_index_path="audit_index.index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, faiss_index_path)

    with open("audit_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")


# Truy vấn FAISS với câu hỏi và lưu kết quả vào file TXT
def search_faiss_and_save_to_txt(query, model, faiss_index_path="audit_index.index", top_k=1, output_file="search_results.txt"):
    query_embedding = model.encode([query])
    index = faiss.read_index(faiss_index_path)
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)

    # Đọc lại các chunks
    with open("audit_chunks.txt", "r", encoding="utf-8") as f:
        all_chunks = f.read().split("\n---\n")

    # Lưu kết quả vào file TXT
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        for i in I[0]:
            f.write(f"Result {i+1}:\n{all_chunks[i]}\n\n{'-'*50}\n")

    print(f"[INFO] Kết quả đã được lưu vào {output_file}")


# Xử lý file CSV và lưu embeddings vào FAISS
def process_csv(csv_path, model):
    # Đọc dữ liệu từ CSV và gom nội dung
    combined_texts = read_and_process_csv(csv_path)

    # Sinh embedding cho toàn bộ văn bản đã gom
    embeddings = embed_chunks_from_text(combined_texts, model)

    # Lưu embeddings vào FAISS
    save_to_faiss(embeddings, combined_texts)

    print(f"[INFO] Đã lưu embeddings vào FAISS và chunks vào file audit_chunks.txt.")


if __name__ == "__main__":
    # Tải mô hình Sentence-BERT
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Đọc file CSV và xử lý
    csv_path = 'datasetbyshu.csv'  # Đảm bảo đường dẫn đúng
    process_csv(csv_path, model)

    # Truy vấn và lưu kết quả vào file TXT
    query = "Find vulnerabilities in the lucky token contract"
    search_faiss_and_save_to_txt(query, model, output_file="search_results.txt")