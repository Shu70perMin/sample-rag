import pandas as pd
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from RAG.config_loader import config_data

# === Load config ===
CSV_PATH = config_data["CSV_PATH"]
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
EMBEDDING_MODEL = config_data["VECTOR_DB_SENTENCE_EMBEDDING_MODEL"]
CHUNK_SIZE = config_data.get("VECTOR_DB_CHUNK_SIZE", 650)
CHUNK_OVERLAP = config_data.get("VECTOR_DB_CHUNK_OVERLAP", 200)
BATCH_SIZE = config_data.get("VECTOR_DB_BATCH_SIZE", 200)


def load_data():
    print("Đang load file CSV từ:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df["node_context"] = df["content"].astype(str) + "\n\n" + df["sol"].astype(str)
    df = df[["project_name", "node_context"]].rename(columns={"project_name": "project_name"})
    return df


def create_vectordb():
    start_time = time.time()
    df = load_data()

    print(" Đang chia nhỏ context...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for _, row in df.iterrows():
        chunks = text_splitter.split_text(row["node_context"])
        for chunk in chunks:
           
            docs.append(Document(page_content=chunk, metadata={"project_name": row["project_name"]}))

    print(f" Tổng số chunks: {len(docs)}")
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(embedding_function=embedding_function, persist_directory=VECTOR_DB_PATH)

    print(" Đang thêm tài liệu vào VectorDB...")
    for batch in batches:
        vectorstore.add_documents(documents=batch)

    vectorstore.persist()
    duration = round((time.time() - start_time) / 60, 2)
    print(f" VectorDB đã được tạo trong {duration} phút tại: {VECTOR_DB_PATH}")


if __name__ == "__main__":
    create_vectordb()
