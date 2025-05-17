import re
import time
import logging
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from RAG.config_loader import config_data

# === Load config ===
CSV_PATH = config_data["CSV_PATH"]
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
EMBEDDING_MODEL = config_data["VECTOR_DB_SENTENCE_EMBEDDING_MODEL"]
CHUNK_SIZE = config_data.get("VECTOR_DB_CHUNK_SIZE", 650)
CHUNK_OVERLAP = config_data.get("VECTOR_DB_CHUNK_OVERLAP", 200)
BATCH_SIZE = config_data.get("VECTOR_DB_BATCH_SIZE", 200)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_data():
    logging.info(f"Đang load file CSV từ: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    logging.info(f"Đã load file CSV với {len(df)} dòng.")
    df["node_context"] = df["report"].astype(str)
    return df


def create_vectordb():
    start_time = time.time()
    df = load_data()

    logging.info("Đang chia nhỏ context thành các chunk...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    total_chunks = 0

    for idx, row in df.iterrows():
        content_str = str(row["content"])

        # Trích xuất category
        category_match = re.search(r'"category"\s*:\s*"([^"]+)"', content_str)
        category = category_match.group(1) if category_match else "Unknown"

        # Trích xuất function
        function_match = re.search(r'"function"\s*:\s*"([^"]+)"', content_str)
        function = function_match.group(1) if function_match else "Unknown"

        metadata = {
            "category": category,
            "function": function
        }

        chunks = text_splitter.split_text(row["node_context"])

        logging.info(f"Dòng {idx}: category={category}, function={function}, số chunk={len(chunks)}")

        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=metadata))
            total_chunks += 1

    logging.info(f"Tổng số chunk tạo được: {total_chunks}")

    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(embedding_function=embedding_model, persist_directory=VECTOR_DB_PATH)

    logging.info("Đang thêm tài liệu vào VectorDB...")
    for batch_idx, batch in enumerate(batches):
        vectorstore.add_documents(documents=batch)
        logging.info(f"Đã thêm batch {batch_idx + 1}/{len(batches)} với {len(batch)} documents.")

    vectorstore.persist()
    duration = round((time.time() - start_time) / 60, 2)
    logging.info(f"VectorDB đã được tạo xong trong {duration} phút tại: {VECTOR_DB_PATH}")


if __name__ == "__main__":
    create_vectordb()
