import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain.schema import Document
from RAG.config_loader import config_data, system_prompts
import argparse

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"



def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    return B_INST + system_prompt + instruction + E_INST

# Load sentence transformer
def load_sentence_transformer(model_name):
    return SentenceTransformerEmbeddings(model_name=model_name)

# Load vector store (Chroma)
def load_chroma():
    embedding_function = load_sentence_transformer(config_data["VECTOR_DB_SENTENCE_EMBEDDING_MODEL"])
    return Chroma(persist_directory=config_data["VECTOR_DB_PATH"], embedding_function=embedding_function), embedding_function

# Load context dataframe from CSV
def load_context_dataframe():
    path = config_data["CSV_PATH"]
    df = pd.read_csv(path)
    df["node_context"] = df["content"].astype(str) + "\n\n" + df["sol"].astype(str)
    return df[["project_name", "node_context"]].rename(columns={"project_name": "node_name"})

logging.basicConfig(filename='debug_log.txt', level=logging.DEBUG, format='%(asctime)s - %(message)s')

def retrieve_context(question, vectorstore, embedding_function, context_df):
    # Lấy hits từ vectorstore, k=1 để lấy tài liệu phù hợp nhất
    hits = vectorstore.similarity_search_with_score(question, k=1)
    logging.debug("Hits: %s", hits)  # Ghi log vào file debug_log.txt
    
    max_per_node = int(config_data["CONTEXT_VOLUME"] / 5)  # Số lượng tối đa context cho mỗi node
    question_emb = embedding_function.embed_query(question)  # Tạo embedding cho câu hỏi
    full_context = ""

    for doc, score in hits:
        project_name = doc.metadata.get("project_name")
        
        # Kiểm tra nếu project_name có trong context
        if project_name not in context_df["node_name"].values:
            logging.debug("Không tìm thấy project_name '%s' trong context!", project_name)  # Ghi log
            continue

        node_context = context_df[context_df["node_name"] == project_name]["node_context"].values[0]
        chunks = node_context.split(". ")  # Chia context thành các đoạn nhỏ
        
        # Thêm tất cả các chunks vào full_context mà không lọc theo cosine similarity
        full_context += ". ".join(chunks[:max_per_node]) + ". "

    # Trả về toàn bộ context đã được ghép lại
    logging.debug("Full context: %s", full_context)  # Ghi log
    return full_context


def main():
    parser = argparse.ArgumentParser(description='LlamaRAG via Ollama API')
    parser.add_argument('question', type=str, help='Your input question')
    args = parser.parse_args()
    
    print("→ Đang load vector DB và embedding...")
    vectorstore, embedding_function = load_chroma()
    context_df = load_context_dataframe()

    print("→ Truy xuất ngữ cảnh từ vector DB...")
    context = retrieve_context(args.question, vectorstore, embedding_function, context_df)
    question = args.question

    print("→ Gọi Ollama model sinh câu trả lời...")
    prompt_text = get_prompt(f"Context:\n\n{context} \n\nQuestion: {question}", system_prompts["SMART_CONTRACT_ANALYSIS"])
    prompt_text = prompt_text.replace("{", "{{").replace("}", "}}")
   
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
    llm = OllamaLLM(model="llama2", temperature=0) 

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": args.question
    })

    print("\n✅ Kết quả:")
    print(response)
   

if __name__ == "__main__":
    main()
