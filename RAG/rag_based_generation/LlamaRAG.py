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
import json
import sys

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

def analyze_code_and_map_errors(code):
    # Gọi LLM để phân tích mã nguồn Solidity và phát hiện lỗi
    llm = OllamaLLM(model="llama2", temperature=0)  # Dùng LLM cho phân tích mã nguồn
    
    # Tạo prompt cho LLM để phân tích mã nguồn Solidity
    prompt_code_analysis = get_prompt(f"Please analyze the following Solidity code and detect any vulnerabilities:\n\n{code}", system_prompts["SMART_CONTRACT_ANALYSIS"])
    prompt_code_analysis = prompt_code_analysis.replace("{", "{{").replace("}", "}}")
    #print(prompt_code_analysis)
    # LLM sẽ trả về kết quả phân tích mã nguồn
    analysis_result = llm.invoke(prompt_code_analysis)
    
    # Tải file relation_map.json chứa thông tin về các lỗ hổng bảo mật
    with open('RAG/rag_based_generation/relation_map.json', 'r') as f:
        relation_map = json.load(f)

    # Ánh xạ các lỗi bảo mật đã liệt kê trong SMART_CONTRACT_ANALYSIS vào các lỗi trong relation_map
    vulnerabilities = []
    vulnerabilities_info = []

    if "gasless" in analysis_result.lower():
        vulnerabilities.append("Gasless")
        vulnerabilities_info.append(relation_map["Gasless"])  # Thêm thông tin chi tiết từ relation_map
    if "unchecked call" in analysis_result.lower():
        vulnerabilities.append("UncheckedExternalCall")
        vulnerabilities_info.append(relation_map["Unchecked External Call"])
    if "reentrancy" in analysis_result.lower():
        vulnerabilities.append("Reentrancy")
        vulnerabilities_info.append(relation_map["Reentrancy"])
    if "timestamp dependency" in analysis_result.lower():
        vulnerabilities.append("TimestampDependency")
        vulnerabilities_info.append(relation_map["Timestamp Dependency"])
    if "block number dependency" in analysis_result.lower():
        vulnerabilities.append("BlockNumberDependency")
        vulnerabilities_info.append(relation_map["Block Number Dependency"])
    if "dangerous delegatecall" in analysis_result.lower():
        vulnerabilities.append("DangerousDelegatecall")
        vulnerabilities_info.append(relation_map["DangerousDelegatecall"])
    if "freezing ether" in analysis_result.lower():
        vulnerabilities.append("FreezingEther")
        vulnerabilities_info.append(relation_map["Freezing Ether"])
    if "integer overflow" in analysis_result.lower():
        vulnerabilities.append("Integer Overflow")
        vulnerabilities_info.append(relation_map["Integer Overflow"])
    if "integer underflow" in analysis_result.lower():
        vulnerabilities.append("Integer Underflow")
        vulnerabilities_info.append(relation_map["Integer Underflow"])
    if "unexpected ether" in analysis_result.lower():
        vulnerabilities.append("UnexpectedEther")
        vulnerabilities_info.append(relation_map["Unexpected Ether"])
    if "authorization through tx.origin" in analysis_result.lower():
        vulnerabilities.append("TxOriginAuth")
        vulnerabilities_info.append(relation_map["Authorization through tx.origin"])
    if "false assert" in analysis_result.lower():
        vulnerabilities.append("FalseAssert")
        vulnerabilities_info.append(relation_map["False Assert"])
    if "false suicide" in analysis_result.lower():
        vulnerabilities.append("FalseSuicide")
        vulnerabilities_info.append(relation_map["False Suicide"])

    vulnerabilities_info_json = {
        "vulnerabilities": vulnerabilities_info
    }

    return vulnerabilities, analysis_result, vulnerabilities_info_json


def main():
    #parser = argparse.ArgumentParser(description='LlamaRAG via Ollama API')
    #parser.add_argument('solidity_code', type=str, help='Solidity code to be analyzed')
    #args = parser.parse_args()
    
    solidity_code = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VulnerableContract {
    mapping(address => uint) public balances;
    bool private lock = false;

    // Lỗi Reentrancy ở đây
    function withdraw(uint _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");

        // Lỗi: Chuyển tiền trước khi cập nhật trạng thái
        payable(msg.sender).transfer(_amount);

        // Reentrancy vulnerability: Một hợp đồng khác có thể gọi lại hàm này
        balances[msg.sender] -= _amount;
    }

    // Hàm gửi tiền vào hợp đồng
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // Lỗi Integer Overflow trong hàm này
    function addToBalance(uint _amount) public {
        // Lỗi: Không kiểm tra tràn số
        balances[msg.sender] += _amount;
    }
}

"""

    print("→ Đang phân tích mã nguồn Solidity...")
    
    # Phân tích mã Solidity và ánh xạ các lỗi
    vulnerabilities, analysis_result, vulnerabilities_info_json = analyze_code_and_map_errors(solidity_code)
    
    print("→ Tạo câu truy vấn từ các lỗ hổng bảo mật đã phát hiện...")
    print(f"Vulnerabilities detected: {vulnerabilities}")
    sys.exit()
    # Truy xuất ngữ cảnh từ vector DB 
    vectorstore, embedding_function = load_chroma()
    context_df = load_context_dataframe()

    print("→ Truy xuất ngữ cảnh từ vector DB...")
    context = retrieve_context(f"Analyze these vulnerabilities: {', '.join(vulnerabilities)}", vectorstore, embedding_function, context_df)
    
    print("→ Gọi Ollama model để sinh câu trả lời...")
    prompt_text = get_prompt(f"Context:\n\n{context} \n\nVulnerabilities: {', '.join(vulnerabilities)}", system_prompts["SMART_CONTRACT_ANALYSIS"])
    prompt_text = prompt_text.replace("{", "{{").replace("}", "}}")
   
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "vulnerabilities"])
    llm = OllamaLLM(model="llama2", temperature=0) 

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "vulnerabilities": ", ".join(vulnerabilities)
    })

    print("\n✅ Kết quả:")
    print(response)
   

if __name__ == "__main__":
    main()
