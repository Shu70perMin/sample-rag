import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from RAG.config_loader import config_data, system_prompts

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    return B_INST + system_prompt + instruction + E_INST

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False, method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, revision=branch_name, cache_dir=cache_dir)
    else:
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, revision=branch_name, cache_dir=cache_dir, legacy=False)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, revision=branch_name, cache_dir=cache_dir)

    streamer = TextStreamer(tokenizer) if stream else None
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=max_new_tokens, do_sample=True, streamer=streamer)

    return HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": temperature, "top_p": top_p})

def load_sentence_transformer(model_name):
    return SentenceTransformerEmbeddings(model_name=model_name)

def load_chroma():
    embedding_function = load_sentence_transformer(config_data["VECTOR_DB_SENTENCE_EMBEDDING_MODEL"])
    return Chroma(persist_directory=config_data["VECTOR_DB_PATH"], embedding_function=embedding_function), embedding_function

def load_context_dataframe():
    path = config_data["NODE_CONTEXT_PATH"]
    df = pd.read_csv(path)
    df["node_context"] = df["content"].astype(str) + "\n\n" + df["sol"].astype(str)
    return df[["project_name", "node_context"]].rename(columns={"project_name": "node_name"})

def retrieve_context(question, vectorstore, embedding_function, context_df):
    hits = vectorstore.similarity_search_with_score(question, k=5)
    max_per_node = int(config_data["CONTEXT_VOLUME"] / 5)
    question_emb = embedding_function.embed_query(question)
    full_context = ""

    for node in hits:
        node_name = node[0].page_content
        context = context_df[context_df.node_name == node_name].node_context.values[0]
        chunks = context.split(". ")
        chunk_embs = embedding_function.embed_documents(chunks)
        sims = [cosine_similarity(np.array(question_emb).reshape(1, -1), np.array(ce).reshape(1, -1)) for ce in chunk_embs]
        sims = sorted([(s[0][0], i) for i, s in enumerate(sims)], reverse=True)
        threshold = np.percentile([s[0] for s in sims], config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
        valid_idxs = [i for s, i in sims if s > threshold and s > config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]]
        if len(valid_idxs) > max_per_node:
            valid_idxs = valid_idxs[:max_per_node]
        full_context += ". ".join([chunks[i] for i in valid_idxs]) + ". "

    return full_context

def run_llama_only_pipeline(question):
    print("→ Đang load vector DB và embedding...")
    vectorstore, embedding_function = load_chroma()
    context_df = load_context_dataframe()

    print("→ Truy xuất ngữ cảnh từ vector DB...")
    context = retrieve_context(question, vectorstore, embedding_function, context_df)

    print("→ Gọi LLaMA model sinh câu trả lời...")
    prompt_text = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompts["SMART_CONTRACT_ANALYSIS"])
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
    llm = llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"], config_data["LLM_CACHE_DIR"], temperature=0, stream=True)
    chain = LLMChain(prompt=prompt, llm=llm)
    response = chain.run(context=context, question=question)
    print("\n✅ Kết quả:")
    print(response)
