import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from kg_rag.config_loader import config_data, system_prompts

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + system_prompt + instruction + E_INST
    return prompt_template

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False, method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, revision=branch_name, cache_dir=cache_dir)
    elif method == 'method-2':
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, revision=branch_name, cache_dir=cache_dir, legacy=False)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, revision=branch_name, cache_dir=cache_dir)

    if not stream:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=max_new_tokens, do_sample=True)
    else:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=max_new_tokens, do_sample=True, streamer=streamer)

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": temperature, "top_p": top_p})
    return llm

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold):
    node_hits = vectorstore.similarity_search_with_score(question, k=5)
    max_number = int(context_volume / 5)
    question_embedding = embedding_function.embed_query(question)
    context_result = ""

    for node in node_hits:
        node_name = node[0].page_content
        node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        context_chunks = node_context.split(". ")
        chunk_embeddings = embedding_function.embed_documents(context_chunks)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(emb).reshape(1, -1)) for emb in chunk_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
        high_sim_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
        if len(high_sim_indices) > max_number:
            high_sim_indices = high_sim_indices[:max_number]
        high_context = [context_chunks[i] for i in high_sim_indices]
        context_result += ". ".join(high_context) + ". "

    return context_result

def run_llama_only_pipeline(question, vectorstore, node_context_df, embedding_model):
    print("→ Đang truy xuất ngữ cảnh từ vector database...")
    context = retrieve_context(
        question,
        vectorstore,
        embedding_model,
        node_context_df,
        config_data["CONTEXT_VOLUME"],
        config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"],
        config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]
    )

    print("→ Đang khởi tạo LLaMA và sinh câu trả lời...")
    prompt_text = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompts["DRUG_REPURPOSING"])
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
    llm = llama_model(
        config_data["LLAMA_MODEL_NAME"],
        config_data["LLAMA_MODEL_BRANCH"],
        config_data["LLM_CACHE_DIR"],
        temperature=0,
        stream=True
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    response = chain.run(context=context, question=question)
    print("\n✅ Kết quả:")
    print(response)
