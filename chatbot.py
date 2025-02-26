# chatbot.py
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from call_llm import call_llm
import torch

# Parâmetros dos arquivos pré-processados
DATA_FILE = "preprocessed_data.pkl"
FAISS_FILE = "faiss_index.index"
DOC_FOLDER = "Docs"

# Inicializa o modelo de embeddings (usado para gerar consulta)
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_preprocessed_data(data_file=DATA_FILE, faiss_file=FAISS_FILE):
    """
    Carrega os dados pré-processados. Se não existirem, chama o pre_process.
    Retorna:
       - document_index, chunk_metadata, faiss_index
    """
    if not os.path.exists(data_file) or not os.path.exists(faiss_file):
        # Se os dados não existem, importe e chame o módulo de pré-processamento.
        print("Dados pré-processados não encontrados. Processando documentos...")
        from pre_preocessing import preprocess_documents
        document_index, chunk_metadata, index = preprocess_documents(doc_folder=DOC_FOLDER,
                                                                       data_output=data_file,
                                                                       faiss_output=faiss_file)
    else:
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        document_index = data["document_index"]
        chunk_metadata = data["chunk_metadata"]
        index = faiss.read_index(faiss_file)
    return document_index, chunk_metadata, index

def search_relevant_chunks(query, chunk_metadata, faiss_index, top_k=3):
    """
    Busca os chunks relevantes para a consulta usando o índice FAISS.
    Retorna uma string com os metadados dos chunks encontrados (pode incluir o chunk e/ou o resumo acumulado).
    """
    query_emb = model.encode(query, convert_to_tensor=False).astype("float32")
    query_emb = np.expand_dims(query_emb, axis=0)
    # Normaliza o embedding de consulta
    faiss.normalize_L2(query_emb)
    
    distances, indices = faiss_index.search(query_emb, top_k)
    # Recupera os metadados
    results = []
    for idx in indices[0]:
        if idx < len(chunk_metadata):
            meta = chunk_metadata[idx]
            # Você pode escolher mostrar o chunk original e/ou o resumo acumulado
            results.append(f"Documento: {meta['doc']}\nChunk: {meta['chunk']}\nResumo Acumulado: {meta['accum_summary']}")
    return "\n\n".join(results)

def melhorar_prompt(prompt):
    """
    Chama a LLM para sugerir melhorias e palavras-chave no prompt.
    """
    prompt_enhancer = f"""
Você é um especialista em otimização de prompts.
Analise o seguinte prompt e sugira palavras-chave ou pequenas melhorias para enriquecer o contexto:

{prompt}

Forneça apenas as palavras-chave e/ou pequenas melhorias.
"""
    sugestao = call_llm(prompt_enhancer)
    return prompt + "\nPalavras-chave sugeridas: " + sugestao

def processar_mensagem(usuario_msg, historico_conversa, document_index, chunk_metadata, faiss_index):
    """
    Processa a mensagem do usuário:
      - Busca os chunks relevantes (com metadados) usando o índice vetorial;
      - Monta um prompt incluindo o histórico, a mensagem e os metadados recuperados;
      - Usa o melhorador de prompt;
      - Chama a LLM para gerar a resposta final.
    """
    relevant_chunks = search_relevant_chunks(usuario_msg, chunk_metadata, faiss_index)
    
    prompt_base = f"""
Histórico da conversa:
{historico_conversa}

Mensagem do usuário:
{usuario_msg}

Informações recuperadas dos documentos:
{relevant_chunks}

Com base nessas informações, elabore uma resposta detalhada e coerente.
"""
    prompt_final = melhorar_prompt(prompt_base)
    resposta = call_llm(prompt_final)
    return resposta

def main():
    # Carrega os dados pré-processados
    document_index, chunk_metadata, faiss_index = load_preprocessed_data()
    historico_conversa = ""
    print("\n--- Chatbot Iniciado --- (digite 'sair' para encerrar)")
    
    while True:
        usuario_msg = input("\nVocê: ")
        if usuario_msg.lower() in ["sair", "exit", "quit"]:
            break
        resposta = processar_mensagem(usuario_msg, historico_conversa, document_index, chunk_metadata, faiss_index)
        print("\nChatbot:", resposta)
        historico_conversa += f"\nUsuário: {usuario_msg}\nChatbot: {resposta}\n"

if __name__ == "__main__":
    main()
