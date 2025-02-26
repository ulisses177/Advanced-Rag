# pre_preocessing.py
import os
import glob
import pickle
import numpy as np
import torch
import faiss
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from call_llm import call_llm

# Inicializa o modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

###############################
# Funções Auxiliares de Leitura e Processamento
###############################

def read_document(file_path):
    """
    Lê o conteúdo de um documento com base na sua extensão.
    Suporta: .pdf, .txt, .md e outros formatos de texto.
    """
    ext = os.path.splitext(file_path)[1].lower()
    content = ""
    
    if ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
                    else:
                        print(f"Aviso: Página {i} de {file_path} não retornou texto.")
        except Exception as e:
            print(f"Erro ao ler PDF {file_path}: {e}")
    elif ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Erro ao ler {file_path}: {e}")
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Erro ao ler {file_path} (formato desconhecido): {e}")
    # Opcional: debug
    print(f"Conteúdo extraído de {file_path} (primeiros 300 caracteres):\n{content[:300]}\n")
    return content

def split_into_chunks(text, chunk_size=500, overlap=20):
    """
    Divide um texto em chunks com tamanho máximo 'chunk_size' e sobreposição 'overlap'.
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def calcular_embedding(text):
    """
    Calcula o embedding de um texto usando SentenceTransformers.
    """
    return model.encode(text, convert_to_tensor=True)

def buscar_resumos_similares(chunk, resumos_parciais, top_k=2):
    """
    Dado um chunk e uma lista de resumos parciais (cada um com 'texto' e 'embedding'),
    retorna os top_k resumos que são mais similares ao chunk.
    """
    if not resumos_parciais:
        return ""
    chunk_emb = calcular_embedding(chunk)
    # Empilha os embeddings
    embeddings = [item['embedding'] for item in resumos_parciais]
    embeddings_tensor = torch.stack(embeddings)
    
    sims = util.cos_sim(chunk_emb, embeddings_tensor)[0]
    sims = sims.cpu().numpy()
    top_indices = sims.argsort()[-top_k:][::-1]
    similares = [resumos_parciais[i]['texto'] for i in top_indices]
    return "\n".join(similares)

###############################
# Processamento de Documento e Indexação
###############################

def process_document(file_path):
    """
    Processa um documento:
      - Lê o conteúdo;
      - Divide em chunks;
      - Para cada chunk, atualiza o resumo acumulado (usando LLM e buscando resumos anteriores similares).
    
    Retorna:
      - resumo_final: resumo final do documento.
      - chunks_info: lista de dicionários, cada um com:
            "chunk": o texto do chunk,
            "accum_summary": o resumo acumulado após processar esse chunk.
    """
    text = read_document(file_path)
    if not text:
        print(f"Nenhum conteúdo extraído de {file_path}.")
        return "", []
    
    chunks = split_into_chunks(text)
    resumo_acumulado = ""
    resumos_parciais = []  # Para busca de similaridade durante o processo
    chunks_info = []       # Para armazenar cada chunk e seu metadado acumulado
    
    for chunk in chunks:
        # Busca resumos anteriores similares
        resumos_similares = buscar_resumos_similares(chunk, resumos_parciais)
        
        prompt = f"""
Você é um assistente especializado em extrair as informações mais relevantes de um documento.
Resumo acumulado atual:
{resumo_acumulado}

Chunk atual:
{chunk}

Resumos anteriores similares:
{resumos_similares}

Com base nessas informações, atualize o resumo do documento, garantindo que nenhuma informação importante seja perdida.
"""
        # Chama a LLM para atualizar o resumo
        resumo_acumulado = call_llm(prompt)
        # Calcula embedding do resumo para futuras buscas
        resumo_emb = calcular_embedding(resumo_acumulado)
        resumos_parciais.append({
            "texto": resumo_acumulado,
            "embedding": resumo_emb
        })
        # Salva o metadado do chunk: o próprio chunk e o resumo acumulado até aqui.
        chunks_info.append({
            "chunk": chunk,
            "accum_summary": resumo_acumulado
        })
    return resumo_acumulado, chunks_info

def preprocess_documents(doc_folder="Docs",
                         data_output="preprocessed_data.pkl",
                         faiss_output="faiss_index.index"):
    """
    Processa todos os documentos na pasta 'doc_folder'.
    Cria um índice FAISS para os chunks (com seus embeddings) e salva os metadados.
    Salva:
       - Um arquivo pickle com {document_index, chunk_metadata}
       - Um índice FAISS salvo em 'faiss_output'
    Retorna:
       document_index, chunk_metadata, faiss_index
    """
    document_index = {}
    chunk_metadata = []  # Lista de dicionários: cada dicionário terá: doc, chunk, accum_summary
    file_patterns = ["*.pdf", "*.txt", "*.md"]
    file_paths = []
    for pattern in file_patterns:
        file_paths.extend(glob.glob(os.path.join(doc_folder, pattern)))
    
    for file_path in file_paths:
        print(f"\nProcessando documento: {file_path}")
        resumo, chunks_info = process_document(file_path)
        doc_name = os.path.basename(file_path)
        document_index[doc_name] = {
            "resumo": resumo,
            "chunks": chunks_info
        }
        # Para cada chunk, adiciona metadados para indexação
        for entry in chunks_info:
            meta = {
                "doc": doc_name,
                "chunk": entry["chunk"],
                "accum_summary": entry["accum_summary"]
            }
            chunk_metadata.append(meta)
    
    # Cria um vetor para cada chunk usando os embeddings do texto original do chunk.
    # (Você pode optar por usar o resumo acumulado se preferir.)
    embeddings = []
    for meta in chunk_metadata:
        emb = model.encode(meta["chunk"], convert_to_tensor=False)
        embeddings.append(emb)
    embeddings = np.array(embeddings).astype("float32")
    
    # Normaliza os vetores para usar similaridade por cosseno
    faiss.normalize_L2(embeddings)
    embedding_dim = embeddings.shape[1]
    
    # Cria um índice FAISS simples (IndexFlatIP para similaridade interna produto)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    # Salva os metadados e o document_index
    data = {
        "document_index": document_index,
        "chunk_metadata": chunk_metadata
    }
    with open(data_output, "wb") as f:
        pickle.dump(data, f)
    print(f"\nDados pré-processados salvos em {data_output}.")
    
    # Salva o índice FAISS
    faiss.write_index(index, faiss_output)
    print(f"Índice FAISS salvo em {faiss_output}.")
    
    return document_index, chunk_metadata, index

if __name__ == "__main__":
    preprocess_documents()
