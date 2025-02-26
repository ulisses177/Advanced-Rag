
import random
import requests
import json
import traceback


# ============================
# Função para chamar a LLM
# ============================
def call_llm(prompt):
    """
    Chama a API do Ollama para gerar uma resposta com base no prompt.
    Retorna o texto completo gerado pela LLM.
    """
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    DEBUG = True
    data = {
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": True
    }
    
    if DEBUG:
        print(f"\nDEBUG: Enviando prompt para a API. Tamanho do prompt: {len(prompt)} caracteres.")
    
    try:
        response = requests.post(OLLAMA_API_URL, json=data, stream=True, timeout=30)
        if response.status_code == 200:
            full_response = ""
            print("\nGerando resposta da LLM:", end=" ", flush=True)
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                    except Exception as json_err:
                        if DEBUG:
                            print(f"\nDEBUG: Erro ao decodificar JSON: {json_err}")
                        continue
                    if 'response' in json_response:
                        token = json_response['response']
                        full_response += token
                        print(token, end="", flush=True)
            print()  # nova linha
            return full_response.strip()
        else:
            print(f"Erro na API: Código {response.status_code}")
            if DEBUG:
                print(f"DEBUG: Resposta da API: {response.text}")
            return prompt
    except Exception as e:
        print(f"Erro na chamada à LLM: {str(e)}")
        if DEBUG:
            traceback.print_exc()
        return prompt