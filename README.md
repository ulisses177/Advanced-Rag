# 📚 Chatbot com Memória Vetorial

Um chatbot inteligente que utiliza embeddings e busca vetorial para responder perguntas com base em documentos fornecidos.

## 🔍 Estrutura do Projeto

- `chatbot.py`: Interface principal do chatbot
- `call_llm.py`: Gerenciador de chamadas à LLM (Ollama)
- `pre_processing.py`: Processamento e indexação de documentos
- `Docs/`: Pasta para armazenar documentos (PDFs, TXTs, MDs)

## 🛠️ Requisitos

- Python 3.8+
- Ollama instalado e rodando localmente
- Modelo deepseek-r1 baixado no Ollama

## ⚙️ Instalação

### Usando Conda

``bash
Criar ambiente
conda create -n chatbot-env python=3.8
conda activate chatbot-env
Instalar dependências
pip install -r requirements.txt``

### Usando venv

```bash
Criar ambiente
python -m venv venv
Ativar ambiente (Windows)
.\venv\Scripts\activate
Ativar ambiente (Linux/Mac)
source venv/bin/activate
Instalar dependências
pip install -r requirements.txt```



## 🚀 Como Usar

1. Coloque seus documentos na pasta `Docs/`
2. Certifique-se que o Ollama está rodando com o modelo deepseek-r1
3. Execute o chatbot:


```bash
python chatbot.py```  


O sistema irá:
1. Processar os documentos automaticamente na primeira execução
2. Iniciar uma interface de chat no terminal
3. Responder perguntas com base no conteúdo dos documentos

## 📝 Notas

- Suporta documentos em PDF, TXT e MD
- Utiliza embeddings para busca semântica
- Mantém histórico de conversa para contexto
