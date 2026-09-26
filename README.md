# Ingestão e Busca Semântica (LangChain + pgVector)

Projeto do desafio de ingestão de PDF e busca semântica via CLI.
Usa **Gemini** para embeddings e para a LLM, com PostgreSQL + pgVector.

## Pré-requisitos

- Python 3.12+
- Docker e Docker Compose
- API Key do Google (Gemini)

## Como executar

### 1. Ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

```bash
cp .env.example .env
```

Preencha no `.env`:

- `GOOGLE_API_KEY` — sua chave do Gemini
- `GOOGLE_EMBEDDING_MODEL` — ex.: `models/gemini-embedding-2`
- `GOOGLE_LLM_MODEL` — ex.: `gemini-3.8-flash`
- `DATABASE_URL` — ex.: `postgresql+psycopg://postgres:postgres@localhost:5432/rag`
- `PG_VECTOR_COLLECTION_NAME` — ex.: `pdf_documents`
- `PDF_PATH` — ex.: `document.pdf`

### 3. Subir o banco

```bash
docker compose up -d
```

### 4. Ingerir o PDF

```bash
python src/ingest.py
```

### 5. Rodar o chat

```bash
python src/chat.py
```

Exemplo:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual o faturamento da Alfa Energia S.A.?
RESPOSTA: O faturamento da Alfa Energia S.A. é de R$ 722.875.391,46.

PERGUNTA: Em que ano foi fundada a Alfa IA Indústria?
RESPOSTA: A Alfa IA Indústria foi fundada em 2020.

```

```
Exemplos de perguntas fora do contexto:

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

```

Para sair do chat: `sair`

## Estrutura

- `src/ingest.py` — lê o PDF, gera chunks/embeddings e grava no pgVector
- `src/search.py` — busca (k=10), monta o prompt e chama a LLM
- `src/chat.py` — interface no terminal
- `document.pdf` — PDF usado na ingestão
- `docker-compose.yml` — PostgreSQL com pgVector
