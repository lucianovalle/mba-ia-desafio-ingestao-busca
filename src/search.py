import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# carrega o .env
load_dotenv()

# prompt do desafio ({contexto} e {pergunta} sao preenchidos depois)
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(question=None):
    # valida as variaveis do .env
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Erro: GOOGLE_API_KEY nao esta no .env")
        return None

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Erro: DATABASE_URL nao esta no .env")
        return None

    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    if not collection_name:
        print("Erro: PG_VECTOR_COLLECTION_NAME nao esta no .env")
        return None

    model_name = os.getenv("GOOGLE_EMBEDDING_MODEL")
    if not model_name:
        print("Erro: GOOGLE_EMBEDDING_MODEL nao esta no .env")
        return None

    llm_model = os.getenv("GOOGLE_LLM_MODEL")
    if not llm_model:
        print("Erro: GOOGLE_LLM_MODEL nao esta no .env")
        return None

    # embeddings (mesmo modelo da ingestao)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )

    # conexao com o banco vetorial
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )

    # modelo que gera a resposta
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=api_key,
        temperature=0,
    )

    def responder(pergunta):
        # busca os 10 trechos mais parecidos
        resultados = vector_store.similarity_search_with_score(pergunta, k=10)

        if not resultados:
            return "Não tenho informações necessárias para responder sua pergunta."

        # junta os textos encontrados
        pedacos = []
        for documento, score in resultados:
            pedacos.append(documento.page_content)

        contexto = "\n\n".join(pedacos)

        # monta o prompt e chama a LLM
        prompt = PROMPT_TEMPLATE.format(
            contexto=contexto,
            pergunta=pergunta,
        )
        resposta = llm.invoke(prompt)
        return resposta.content

    # se ja veio pergunta, responde agora
    if question:
        return responder(question)

    # senao devolve a funcao para o chat.py
    return responder
