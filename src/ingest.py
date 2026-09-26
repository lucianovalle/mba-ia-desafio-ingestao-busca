import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pega as variáveis do .env (GOOGLE_API_KEY, DATABASE_URL, ...)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def roda_ingestao():

    # recupera chave do arquivo env
    api_key = os.getenv("GOOGLE_API_KEY") 
    # validar se a API está no .env
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não está no .env")

    # recupera URL do banco de dados do arquivo env
    database_url = os.getenv("DATABASE_URL")
    # validar se a URL do banco de dados está no .env
    if not database_url:
        raise ValueError("DATABASE_URL não está no .env")

    # recupera nome da coleção do arquivo env
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    # validar se o nome da coleção está no .env
    if not collection_name:
        raise ValueError("PG_VECTOR_COLLECTION_NAME não está no .env")
    
    # recupera nome do modelo de embedding do arquivo env
    model_name = os.getenv("GOOGLE_EMBEDDING_MODEL")
    # validar se o nome do modelo de embedding está no .env
    if not model_name:
        raise ValueError("GOOGLE_EMBEDDING_MODEL não está no .env")
    
    # caminho do PDF se vier relativo, junta com a pasta do projeto
    pdf_path = Path(os.getenv("PDF_PATH"))
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    # validar se o arquivo PDF existe
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    # carregar o PDF
    print(f"[PDF] Carregando: {pdf_path}...")
    # armazena o PDF em uma lista de páginas 
    loader_pdf = PyPDFLoader(str(pdf_path))

    # o resultado é uma lista de Documentos
    # page_content é o texto da página
    # metadata é o metadado da página (título, autor, etc.)
    documents = loader_pdf.load() 

    # divide o documento em pedaços menores (chunks)
    print("[CHUNK] Dividindo o documento...")
    # classe que divide o documento em pedaços menores (chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # tamanho máximo de cada pedaço
        chunk_overlap=150,    # quanto do final de um pedaço entra no começo do próximo
    )
    # divide o documento em pedaços menores (chunks)
    chunks = text_splitter.split_documents(documents)
    print(f"[CHUNK] Total gerados: {len(chunks)}")

    # embeddings recomendados no README do desafio (Gemini)
    print(f"[EMBED] Usando o modelo: {model_name}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )

    # PGVector cria as tabelas langchain_pg_collection e langchain_pg_embedding
    # pre_delete_collection=True: se eu rodar de novo, apaga a collection antiga
    print("[DB] Conectando no Postgres e gravando...")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    # 10 em 10 para ver o progresso e não sobrecarregar a API
    batch_size = 10
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch = i // batch_size + 1

        # - salva texto + vetor no banco
        try:
            vector_store.add_documents(batch)
            print(f"  -> Batch {current_batch}/{total_batches} ok")
        except Exception as e:
            print(f"Erro no batch {current_batch}: {e}")
            raise

    print("[OK] Ingestão concluída!")

if __name__ == "__main__":
    roda_ingestao()
