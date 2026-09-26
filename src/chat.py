from search import search_prompt


def main():
    # prepara a busca e a LLM
    responder = search_prompt()

    if not responder:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite sua pergunta.")
    print("Para sair, digite: sair")
    print("-" * 40)

    # fica perguntando ate o usuario digitar sair
    while True:
        pergunta = input("PERGUNTA: ").strip()

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando o chat.")
            break

        if not pergunta:
            continue

        try:
            resposta = responder(pergunta)
            print(f"RESPOSTA: {resposta}")
        except Exception as e:
            print(f"Erro ao responder: {e}")

        print("-" * 40)


if __name__ == "__main__":
    main()
