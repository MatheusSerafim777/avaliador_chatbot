from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()  

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modelo recomendado:
MODEL_EMBEDDING = "text-embedding-3-large"

DIMENSIONS = 1024

def get_embedding(text: str):
    text = str(text).strip()
    if not text:
        return None

    params = {
        "model": MODEL_EMBEDDING,
        "input": text
    }

    if DIMENSIONS is not None:
        params["dimensions"] = DIMENSIONS

    response = client.embeddings.create(**params)
    return response.data[0].embedding


def semantic_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    if emb1 is None or emb2 is None:
        return 0.0

    score = cosine_similarity([emb1], [emb2])[0][0]
    return float(score)


def lexical_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()


def classificar(score: float) -> str:
    if score >= 0.85:
        return "excelente"
    elif score >= 0.70:
        return "aceitável"
    elif score >= 0.50:
        return "fraco"
    return "ruim"


def chatbot_faq_score(faq_answer: str, chatbot_answer: str) -> dict:
    sem_score = semantic_similarity(faq_answer, chatbot_answer)
    lex_score = lexical_similarity(faq_answer, chatbot_answer)

    # Você pode ajustar estes pesos depois dos testes
    final_score = (0.8 * sem_score) + (0.2 * lex_score)

    return {
        "semantic_score": round(sem_score, 4),
        "lexical_score": round(lex_score, 4),
        "final_score": round(final_score, 4),
        "classificacao": classificar(final_score)
    }


# ===== CONFIGURE AQUI =====
ARQUIVO_ENTRADA = "comparacao_teste.csv"
ARQUIVO_SAIDA = "comparacao_respostas_avaliado.csv"
COLUNA_FAQ = "Resposta FAQ"
COLUNA_CHATBOT = "Resposta com RAG" #mude de acordo com o nome da coluna que voce esta utilizando
# ==========================


def avaliar_csv(caminho_entrada: str, caminho_saida: str):
    df = pd.read_csv(caminho_entrada)

    if COLUNA_FAQ not in df.columns:
        raise ValueError(f"Coluna '{COLUNA_FAQ}' não encontrada no CSV.")
    if COLUNA_CHATBOT not in df.columns:
        raise ValueError(f"Coluna '{COLUNA_CHATBOT}' não encontrada no CSV.")

    resultados = []

    for _, row in df.iterrows():
        faq = row.get(COLUNA_FAQ, "")
        chatbot = row.get(COLUNA_CHATBOT, "")

        faq = "" if pd.isna(faq) else str(faq)
        chatbot = "" if pd.isna(chatbot) else str(chatbot)

        if not faq.strip() or not chatbot.strip():
            resultados.append({
                "semantic_score": None,
                "lexical_score": None,
                "final_score": None,
                "classificacao": "sem_resposta"
            })
            continue

        resultados.append(chatbot_faq_score(faq, chatbot))

    resultados_df = pd.DataFrame(resultados)
    df_saida = pd.concat([df, resultados_df], axis=1)
    df_saida.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig",
        sep=';',
        decimal=','
    )

    print("Avaliação concluída.")
    print(f"Arquivo salvo em: {caminho_saida}")
    print("\nResumo:")
    print(df_saida["classificacao"].value_counts(dropna=False))
    print(
        "\nMédia final:",
        round(df_saida["final_score"].dropna().mean(), 4)
        if df_saida["final_score"].notna().any()
        else "sem dados"
    )


if __name__ == "__main__":
    avaliar_csv(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)