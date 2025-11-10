import joblib
import re
import string

# --- Função de limpeza (igual à usada no treino) ---
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(f"[{string.punctuation}]", "", texto)
    texto = re.sub(r"\d+", "", texto)
    texto = texto.strip()
    return texto

# --- Carregar modelo e vetorizador treinados ---
try:
    model = joblib.load("modelo_svm.pkl")
    vectorizer = joblib.load("vectorizer_tfidf.pkl")
    print("✅ Modelo e vetorizador carregados com sucesso.")
except Exception as e:
    print(f"❌ Erro ao carregar os arquivos do modelo: {e}")
    model, vectorizer = None, None

# --- Função principal de análise ---
def analisar_sentimento(frase: str) -> str:
    if not model or not vectorizer:
        return "Modelo não carregado."

    frase_limpa = limpar_texto(frase)
    vetor = vectorizer.transform([frase_limpa])
    resultado = model.predict(vetor)[0]

    # Normaliza o nome do sentimento
    sentimento_formatado = resultado.capitalize()

    return sentimento_formatado