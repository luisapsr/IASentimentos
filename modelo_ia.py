import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# --- Caminho do dataset ---
DATASET_PATH = "./database_sentimento/dataset_sentimentos.csv"

# --- Função de limpeza de texto ---
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)  # remove links
    texto = re.sub(f"[{string.punctuation}]", "", texto)  # remove pontuação
    texto = re.sub(r"\d+", "", texto)  # remove números
    texto = texto.strip()
    return texto

# --- Carregar dataset ---
print("📂 Carregando dataset...")
DATASET_PATH = "dataset_sentimentos.csv"
df = pd.read_csv(DATASET_PATH, sep=';', on_bad_lines='skip', encoding='utf-8')

# Verifica se as colunas existem
if "frase" not in df.columns or "sentimento" not in df.columns:
    raise ValueError("O arquivo CSV precisa conter as colunas 'frase' e 'sentimento'.")

# --- Limpeza do texto ---
print("🧹 Limpando textos...")
df["frase_limpa"] = df["frase"].astype(str).apply(limpar_texto)

# --- Separar treino e teste ---
X_train, X_test, y_train, y_test = train_test_split(
    df["frase_limpa"], df["sentimento"], test_size=0.2, random_state=42
)

# --- Vetorização TF-IDF ---
print("🔤 Gerando vetores TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Treinamento do modelo SVM ---
print("🤖 Treinando modelo SVM...")
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# --- Avaliação ---
print("📊 Avaliando modelo...")
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Acurácia: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# --- Salvar modelo e vetorizador ---
print("💾 Salvando modelo e vetorizador...")
joblib.dump(model, "modelo_svm.pkl")
joblib.dump(vectorizer, "vectorizer_tfidf.pkl")

print("\n🏁 Treinamento concluído com sucesso!")
print("📁 Arquivos gerados: modelo_svm.pkl e vectorizer_tfidf.pkl")