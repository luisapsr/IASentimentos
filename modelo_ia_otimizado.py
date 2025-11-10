import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# --- Caminho do dataset ---
DATASET_PATH = "./database_sentimento/dataset_sentimentos.csv"

# --- Função de limpeza de texto ---
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)  # remove links
    texto = re.sub(r"[^\w\sáéíóúâêîôûãõç]", "", texto)  # mantém acentos
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

print("📂 Carregando dataset...")
try:
    df = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8", on_bad_lines="skip")
except UnicodeDecodeError:
    df = pd.read_csv(DATASET_PATH, sep=";", encoding="latin1", on_bad_lines="skip")

if not {'frase', 'sentimento'}.issubset(df.columns):
    raise ValueError("O CSV precisa conter as colunas 'frase' e 'sentimento'.")

print("🔍 Quantidade original por classe:")
print(df["sentimento"].value_counts())

# --- Limpeza ---
print("🧹 Limpando textos...")
df["frase_limpa"] = df["frase"].astype(str).apply(limpar_texto)

# --- Balanceamento das classes ---
print("⚖️ Balanceando classes...")
min_count = df["sentimento"].value_counts().min()
df_bal = pd.concat([
    resample(grupo, replace=True, n_samples=min_count, random_state=42)
    for _, grupo in df.groupby("sentimento")
])

print("✅ Quantidade balanceada por classe:")
print(df_bal["sentimento"].value_counts())

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    df_bal["frase_limpa"], df_bal["sentimento"], test_size=0.2, random_state=42
)

# --- Vetorização TF-IDF ---
print("🔤 Gerando vetores TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Treino com GridSearch ---
print("🤖 Buscando melhor modelo (GridSearch)...")
param_grid = {
    "C": [0.1, 1, 5, 10],
    "loss": ["hinge", "squared_hinge"]
}
grid = GridSearchCV(LinearSVC(), param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train_tfidf, y_train)

melhor_modelo = grid.best_estimator_
print(f"🏆 Melhor modelo: {grid.best_params_}")

# --- Avaliação ---
print("📊 Avaliando modelo...")
y_pred = melhor_modelo.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Acurácia final: {acc:.2%}\n")
print(classification_report(y_test, y_pred))

# --- Salvar modelo ---
print("💾 Salvando modelo e vetorizador...")
joblib.dump(melhor_modelo, "modelo_svm_otimizado.pkl")
joblib.dump(vectorizer, "vectorizer_tfidf_otimizado.pkl")

print("\n🏁 Treinamento otimizado concluído com sucesso!")