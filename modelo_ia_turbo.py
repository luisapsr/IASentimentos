import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
import nlpaug.augmenter.word as naw
import nltk

# --- Garantir recursos NLTK necessários ---

# WordNet
try:
    from nltk.corpus import wordnet
    wordnet.synsets('test')
except LookupError:
    print("⚡ WordNet não encontrado. Baixando...")
    nltk.download('wordnet')

# POS tagger inglês
try:
    from nltk import pos_tag
    pos_tag(["test"])
except LookupError:
    print("⚡ POS tagger não encontrado. Baixando...")
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

# --- Caminho do dataset ---
DATASET_PATH = "./database_sentimento/dataset_sentimentos.csv"

# --- Função de limpeza ---
def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(f"[{string.punctuation}]", "", texto)
    texto = re.sub(r"\d+", "", texto)
    texto = texto.strip()
    return texto

# --- Carregar dataset ---
print("📂 Carregando dataset...")
df = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8", on_bad_lines="skip")

# --- Validação básica ---
if not {'frase', 'sentimento'}.issubset(df.columns):
    raise ValueError("O dataset precisa conter as colunas 'frase' e 'sentimento'.")

df.dropna(subset=['frase', 'sentimento'], inplace=True)
df["frase_limpa"] = df["frase"].apply(limpar_texto)

# --- Balancear classes ---
print("⚖️ Balanceando classes...")
min_count = df["sentimento"].value_counts().min()
df_bal = df.groupby("sentimento").sample(min_count, random_state=42).reset_index(drop=True)

# --- Data augmentation (gera novas frases com sinônimos) ---
print("🧬 Aumentando dataset com sinônimos...")
aug = naw.SynonymAug(aug_src='wordnet')

df_aug = df_bal.copy()
# Certifica que cada célula é uma string (não lista) após o aumento
df_aug["frase_limpa"] = df_aug["frase_limpa"].apply(lambda x: " ".join(aug.augment(x)))

# Combina dataset original + dataset aumentado
df_final = pd.concat([df_bal, df_aug]).reset_index(drop=True)

# --- Separar treino e teste ---
X_train, X_test, y_train, y_test = train_test_split(
    df_final["frase_limpa"], df_final["sentimento"], test_size=0.2, random_state=42
)

# --- Gerar embeddings BERT ---
print("🧠 Gerando embeddings BERT...")
model_embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X_train_emb = model_embedding.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = model_embedding.encode(X_test.tolist(), show_progress_bar=True)

# --- Codificar labels ---
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# --- Treinar modelo neural MLP ---
print("🤖 Treinando rede neural MLP...")
modelo = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                       solver='adam', max_iter=500, random_state=42)
modelo.fit(X_train_emb, y_train_enc)

# --- Avaliar ---
print("📊 Avaliando modelo...")
y_pred = modelo.predict(X_test_emb)
acc = accuracy_score(y_test_enc, y_pred)
print(f"\n✅ Acurácia: {acc * 100:.2f}%\n")
print(classification_report(y_test_enc, y_pred, target_names=encoder.classes_))

# --- Salvar modelo, encoder e vetorizador BERT ---
print("💾 Salvando modelo, encoder e vetorizador BERT...")
joblib.dump(modelo, "modelo_mlp_turbo.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(model_embedding, "bert_vectorizer.pkl")

print("\n🏁 Treinamento concluído com sucesso!")
print("📁 Gerados: modelo_mlp_turbo.pkl | label_encoder.pkl | bert_vectorizer.pkl")