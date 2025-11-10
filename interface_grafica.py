import tkinter as tk
import joblib

# --- Carregar modelo e vetorizador treinados ---
modelo = joblib.load('modelo_svm.pkl')
vetorizador = joblib.load('vectorizer_tfidf.pkl')

# --- Função para prever sentimento ---
def analisar_sentimento(texto):
    texto = texto.strip()
    if not texto:
        return "Digite algo!"
    vetor = vetorizador.transform([texto])
    sentimento = modelo.predict(vetor)[0]
    return sentimento

# --- Interface Gráfica ---
janela = tk.Tk()
janela.title("Analisador de Sentimentos 🧠💬")
janela.geometry("420x300")
janela.resizable(False, False)
janela.configure(bg="#f2f2f2")

# Título
titulo = tk.Label(
    janela,
    text="Analisador de Sentimentos",
    font=("Segoe UI", 16, "bold"),
    bg="#f2f2f2",
    fg="#333"
)
titulo.pack(pady=10)

# Campo de entrada
entrada = tk.Entry(janela, width=50, font=("Segoe UI", 11))
entrada.pack(pady=10)

# Label de resultado
label_resultado = tk.Label(
    janela,
    text="",
    font=("Segoe UI", 12),
    bg="#f2f2f2",
    fg="#000"
)
label_resultado.pack(pady=20)

# Função de processamento
def executar_analise():
    texto = entrada.get().strip()
    if texto:
        sentimento = analisar_sentimento(texto)
        label_resultado["text"] = f"🩵 Sentimento detectado: {sentimento}"
    else:
        label_resultado["text"] = "Digite uma frase para analisar."

# Botão
botao = tk.Button(
    janela,
    text="Analisar",
    command=executar_analise,
    font=("Segoe UI", 11, "bold"),
    bg="#4CAF50",
    fg="white",
    relief="raised",
    padx=10,
    pady=5
)
botao.pack(pady=10)

# Rodar app
janela.mainloop()