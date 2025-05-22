import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from extract_cv import extract_text_from_pdf
from load_data import preprocess_cv_text, model, vectorizer
import numpy as np

def cargar_pdf():
    filepath = filedialog.askopenfilename(
        filetypes=[("PDF Files", "*.pdf")],
        title="Selecciona un archivo PDF"
    )
    if not filepath:
        return

    # Extraer texto del PDF
    texto = extract_text_from_pdf(filepath)
    if not texto.strip():
        messagebox.showerror("Error", "No se pudo extraer texto del PDF.")
        return

    # Procesar texto para el modelo
    texto_procesado = preprocess_cv_text(texto)
    texto_vector = vectorizer.transform([texto_procesado])
    
    # Predecir con el modelo (supongamos que devuelve probabilidades para cada clase)
    proba = model.predict_proba(texto_vector)[0]  # arreglo con probabilidades
    clases = model.classes_  # nombres de las posiciones
    
    # Obtener la posición con mayor probabilidad y su porcentaje
    max_index = np.argmax(proba)
    posicion_pred = clases[max_index]
    confianza = proba[max_index] * 100

    # Actualizar la interfaz con resultados
    label_posicion.config(text=f"Posición Predicha: {posicion_pred}")
    progress_var.set(confianza)
    label_confianza.config(text=f"Confianza: {confianza:.2f}%")

    # Mostrar posiciones posibles con probabilidad mayor a cierto umbral (ejemplo 5%)
    resultados_text.delete("1.0", tk.END)
    resultados_text.insert(tk.END, "Posiciones posibles:\n")
    for i in np.argsort(proba)[::-1]:  # de mayor a menor
        if proba[i] > 0.05:
            resultados_text.insert(tk.END, f"- {clases[i]}: {proba[i]*100:.2f}%\n")

# Crear ventana principal
root = tk.Tk()
root.title("Clasificador de Currículums")
root.geometry("500x400")
root.resizable(False, False)

# Botón para cargar PDF
btn_cargar = ttk.Button(root, text="Cargar Currículum PDF", command=cargar_pdf)
btn_cargar.pack(pady=20)

# Label para posición predicha
label_posicion = ttk.Label(root, text="Posición Predicha: ", font=("Arial", 14))
label_posicion.pack(pady=10)

# Barra de progreso para confianza
progress_var = tk.DoubleVar()
progress = ttk.Progressbar(root, variable=progress_var, maximum=100, length=300)
progress.pack(pady=5)

# Label para porcentaje confianza
label_confianza = ttk.Label(root, text="Confianza: 0%", font=("Arial", 12))
label_confianza.pack()

# Text box para mostrar posibles posiciones
resultados_text = tk.Text(root, height=8, width=50, state=tk.NORMAL)
resultados_text.pack(pady=15)
resultados_text.insert(tk.END, "Porcentajes de Contratación:\n")

root.mainloop()
