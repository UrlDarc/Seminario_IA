# Seminario_IA

Análisis automatizado de currículums con IA

Este proyecto utiliza técnicas de Inteligencia Artificial y Procesamiento de Lenguaje Natural (NLP) para analizar currículums (CVs) en PDF y determinar si un candidato es apto para ser contratado, así como sugerir el puesto más adecuado (por ejemplo: Líder Operativo o Gerente).

📌 Objetivo
Automatizar el análisis de hojas de vida para mejorar el proceso de reclutamiento en empresas con alta demanda de personal, como cadenas de restaurantes, tiendas de conveniencia o supermercados.

🧠 Tecnologías y herramientas
Python 3.11+

Scikit-learn

Pandas

NumPy

Tkinter (Interfaz gráfica)

PyPDF2 (Extracción de texto)

Joblib (Serialización del modelo entrenado)

Jupyter Notebook (para prototipos y pruebas)

Seminario_IA/
├── src/
│   ├── interface.py          # Interfaz gráfica
│   ├── model_training.py     # Entrenamiento del modelo
│   ├── extract_cv.py         # Extracción de texto desde PDF
│   └── modelo_entrenado.joblib  # Modelo serializado
├── dataset/
│   └── candidatos.csv        # Datos históricos (experiencia, nivel, posición, etc.)
├── README.md

🚀 Cómo usar

- Clona el repositorio:
git clone https://github.com/tuusuario/IA-candidato-clasificador.git
cd IA-candidato-clasificador

- Instala las dependencias:
pip install -r requirements.txt

- Ejecuta la interfaz:
python src/interface.py

# Carga un CV en PDF y el sistema:

- Procesará el texto del CV
- Predecirá si el candidato es contratable o no contratable
- Mostrará el porcentaje de probabilidad

💡 Posibilidades de mejora
Incluir más puestos (Asistente, Supervisor, etc.)

Agregar un sistema de retroalimentación al modelo

Implementar integración con bases de datos de candidatos

Desarollador: David Rodríguez - 1164619
