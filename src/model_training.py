import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def main():
    archivo = r'C:\Users\david.rodriguez\Documents\DD\IA\Seminario_IA\data\Candidatos.xlsx'

    df = pd.read_excel(archivo, engine='openpyxl')

    # Asegurémonos de las columnas
    # Vamos a usar "POSICIÓN" como target (multi-clase) y "CONTRATADO/NO CONTRATADO" como target binario, aquí ejemplo simplificado:
    
    # Para simplificar, transformaremos contratado a 0/1
    df = df.dropna(subset=['POSICIÓN', 'CONTRATADO/NO CONTRATADO'])
    df['CONTRATADO'] = df['CONTRATADO/NO CONTRATADO'].apply(
    lambda x: 1 if str(x).lower() == 'contratado' else 0
)

    df['texto'] = df['EXPERIENCIA'].fillna('').astype(str) + ' ' + \
              df['NIVEL ACADÉMICO'].fillna('').astype(str) + ' ' + \
              df['POSICIÓN'].fillna('').astype(str)


    X = df['texto']
    y = df['CONTRATADO']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    print(f"Precisión en test: {model.score(X_test, y_test):.2f}")

    # Guardar modelo y vectorizador
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == '__main__':
    main()
