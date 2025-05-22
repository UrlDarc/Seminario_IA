import joblib
import re

# Cargamos modelo y vectorizador solo una vez al importar
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def preprocess_cv_text(text):
    # Limpiar texto simple: quitar caracteres no alfanuméricos, pasar a minúsculas
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    return text

def predict_from_text(text):
    clean_text = preprocess_cv_text(text)
    vect_text = vectorizer.transform([clean_text])
    pred_prob = model.predict_proba(vect_text)[0][1]  # probabilidad de contratado
    pred_class = model.predict(vect_text)[0]         # 0 o 1
    return pred_class, pred_prob
