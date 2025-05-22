# extract_cv.py

import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    texto = ""
    with fitz.open(file_path) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto
