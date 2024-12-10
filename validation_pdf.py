import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

# Validación del PDF
def is_valid_pdf(file_path):

    """
    Antes de procesar el archivo, se asegura de que el archivo existe en la ruta proporcionada y tiene una extensión .pdf.
    :param file_path:
    :return: True/False
    """
    if not os.path.exists(file_path):
        print(f"Error: El archivo no existe en la ruta {file_path}.")
        return False
    if not file_path.lower().endswith('.pdf'):
        print(f"Error: El archivo {file_path} no tiene una extensión válida .pdf.")
        return False
    return True

def is_pdf_readable(file_path):

    """
    para validar la estructura básica del PDF y verificar que se pueda leer.
    :param file_path:
    :return: True/Fase
    """
    try:
        reader = PdfReader(file_path)
        if len(reader.pages) == 0:
            print(f"Error: El archivo PDF {file_path} no contiene páginas.")
            return False
        return True
    except Exception as e:
        print(f"Error al leer el archivo PDF {file_path}: {e}")
        return False

def has_text_content(file_path):

    """
    verificar si el contenido de texto está presente utilizando PyPDF2 o un método similar.
    :param file_path:
    :return: True/Fañse
    """
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text().strip():  # Verifica si hay texto extraído
                return True
        print(f"Advertencia: El archivo PDF {file_path} no contiene texto legible.")
        return False
    except Exception as e:
        print(f"Error al leer el contenido del PDF {file_path}: {e}")
        return False

def extract_text_with_ocr(file_path):

    """
    Si el PDF no contiene texto legible, se usa herramientas de OCR (Reconocimiento Óptico de Caracteres) para extraer texto de las imágenes.
    :param file_path: input_file_path
    :return: text/none
    """

    try:
        images = convert_from_path(file_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        if text.strip():
            return text
        print(f"Advertencia: El archivo PDF {file_path} no contiene texto después del OCR.")
        return None
    except Exception as e:
        print(f"Error al procesar el archivo PDF con OCR: {e}")
        return None

def validate_pdf(file_path):

    """
    Valida PDF antes de procesarlo
    :param file_path: input_file_path
    :return: true/false
    """
    if not is_valid_pdf(file_path):
        return False
    if not is_pdf_readable(file_path):
        return False
    if not has_text_content(file_path):
        print("Intentando extraer texto con OCR...")
        text = extract_text_with_ocr(file_path)
        if text:
            print("Texto extraído con éxito usando OCR.")
            return True
        else:
            print("No se pudo extraer texto del archivo PDF.")
            return False
    return True