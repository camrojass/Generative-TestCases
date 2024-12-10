import json
import os
import time
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError
from pinecone import Pinecone
import config
import validation_pdf
import re

# Configuración de claves API.
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
if "PINECONE_API_KEY" not in os.environ:
    os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY

pc = Pinecone(api_key=config.PINECONE_API_KEY)
clientOAI = OpenAI(
    api_key=config.OPENAI_API_KEY
)

# Configuración del índice
INDEX_NAME = pc.Index(config.index_name)
DIMENSION = config.dimension  # Dimensiones del modelo text-embedding-ada-002
METRIC = config.metric  # Métrica para calcular similitud

def generate_embedding(text):
    """
    Genera un embedding utilizando el modelo text-embedding-ada-002 de OpenAI.
    """
    try:
        response_embedding = clientOAI.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        response_dict = response_embedding.model_dump()
        embedding = response_dict['data'][0]['embedding']  # Extraer el embedding generado
        return embedding
    except RateLimitError:
        print("Rate limit alcanzado. Reintentando en 10 segundos...")
        time.sleep(10)  # Esperar antes de intentar de nuevo
    except AuthenticationError:
        print("Error de autenticación. Verifica tu clave API.")
    except OpenAIError as e:
        print(f"Error al interactuar con OpenAI: {e}")
    except Exception as e:
        print(f"Error al generar embedding para el texto '{text}': {e}")
        return None

# Función para extraer parámetros
def extract_parameters_from_text(text):
    """
    Extrae parámetros del texto utilizando NLP y expresiones regulares.
    """
    parameters = []
    param_pattern = re.compile(r"\b([a-zA-Z0-9_]+):\s*(string|int|float|boolean|bool)\b", re.IGNORECASE)
    required_keywords = ["required", "mandatory", "must be provided"]
    optional_keywords = ["optional", "default", "nullable"]

    # Dividir el texto en líneas
    lines = text.split("\n")
    for line in lines:
        match = param_pattern.search(line)
        if match:
            param_name = match.group(1)
            param_type = match.group(2)

            # Determinar si el parámetro es requerido u opcional
            is_required = any(keyword in line.lower() for keyword in required_keywords)
            is_optional = any(keyword in line.lower() for keyword in optional_keywords)

            # Buscar valores predeterminados
            default_value = None
            if "default" in line.lower():
                default_match = re.search(r"default\s*[:=]\s*([\w\"']+)", line, re.IGNORECASE)
                if default_match:
                    default_value = default_match.group(1)

            parameters.append({
                "name": param_name,
                "type": param_type,
                "required": is_required and not is_optional,
                "default": default_value
            })
    return parameters

#Función para Extraer Códigos HTTP
def extract_http_statuses(text):
    """
    Extrae solo los códigos de estado HTTP desde el texto.
    """
    http_statuses = []
    # Expresión regular para detectar códigos HTTP (100-599)
    status_pattern = re.compile(r"\b(1[0-9]{2}|2[0-9]{2}|3[0-9]{2}|4[0-9]{2}|5[0-9]{2})\b")

    lines = text.split("\n")
    for line in lines:
        match = status_pattern.search(line)
        if match:
            status_code = match.group(1)
            http_statuses.append(status_code)

    return http_statuses

# Funcion para extraer métodos
def extract_method_from_text(line):
    """
    Extrae el método HTTP de una línea de texto.
    """
    match = re.search(r"\b(GET|POST|PUT|DELETE)\b", line, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Devolver el método en mayúsculas
    return "GET"  # Valor predeterminado


# Función para extraer endpoints y métodos usando heurísticas
def extract_endpoints_and_methods(docs):
    """
    Extrae endpoints, métodos y parámetros desde los documentos procesados utilizando heurísticas.
    :param docs:
    :return: extracted_cases
    """
    extracted_cases = []
    for doc_index, doc in enumerate(docs):
        text = doc.page_content
        lines = text.split("\n")
        for line_index, line in enumerate(lines):
            if "/api/" in line:  # Detectar líneas que contengan endpoints
                method = extract_method_from_text(line)  # Usar la nueva lógica de extracción
                endpoint = line.split()[0]
                surrounding_text = " ".join(lines[max(0, line_index - 2):line_index + 2])

                # Extraer parámetros y códigos de estado HTTP
                parameters = extract_parameters_from_text(surrounding_text)
                http_statuses = extract_http_statuses(surrounding_text)

                case_id = f"doc_{doc_index}_line_{line_index}"
                extracted_cases.append({
                    "id": case_id,
                    "endpoint": endpoint,
                    "method": method,
                    "parameters": parameters,
                    "http_statuses": http_statuses
                })
    return extracted_cases

def generate_test_case(input_file_path):
    try:
        # Cargar y procesar el archivo PDF
        if not validation_pdf.validate_pdf(input_file_path):
            raise ValueError("El archivo PDF no es válido o no contiene texto legible.")

        loader = UnstructuredPDFLoader(input_file_path)
        docs = loader.load()

        # Dividir el texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
                                                       length_function=len, is_separator_regex=False)
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError(f"No se generaron fragmentos del texto del archivo: {input_file_path}")

        # Extraer endpoints, métodos y parámetros del PDF
        extracted_cases = extract_endpoints_and_methods(splits)
        if not extracted_cases:
            raise ValueError(f"No se encontraron endpoints ni métodos en el archivo: {input_file_path}")

        # Configuración del modelo y prompt
        prompt_template = PromptTemplate(
            input_variables=["endpoint", "method", "parameters"],
            template=(
                "Genera un caso de prueba en JSON para el endpoint {endpoint} "
                "usando el método {method}, los parámetros {parameters} y el código de estado HTTP: {http_statuses}."
                "Incluye título, objetivo, pasos y resultados esperados."
            )
        )

        # Generar casos de prueba dinámicamente
        test_cases = []
        excel_data = []
        embeddings = []
        max_retries = 5
        for case in extracted_cases:
            for attempt in range(max_retries):
                try:
                    http_statuses_text = ", ".join(
                        [f"{status['status_code']} " for status in
                         case.get("http_statuses", [])]
                    ) if not case.get("http_statuses") else "HTTP 200"
                    response = clientOAI.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "user",
                            "content": prompt_template.format(
                                endpoint=case["endpoint"],
                                method=case["method"],
                                parameters=case["parameters"],
                                http_statuses=http_statuses_text
                            )
                        }],
                        max_tokens=config.token,  # Limitar la respuesta a X tokens
                        temperature=config.temperature
                    )

                    response_dict = response.model_dump()
                    test_case_json = response_dict['choices'][0]['message']['content']

                    cleaned_content = test_case_json.replace("\n", " ").strip()
                    print(cleaned_content)
                    test_cases.append(cleaned_content)
                    excel_data.append({
                        "Título": f"Prueba para {case['endpoint']}",
                        "Objetivo": f"Validar la funcionalidad del endpoint {case['endpoint']} con el método {case['method']}",
                        "Pasos": f"Enviar una solicitud {case['method']} al endpoint {case['endpoint']} con los parámetros {case['parameters']}",
                        "Resultado Esperado": f"Validar los códigos HTTP: {case['http_statuses']}"
                    })

                    # Generar embedding para el endpoint
                    if "id" in case:
                        embedding = generate_embedding(case["endpoint"])
                        if embedding:
                            embeddings.append((case["id"], embedding))
                    else:
                        print(f"Advertencia: El caso no tiene un 'id'. Caso: {case}")
                    break
                except RateLimitError:
                    if attempt < max_retries - 1:
                        print(
                            f"Rate limit alcanzado. Reintentando en {2 ** attempt} segundos... (Intento {attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)  # Retrasos exponenciales
                    else:
                        print("Se alcanzó el límite de reintentos por Rate Limiting.")
                        raise
                except AuthenticationError:
                    print("Error de autenticación. Verifica tu clave API.")
                except OpenAIError as e:
                    print(f"Error al interactuar con OpenAI: {e}")
                except Exception as e:
                    print(f"Error inesperado al llamar a la API: {e}")
                    raise

            # Almacenar embeddings en Pinecone
            if embeddings:
                print(f"Almacenando {len(embeddings)} embeddings en el índice '{INDEX_NAME}'...")
                INDEX_NAME.upsert(vectors=embeddings)
                print("Embeddings almacenados exitosamente.")
            else:
                print("No se generaron embeddings para almacenar.")

        return test_cases, excel_data
    except Exception as e:
        print(f"Error en generate_test_case: {e}")
        raise


def save_postman_json(test_cases, output_file_path, input_file_path):
    """
    Guarda los casos de prueba en un archivo JSON para Postman.
    """
    # Extraer el nombre del archivo PDF sin extensión
    pdf_name = os.path.basename(input_file_path).replace(".pdf", "")

    # Crear la colección Postman con el nombre dinámico
    postman_collection = {
        "info": {
            "name": f"Test API Collection for {pdf_name}",
            "_postman_id": "12345",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": test_cases
    }

    # Guardar el archivo JSON
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(postman_collection, file, indent=4, ensure_ascii=False)

    print(f"Archivo JSON guardado con éxito en {output_file_path}.")


def save_excel(excel_data, excel_file_path):
    """
    Guarda los casos de prueba en un archivo Excel.
    """
    df = pd.DataFrame(excel_data)
    df.to_excel(excel_file_path, index=False, engine='openpyxl')
    print(f"Detalles guardados en el archivo Excel {excel_file_path}.")


# Ejecución del proceso.
input_file_path = config.input_file_path
output_json_path = config.output_file_path
output_excel_path = config.output_excel_file_path

# Generación de casos de prueba y guardado de resultados.
try:
    # Generar casos de prueba y embeddings
    test_cases, excel_data = generate_test_case(input_file_path)
    print("Embeddings almacenados exitosamente en Pinecone.")
    # Guardar los casos de prueba en JSON y Excel
    save_postman_json(test_cases, output_json_path, input_file_path)
    save_excel(excel_data, output_excel_path)
    print("Casos de prueba generados y guardados con éxito.")
except Exception as e:
    print(f"Error en la generación de casos de prueba: {e}")
