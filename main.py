import json
import os
import time
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError
from pinecone import Pinecone
import uuid
# python files
import config
import validation_pdf
import extract_info


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
    except Exception as e:
        print(f"Error al generar embedding para el texto '{text}': {e}")
        return None

def call_openai_with_retries(func, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except AuthenticationError:
            print("Error de autenticación. Verifica tu clave API.")
        except OpenAIError as e:
            print(f"Error al interactuar con OpenAI: {e}")
        except RateLimitError:
            if attempt < max_retries - 1:
                print(f"Rate limit alcanzado. Reintentando en {2 ** attempt} segundos...")
                time.sleep(2 ** attempt)  # Retrasos exponenciales
            else:
                print("Se alcanzó el límite de reintentos por Rate Limiting.")
                raise
        except Exception as e:
            print(f"Error inesperado al llamar a la API: {e}")
            raise

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
            print(f"Advertencia: No se generaron fragmentos del texto del archivo: {input_file_path}")
            return [], []

        # Extraer endpoints, métodos y parámetros del PDF
        extracted_cases = extract_info.extract_endpoints_and_methods(splits)
        if not extracted_cases:
            print(f"Advertencia: No se encontraron endpoints ni métodos en el archivo: {input_file_path}")
            return [], []

        # Configuración del modelo y prompt
        prompt_template = PromptTemplate(
            input_variables=["endpoint", "method", "parameters"],
            template=(
                "Genera al menos dos casos de prueba en JSON para el endpoint {endpoint} "
                "usando el método {method}, los parámetros:\n{parameters} "
                "y los códigos de estado HTTP: {http_statuses}.\n"
                "Incluye título, objetivo, pasos y resultados esperados."
            )
        )

        # Generar casos de prueba dinámicamente
        test_cases = []
        excel_data = []
        embeddings = []
        postman_collect_items = []
        for case in extracted_cases:
            try:
                http_statuses_text = ", ".join(case["http_statuses"]) or "HTTP 200"
                parameters_text = "\n".join([f"- {param['name']} ({param['type']}) ({param['source']})" for param in case["parameters"]]) or "Sin parámetros"
                response = call_openai_with_retries(
                    clientOAI.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": prompt_template.format(
                            endpoint=case["endpoint"],
                            method=case["method"],
                            parameters=parameters_text,
                            http_statuses=http_statuses_text
                        )
                    }],
                    max_tokens=config.token,
                    temperature=config.temperature
                )

                response_dict = response.model_dump()
                test_case_json = response_dict['choices'][0]['message']['content']
                cleaned_content = test_case_json.replace("\n", " ").strip()
                # Preparar data para postman y excel
                test_cases.append(cleaned_content)
                excel_data.append({
                    "Título": f"Prueba para {case['endpoint']}",
                    "Objetivo": f"Validar la funcionalidad del endpoint {case['endpoint']} con el método {case['method']}",
                    "Pasos": f"Enviar una solicitud {case['method']} al endpoint {case['endpoint']} con los parámetros {case['parameters']}",
                    "Resultado Esperado": f"Validar los códigos HTTP: {case['http_statuses']}"
                })
                # Almacenar en Postman JSON
                postman_collect_items.append({
                    "name": f"Prueba para {case['endpoint']} con metodo {case['method']}",
                    "request": {
                        "method": case["method"],
                        "url": {
                            "raw": case["endpoint"],
                            "host": [case["endpoint"]]
                        },
                        "body": {"mode": "raw", "raw": json.dumps(case["parameters"], indent=2)},
                        "description": f"Validar los códigos HTTP: {http_statuses_text}"
                    }
                }
                )
                # Generar embedding para el endpoint
                if "id" in case:
                    embedding = generate_embedding(case["endpoint"])
                    if embedding:
                        embeddings.append((case["id"], embedding))
                else:
                    print(f"Advertencia: El caso no tiene un 'id'. Caso: {case}")

            except Exception as e:
                print(f"Error en generate_test_case: {e}")
                raise

            # Almacenar embeddings en Pinecone
            if embeddings:
                print(f"Almacenando {len(embeddings)} embeddings en el índice '{INDEX_NAME}'...")
                INDEX_NAME.upsert(vectors=embeddings)
                print("Embeddings almacenados exitosamente.")
            else:
                print("No se generaron embeddings para almacenar.")

        return postman_collect_items, excel_data
    except Exception as e:
        print(f"Error en generate_test_case: {e}")
        raise


def save_postman_json(postman_collect_items, output_file_path, input_file_path):
    """
    Guarda los casos de prueba en un archivo JSON para Postman.
    """
    # Extraer el nombre del archivo PDF sin extensión
    pdf_name = os.path.basename(input_file_path).replace(".pdf", "")

    # Crear la colección Postman con el nombre dinámico
    postman_collection = {
        "info": {
            "name": f"Test API Collection for {pdf_name}",
            "_postman_id": f"{uuid.uuid4()}",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": postman_collect_items
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
    print("Casos de prueba generados exitosamente.")
    # Guardar los casos de prueba en JSON y Excel
    save_postman_json(test_cases, output_json_path, input_file_path)
    save_excel(excel_data, output_excel_path)
    print("Casos de prueba guardados con éxito.")
except Exception as e:
    print(f"Error en la generación de casos de prueba: {e}")
