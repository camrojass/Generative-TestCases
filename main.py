import os
import json
import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import config
import time
import openai

# Configuración de claves API.
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Configuración del índice
INDEX_NAME = pc.Index(config.index_name)
DIMENSION = config.dimension  # Dimensiones del modelo text-embedding-ada-002
METRIC = config.metric  # Métrica para calcular similitud
# Conectar al índice
index = INDEX_NAME
"""
# Crear o verificar el índice en Pinecone
if index not in pc.list_indexes():
    print(f"Creando índice '{index}' en Pinecone")
    pc.create_index(
        name=index,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Índice '{index}' ya existe en Pinecone.")
"""
def generate_embedding(text):
    """
    Genera un embedding utilizando el modelo text-embedding-ada-002 de OpenAI.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']  # Extraer el embedding generado
        return embedding
    except Exception as e:
        print(f"Error al generar embedding para el texto '{text}': {e}")
        return None

def extract_endpoints_and_methods(docs):
    """
    Extrae endpoints, métodos y parámetros desde los documentos procesados.
    """
    extracted_cases = []
    for doc in docs:
        # Usar análisis básico para extraer información clave del texto (puedes usar NLP avanzado si es necesario).
        text = doc.page_content
        # Simula la extracción: busca patrones típicos de endpoints y métodos (mejorable con NLP).
        lines = text.split("\n")
        for line in lines:
            if "/api/" in line and ("GET" in line or "POST" in line or "PUT" in line or "DELETE" in line):
                method = next((m for m in ["GET", "POST", "PUT", "DELETE"] if m in line), "GET")
                endpoint = line.split()[0]
                parameters = {"example_param": "value"}  # Puedes ajustar cómo se identifican parámetros.
                extracted_cases.append({"endpoint": endpoint, "method": method, "parameters": parameters})
    return extracted_cases

def generate_test_case(input_file_path):
    try:
        # Cargar y procesar el archivo PDF
        loader = UnstructuredPDFLoader(input_file_path)
        docs = loader.load()
        if not docs:
            raise ValueError(f"No se pudo cargar contenido del archivo: {input_file_path}")

        # Dividir el texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, length_function=len, is_separator_regex=False)
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError(f"No se generaron fragmentos del texto del archivo: {input_file_path}")

        # Crear el vector store
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Extraer endpoints, métodos y parámetros del PDF
        extracted_cases = extract_endpoints_and_methods(splits)
        if not extracted_cases:
            raise ValueError(f"No se encontraron endpoints ni métodos en el archivo: {input_file_path}")

        # Configuración del modelo y prompt
        prompt_template = PromptTemplate(
            input_variables=["endpoint", "method", "parameters"],
            template=(
                "Genera un caso de prueba en JSON para el endpoint {endpoint} "
                "usando el método {method} y los parámetros {parameters}. "
                "Incluye título, objetivo, pasos y resultados esperados."
            )
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Generar casos de prueba dinámicamente con `max_tokens`
        test_cases = []
        excel_data = []
        embeddings = []
        max_retries = 5
        for case in extracted_cases:
            for attempt in range(max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "user",
                            "content": prompt_template.format(
                                endpoint=case["endpoint"],
                                method=case["method"],
                                parameters=case["parameters"]
                            )
                        }],
                        max_tokens=config.token,  # Limitar la respuesta a XXX tokens
                        temperature=config.temperature
                    )
                    test_case_json = response['choices'][0]['message']['content']
                    test_cases.append(test_case_json)
                    excel_data.append({
                        "Título": f"Prueba para {case['endpoint']}",
                        "Objetivo": f"Validar la funcionalidad del endpoint {case['endpoint']} con el método {case['method']}",
                        "Pasos": f"Enviar una solicitud {case['method']} al endpoint {case['endpoint']} con los parámetros {case['parameters']}",
                        "Resultado Esperado": "Recibir un código de estado 200 y los datos esperados"
                    })

                    # Generar embedding para el endpoint
                    embedding = generate_embedding(case["endpoint"])
                    if embedding:
                        embeddings.append((case["id"], embedding))

                    break
                except openai.error.RateLimitError:
                    if attempt < max_retries - 1:
                        print(f"Rate limit alcanzado. Reintentando en {2**attempt} segundos... (Intento {attempt + 1}/{max_retries})")
                        time.sleep(2**attempt)  # Retrasos exponenciales
                    else:
                        print("Se alcanzó el límite de reintentos por Rate Limiting.")
                        raise
                except Exception as e:
                    print(f"Error inesperado al llamar a la API: {e}")
                    raise

            # Almacenar embeddings en Pinecone
            if embeddings:
                print(f"Almacenando {len(embeddings)} embeddings en el índice '{INDEX_NAME}'...")
                index.upsert(vectors=embeddings)
                print("Embeddings almacenados exitosamente.")
            else:
                print("No se generaron embeddings para almacenar.")

        return test_cases, excel_data, embeddings

    except Exception as e:
        print(f"Error en generate_test_case: {e}")
        raise

def save_postman_json(test_cases, output_file_path):
    """
    Guarda los casos de prueba en un archivo JSON para Postman.
    """
    postman_collection = {
        "info": {"name": "Test API Collection", "_postman_id": "12345", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
        "item": test_cases
    }
    with open(output_file_path, 'w') as file:
        json.dump(postman_collection, file, indent=4)

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
    save_postman_json(test_cases, output_json_path)
    save_excel(excel_data, output_excel_path)
    print("Casos de prueba generados y guardados con éxito.")
except Exception as e:
    print(f"Error en la generación de casos de prueba: {e}")
