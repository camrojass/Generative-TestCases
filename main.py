import os
import json
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
import config

# Configuración de claves API.
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
os.environ["PINECONE_ENV"] = config.PINECONE_ENV

def generate_test_case(file_path):
    # Cargar y procesar el archivo PDF.
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()

    # División en fragmentos.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    # Creación del vector store.
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Configuración de LLM y prompt.
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Creación de casos de prueba en JSON.
    prompt_template = "Genera un caso de prueba en JSON para el endpoint {endpoint} con el método {method} y los parámetros {parameters}."
    llm_input = {
        "endpoint": "/api/v1/user",
        "method": "POST",
        "parameters": {"username": "test_user", "password": "1234"}
    }
    test_case_json = llm(prompt_template.format(**llm_input))

    return test_case_json

def save_postman_json(test_cases, output_file_path):
    # Estructura de colección Postman.
    postman_collection = {
        "info": {"name": "Test API Collection", "_postman_id": "12345", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
        "item": test_cases
    }
    with open(output_file_path, 'w') as file:
        json.dump(postman_collection, file, indent=4)

def execute_tests(output_file_path):
    # Ejecuta las pruebas en Newman (Postman CLI).
    os.system(f'newman run {output_file_path} --reporters cli,html --reporter-html-export results.html')

# Ejecución del proceso.
input_file_path = "C:\\Repos\\Proyect\\input\\doc.txt"
output_file_path = "C:\\Repos\\Proyect\\output\\postman_collection.json"

# Generación y guardado del archivo de colección de Postman.
test_cases = [generate_test_case(input_file_path)]
save_postman_json(test_cases, output_file_path)

# Ejecución de las pruebas con Newman.
execute_tests(output_file_path)
