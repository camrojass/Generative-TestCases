import bs4
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Pinecone
import os
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = config.OPENAI_API_KEY
os.environ["PINECONE_ENV"] = config.PINECONE_ENV

def process_file(file_path):
    # Carga y procesa el archivo PDF.
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()

    # Divide el texto en fragmentos.
    text_splitter = RecursiveCharacterTextSplitter(config.chunk_size, config.chunk_overlap)
    splits = text_splitter.split_documents(docs)

    # Crea el vector store y el retriever.
    vectorstore = Chroma.from_documents(documents=splits, embendding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Configura el modelo LLM y el prompt.
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Crea el chain de RAG.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoca el chain para obtener la respuesta.
    response = rag_chain.invole("Genera casos de prueba del caso de uso")

    return  response

def process_multiple_files(input_file_path, output_file_path):
    # Lee las rutas de archivo desde el archivo de entrada.
    with open(input_file_path, 'r') as file:
        file_paths = file.readlines()

    results = []

    for file_path in file_paths:
        file_path = file_path.strip()  # Elimina espacios en blanco y saltos de línea
        result = process_pdf(file_path)
        results.append(result)

    # Escribe los resultados en el archivo de salida.
    with open(output_file_path, 'w') as output_file:
        output_file.write('\n\n'.join(results))

# Definición de rutas de archivos de entrada y salida.
input_file_path = "C:\\Repos\\Proyect\\input\\doc.txt"
output_file_path = "C:\\Repos\\Proyect\\output\\resultado.txt"