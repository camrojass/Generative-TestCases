# Gen_TestCase
La documentación generada con apoyo de ChatGPT

### Documentación del Código

Este código es una solución automatizada para procesar documentos PDF, extraer endpoints de API, generar casos de prueba con OpenAI, almacenar embeddings en Pinecone, y guardar los resultados en formatos JSON (Postman) y Excel. A continuación se detalla cada sección:

---

### **1. Configuración**

#### **Claves API**
```python
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
pc = Pinecone(api_key=config.PINECONE_API_KEY)
```
- **`OPENAI_API_KEY`**: Clave API para interactuar con OpenAI.
- **`PINECONE_API_KEY`**: Clave API para interactuar con Pinecone.

#### **Índice de Pinecone**
```python
INDEX_NAME = pc.Index(config.index_name)
DIMENSION = config.dimension  # Dimensiones del modelo text-embedding-ada-002
METRIC = config.metric  # Métrica para calcular similitud
```
- Se configura el índice de Pinecone donde se almacenarán los embeddings generados. Si el índice no existe, se crea.

---

### **2. Funciones**

#### **2.1. `generate_embedding`**
```python
def generate_embedding(text):
    """
    Genera un embedding utilizando el modelo text-embedding-ada-002 de OpenAI.
    """
```
- Genera un embedding (vector numérico) para un texto dado usando el modelo `text-embedding-ada-002` de OpenAI.

#### **2.2. `extract_endpoints_and_methods`**
```python
def extract_endpoints_and_methods(docs):
    """
    Extrae endpoints, métodos y parámetros desde los documentos procesados.
    """
```
- Extrae información clave, como endpoints, métodos HTTP (GET, POST, PUT, DELETE) y parámetros, desde el contenido de los documentos PDF.

#### **2.3. `generate_test_case`**
```python
def generate_test_case(input_file_path):
    """
    Procesa un archivo PDF y genera casos de prueba dinámicos y embeddings.
    """
```
- **Flujo**:
  1. **Carga del PDF**: Utiliza `UnstructuredPDFLoader` para cargar el contenido del archivo PDF.
  2. **División del texto**: Usa `RecursiveCharacterTextSplitter` para dividir el texto en fragmentos manejables.
  3. **Extracción de endpoints**: Usa `extract_endpoints_and_methods` para identificar endpoints y métodos HTTP.
  4. **Generación de casos de prueba**: Usa OpenAI para generar casos de prueba detallados en formato JSON.
  5. **Almacenamiento en Pinecone**: Genera embeddings para cada endpoint y los almacena en el índice de Pinecone.

#### **2.4. `save_postman_json`**
```python
def save_postman_json(test_cases, output_file_path):
    """
    Guarda los casos de prueba en un archivo JSON para Postman.
    """
```
- Guarda los casos de prueba generados en un archivo JSON compatible con Postman.

#### **2.5. `save_excel`**
```python
def save_excel(excel_data, excel_file_path):
    """
    Guarda los casos de prueba en un archivo Excel.
    """
```
- Documenta los casos de prueba generados en un archivo Excel para facilitar la revisión.

---

### **3. Ejecución del Proceso**

```python
input_file_path = config.input_file_path
output_json_path = config.output_file_path
output_excel_path = config.output_excel_file_path
```
- **`input_file_path`**: Ruta del archivo PDF de entrada.
- **`output_json_path`**: Ruta del archivo JSON donde se guardarán los casos de prueba para Postman.
- **`output_excel_path`**: Ruta del archivo Excel donde se documentarán los casos de prueba.

---

### **4. Flujo Principal**

#### **Carga y Procesamiento del PDF**
```python
loader = UnstructuredPDFLoader(input_file_path)
docs = loader.load()
```
- Carga el contenido del archivo PDF.

#### **División del Texto**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    length_function=len,
    is_separator_regex=False
)
splits = text_splitter.split_documents(docs)
```
- Divide el contenido del PDF en fragmentos manejables.

#### **Generación de Casos de Prueba**
```python
prompt_template = PromptTemplate(
    input_variables=["endpoint", "method", "parameters"],
    template=(
        "Genera un caso de prueba en JSON para el endpoint {endpoint} "
        "usando el método {method} y los parámetros {parameters}. "
        "Incluye título, objetivo, pasos y resultados esperados."
    )
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```
- Usa un modelo de lenguaje (ChatGPT) para generar casos de prueba dinámicos basados en los endpoints extraídos.

#### **Almacenamiento en Pinecone**
```python
embedding = generate_embedding(case["endpoint"])
index.upsert(vectors=embeddings)
```
- Genera embeddings para cada endpoint y los almacena en Pinecone para búsquedas semánticas.

#### **Guardado de Resultados**
```python
save_postman_json(test_cases, output_json_path)
save_excel(excel_data, output_excel_path)
```
- Los casos de prueba se guardan en JSON (Postman) y Excel para documentación y análisis.

---

### **5. Salida**

1. **JSON para Postman**:
   - Contiene los casos de prueba generados, listos para importarse y ejecutarse en Postman.
   
2. **Excel**:
   - Documenta los endpoints, métodos y parámetros junto con los pasos y resultados esperados.

3. **Embeddings en Pinecone**:
   - Los embeddings generados para los endpoints se almacenan en un índice de Pinecone para búsquedas rápidas y escalables.

---

### **6. Requisitos Previos**

El cuadro a continuación presenta el resumen de los requisitos previos

| Categoría                | Detalle                                                                                               |
|--------------------------|------------------------------------------------------------------------------------------------------|
| **Bibliotecas**          | `langchain`, `pinecone-client`, `openai`, `chromadb`, `pandas`, `openpyxl`, `langchain_community`    |
| **Pinecone**             | Cuenta activa con clave API, entorno configurado, índice configurado con dimensiones correctas.      |
| **OpenAI**               | Cuenta activa con clave API y límites de uso configurados.                                          |
| **PDF**                  | Archivo PDF con endpoints estructurados o semiestructurados.                                        |
| **Archivos de Configuración** | `config.py` con claves API y configuraciones como rutas, chunk size, tokens, etc.                   |
| **Postman y Newman**     | Instala Newman para ejecutar los casos de prueba generados en Postman.                              |
| **OCR**                  | Si el PDF contiene texto en imágenes, se necesita una herramienta de OCR como `tesseract`.          |

---
A continuación, se describe uno a uno los puntos

1. **Bibliotecas Necesarias**:
   ```bash
   pip install openai pinecone-client langchain pandas openpyxl
   ```
  Además de las bibliotecas principales mencionadas, podrías necesitar las siguientes:
  
  - **`langchain` y sus módulos específicos**:
     - Asegúrate de instalar módulos adicionales si el código los requiere:
       ```bash
       pip install langchain
       pip install langchain[all]
       ```
  
  - **Otros módulos relacionados**:
     - **`langchain_community`**: Algunos módulos como `UnstructuredPDFLoader` provienen de la comunidad de LangChain:
       ```bash
       pip install langchain_community
       ```
  
     - **`langchain_chroma`**: Maneja almacenamiento vectorial basado en Chroma:
       ```bash
       pip install chromadb
       ```
  
     - **`langchain_openai`**: Extensiones específicas para OpenAI:
       ```bash
       pip install openai
       ```

2. **Configuración**:
l archivo `config.py` debe contener todas las configuraciones necesarias para el flujo:

```python
# config.py
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
index_name = "your_index_name"
dimension = 1536
metric = "cosine"
chunk_size = 1000
chunk_overlap = 200
token = 256
temperature = 0.2
input_file_path = "ruta/al/archivo/input.pdf"
output_file_path = "ruta/al/archivo/output.json"
output_excel_file_path = "ruta/al/archivo/output.xlsx"
```

3. **Archivo PDF**:
El código requiere un archivo PDF que contenga información estructurada o semiestructurada sobre los endpoints de la API:

- **Contenido esperado en el PDF**:
   - Endpoint URLs, como `/api/login`.
   - Métodos HTTP, como `GET`, `POST`, `PUT`, `DELETE`.
   - Parámetros opcionales, como `username`, `password`.

- **Estructura mínima esperada**:
   - El PDF debe contener el texto en formato legible y no como una imagen. Si el PDF está en formato de imagen, será necesario realizar un paso adicional de OCR (Reconocimiento Óptico de Caracteres).
 
4. **Herramientas de Exportación y Ejecución**:

- **Postman y Newman**:
   - Para ejecutar los casos de prueba generados, necesitas instalar Newman (CLI de Postman):
     ```bash
     npm install -g newman
     ```

- **Bibliotecas para Exportar Archivos Excel**:
   - El código utiliza `openpyxl` para exportar archivos Excel. Instálalo si no está disponible:
     ```bash
     pip install openpyxl
     ```
5. **Manejo de Límite de OpenAI API**:

- **Configura alertas de uso en OpenAI**:
   - Configura notificaciones desde tu cuenta para evitar exceder los límites de uso.

- **Controla el uso de tokens**:
   - Ajusta los parámetros `max_tokens` y `chunk_size` para reducir el consumo de la API.
  
6. **Permisos y Configuración del Sistema**:

- **Acceso a Archivos**:
   - Verifica que el programa tiene permisos de lectura y escritura en las rutas especificadas:
     - PDF de entrada.
     - JSON y Excel de salida.

- **Variable de Entorno**:
   - Verifica que las claves API se leen correctamente desde el archivo `config.py` o como variables de entorno.

### **7. Consideraciones y Mejoras**

- **Escalabilidad**:
  - El código es adecuado para procesar múltiples documentos y almacenarlos en Pinecone, lo que permite búsquedas rápidas basadas en embeddings.
  
- **Límites de la API**:
  - Maneja los límites de la API de OpenAI con reintentos y pausas exponenciales.

- **Flexibilidad**:
  - Configuraciones como `chunk_size`, `temperature` y `max_tokens` son fácilmente ajustables para adaptarse a diferentes necesidades.

Con esta documentación, el flujo y propósito del código quedan claramente explicados para usuarios y desarrolladores.

# Autor
* **Camilo Alejandro Rojas** - *Trabajo y documentación* - [camrojass](https://github.com/camrojass)

# Referencias
* OpenAI. Url: https://python.langchain.com/docs/integrations/llms/openai
* Retrieval-augmented generation (RAG). Url: https://python.langchain.com/docs/use_cases/question_answering/
* Pinecone. Url: https://python.langchain.com/docs/integrations/vectorstores/pinecone
* ChatGPT. Url: https://chat.openai.com/
