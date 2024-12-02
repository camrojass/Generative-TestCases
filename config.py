#################### Variables ########################
os.environ["OPENAI_API_KEY"] = "key_openIA"
PINECONE_API_KEY = "key_pinecone"
PINECONE_ENV = "gcp-starter"

# Variables for VectorDataUse
chunk_size = 1000
chunk_overlap = 200

#OpenAI
model = "gpt-3.5-turbo"
temperature = 0.2
token = 150

# Variables for PineconeUse
metric = "cosine"
dimension = 20000
index_name = "dbindex"
cloud = "aws"
region = "us-east-1"

# Variables archivos locales
input_file_path = "C:\\Repos\\Proyect\\input\\OpenAPI.pdf"
output_file_path = "C:\\Repos\\Proyect\\output\\postmanCollection.json"
output_excel_file_path = "C:\\Repos\\Proyect\\output\\testCases.xlsx"
