#################### Variables ########################
os.environ["OPENAI_API_KEY"] = "key_openIA"
PINECONE_API_KEY = "key_pinecone"
PINECONE_ENV = "gcp-starter"


# Variables for VectorDataUse
chunk_size=1000
chunk_overlap=200


# Variables for PineconeUse
metric="cosine"
dimension=1536
index_name = "chatgptdbvector"