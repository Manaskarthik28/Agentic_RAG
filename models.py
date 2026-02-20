from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os 


load_dotenv()
# create a vector store and embeddings
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv('index'))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key = os.getenv('GOOGLE_API_KEY'), output_dimensionality=1024)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = os.getenv('GOOGLE_API_KEY'))
