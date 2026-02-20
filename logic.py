from models import vector_store, model
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

chat_history = []

# function to handle document uploads
def preprocess_documents(stream):
    reader = PdfReader(stream)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()
    text_splits = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0,
    )
    all_splits = text_splits.split_text(full_text)
    vector_store.add_texts(all_splits)
    return {"response": "Document is successfully added"}

# function llm can use to query
@tool 
def retrieve_context(query: str):
    '''use this tool to retrieve the context from database for user query and always respond the answer'''
    results = vector_store.similarity_search(query, k=3)
    context = "".join([ans.page_content for ans in results])
    return context
    

# function for agent to answer
def generate(query: str):
    prompt = f"based on {query} always use retrieve_context tool to answer from the documents.check {chat_history} to answer user query before checking documents"
    agent = create_agent(
        model = model,
        tools = [retrieve_context],
        system_prompt = prompt,
    )
    answer = agent.invoke({"messages": [("user", query)]})
    chat_history.append({"user": query})
    chat_history.append({"AI response": answer})
    return answer["messages"][-1].content

