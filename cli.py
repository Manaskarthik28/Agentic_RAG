from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from logic import preprocess_documents, generate
from pydantic import BaseModel
import os 
import io
import json

app = FastAPI()

class QueryResponse(BaseModel):
    query: str

@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    file = await file.read()
    stream = io.BytesIO(file)
    doc_result = preprocess_documents(stream)
    return {"doc_result": "Document Uploaded Successfully"}

@app.post('/response')
async def answer(data: QueryResponse):
    llm_result = generate(data.query)
    with open("logfile.txt", "a") as f:
        f.write(json.dumps({"User query": data.query}, indent=4))
        f.write(json.dumps({"AI Response": llm_result}, indent=4))
    return {"llm_result": llm_result[0]["text"]}

@app.get('/', response_class = FileResponse)
async def greet():
    return FileResponse('index.html')