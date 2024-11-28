from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.pipeline import preprocess, inference, postprocess
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    return {"status" : "healthy"}

@app.post("/query")
async def process_query(request: QueryRequest):
    preprocessed_query = preprocess(request.query)
    model_response = inference(preprocessed_query)
    final_response = postprocess(model_response)
    return {"response" : final_response}

