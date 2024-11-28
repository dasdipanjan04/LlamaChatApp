from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status" : "healthy"}

@app.post("/query")
async def process_query(query: str):
    return {"response" : f"Test Query is {query}"}
