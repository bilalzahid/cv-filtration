from fastapi import FastAPI, UploadFile, File
from typing import List
from src.services.process_service import process_request

app = FastAPI(title="CV Ranking Engine")

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/process")
async def process(
        cvs: List[UploadFile] = File(...),
        jd: UploadFile = File(...)
):
    """
    Accepts multiple CV files + one JD file
    Parses, embeds, ranks, returns result immediately
    """
    return await process_request(cvs, jd)
