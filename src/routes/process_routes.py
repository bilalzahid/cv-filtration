from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.services.pipeline_service import run_complete_pipeline

router = APIRouter(tags=["Process"])


@router.post("/process")
async def process_endpoint(
        jd: UploadFile = File(...),
        cvs: List[UploadFile] = File(...)
):
    """
    jd  : single JD file (pdf/docx)
    cvs : multiple CV files (pdf/docx)
    """
    try:
        response = await run_complete_pipeline(jd, cvs)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
