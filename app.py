import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from pipeline import build_pipeline, load_pdf_pages  
from decouple import config
import uvicorn
import uuid


api = FastAPI()
langgraph_app = build_pipeline()  


@api.post("/api/process")
async def process_claim(
    claim_id: str = Form(default=None),
    file: UploadFile = File(...)
):
    if not claim_id:
        claim_id = f"CLM_{uuid.uuid4().hex[:8].upper()}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pages = load_pdf_pages(tmp_path)
        result = langgraph_app.invoke({
            "claim_id": claim_id,
            "pdf_path": tmp_path,
            "pages_base64": pages,
            "page_classifications": {},
            "id_pages": [],
            "discharge_pages": [],
            "bill_pages": [],
            "id_data": None,
            "discharge_data": None,
            "bill_data": None,
            "final_output": None
        })
    finally:
        os.remove(tmp_path)

    return result["final_output"]


if __name__ == "__main__":

    uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)
