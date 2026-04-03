import os
import tempfile  # used for storing pdfs in a directory temporarily
from fastapi import FastAPI, UploadFile, File, Form
from pipeline import build_pipeline, load_pdf_pages
from decouple import config
import uvicorn
import uuid


api = FastAPI()
langgraph_app = build_pipeline()


# function to post pdf contents to the pipeline


@api.post("/api/process")
async def process_claim(
    claim_id: str = Form(default=None),
    file: UploadFile = File(...)
):
    # auto generates claim_id if not provided explicitly
    if not claim_id:
        claim_id = f"CLM_{uuid.uuid4().hex[:8].upper()}"
    # creates temporary directory to store pdf contents and deletes after getting response
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pages = load_pdf_pages(tmp_path)
        # this builds the pipeline and calls the segregator function defined in pipeline
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
    except Exception as e:
        print(f"Exception raised: {e}")
    finally:
        # removes pdf content after execution
        os.remove(tmp_path)

    return result["final_output"]


if __name__ == "__main__":

    uvicorn.run("app:api", host="0.0.0.0", port=8000, reload=True)
