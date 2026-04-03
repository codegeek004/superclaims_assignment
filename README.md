# Claim Processing Pipeline

A FastAPI service that processes PDF insurance claims using LangGraph to orchestrate document segregation and multi-agent data extraction.

## Features

- **Page-level routing** - Each page is classified individually before any extraction happens
- **Multi-agent extraction** - Specialist agents for identity, discharge, and billing data
- **Strict isolation** - Agents only see pages assigned to them, enforced at the data level
- **Vision-powered** - Gemini 2.0 Flash processes scanned/image-protected PDFs
- **Modular** - FastAPI and LangGraph layers are fully decoupled

## Tech Stack

- **Web Framework**: FastAPI + Uvicorn
- **AI Orchestration**: LangGraph
- **LLM / Vision**: Gemini 2.0 Flash (via langchain-google-genai)
- **PDF Processing**: PyMuPDF (fitz)
- **Environment**: python-dotenv

## Deployment

This service is deployed on two platforms:

### Render
The API is hosted at:
```
https://superclaims-assignment-nexv.onrender.com/api/process
```

### GCP (Google Cloud Platform)
The API is also deployed on GCP with Nginx as a reverse proxy, Gunicorn as the WSGI server, and an SSL certificate issued by Let's Encrypt:
```
https://codemos-services.co.in/
```


## Getting Started

### Prerequisites

- Python 3.10+
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com)
- Use `gemini-2.0-flash` specifically — both Flash and Pro are capped at 5 RPM on the free tier, meaning almost any real PDF will hit rate limits. A paid plan is recommended for anything beyond testing.

> **API Rate Limits:** Most LLM APIs (including Gemini free tier) cap requests at 5-10 RPM. This pipeline makes `num_pages + 3` LLM calls per document. For multi-page PDFs on a free tier key, you may hit rate limits. See the [Rate Limits](#rate-limits) section below.

### Installation

##### Clone the repository
```bash
git clone https://github.com/codegeek004/superclaims_assignment.git
cd superclaims_assignment
```

##### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate 
```

##### Install dependencies
```bash
pip install -r requirements.txt
```

##### Add your API key in .env
```bash
GOOGLE_API_KEY=your_key_here
```

##### Start the server
```bash
python app.py
```

Server runs at `http://127.0.0.1:8000`

## Architecture & System Design

### Project Structure

```
project/
├── app.py         --> FastAPI api
├── pipeline.py     --> LangGraph pipeline, agents, and state
├── .env            --> API key 
└── requirements.txt
```

`app.py` doesn't know anything about LangGraph or Gemini. It receives the file, calls `invoke()`, and returns the result. `pipeline.py` can be tested completely independently without running the server.

### How it works

1. Client sends a `POST /api/process` request with the PDF and an optional `claim_id`
2. FastAPI saves the uploaded PDF to a temporary file on disk
3. fitz opens the temp file and converts every page into a base64-encoded PNG at 150 DPI
4. The LangGraph pipeline starts with the `ClaimState` object initialized, this is the shared state that every node reads from and writes to
5. The **Segregator** runs first. It sends each page image to Gemini one at a time and asks it to classify the page into one of 9 document types
6. After classifying all pages, the Segregator splits the page images into three separate lists, `id_pages`, `discharge_pages`, and `bill_pages` based on the classifications
7. The **ID Agent**, **Discharge Agent**, and **Bill Agent** run next. Each one receives only its own list of pages, sends them to Gemini with an extraction prompt, and writes the result back to state
8. The **Aggregator** runs last. No LLM call here, it just merges the three agent outputs into a single JSON object
9. FastAPI reads `final_output` from the state and returns it as the HTTP response
10. The temp file is deleted regardless of success or failure

### Why fitz?

LangChain's `HumanMessage` content blocks only support `text` and `image_url`, there's no way to pass raw PDF bytes through it to Gemini. fitz renders each page as a PNG at 150 DPI, base64-encodes it, and sends it as an image. Since the target PDF is image-protected (scanned pages, no text layer), this is the right approach regardless.

### Page Routing

The key design decision is that page routing is enforced at the data level, not via prompting. The segregator splits page images into three separate lists. Each agent only receives its own list, it cannot see pages outside its bucket.

```
Page 0 --> "claim_forms"        classified, no agent processes it
Page 1 --> "identity_document"  id_pages[]        --> ID Agent only
Page 2 --> "discharge_summary"  discharge_pages[] --> Discharge Agent only
Page 3 --> "itemized_bill"      bill_pages[]      --> Bill Agent only
Page 4 --> "prescription"       classified, no agent processes it
```

### LangGraph Nodes

**Segregator** - classifies every page into one of 9 types using Gemini vision. Makes 1 LLM call per page. Splits results into three page buckets.

Supported types: `claim_forms` `cheque_or_bank_details` `identity_document` `itemized_bill` `discharge_summary` `prescription` `investigation_report` `cash_receipt` `other`

**ID Agent** - receives only identity document pages. Extracts `patient_name`, `dob`, `id_number`, `policy_number`, `address`, `phone`, `email`. 1 LLM call.

**Discharge Agent** - receives only discharge summary pages. Extracts `diagnosis`, `admit_date`, `discharge_date`, `physician_name`, `hospital_name`, `treatment_summary`, `follow_up_instructions`. 1 LLM call.

**Bill Agent** - receives only itemized bill pages. Extracts all line items with costs and totals. 1 LLM call.

**Aggregator** - no LLM call. Pure Python. Merges all agent outputs into the final JSON.

### LLM Calls Per Request

```
segregator  -->  1 per page
id_agent    -->  1
discharge   -->  1
bill_agent  -->  1
aggregator  -->  0

total  =  num_pages + 3
```

## Rate Limits

> This section is important before running the pipeline in production or on large documents.

Most LLM providers cap free-tier usage at **5-10 requests per minute (RPM)**. This pipeline makes `num_pages + 3` LLM calls per document, so a 10-page PDF requires 13 sequential calls.

**Gemini free tier limits:**
- `gemini-2.0-flash`: 5 RPM


With only 5 RPM available, even a 2-page PDF requires 5 LLM calls, which immediately saturates the limit.

**What this means in practice:**
- Almost any real world PDF will hit the rate limit mid-pipeline on a free tier key
- For production workloads or documents with more than 1–2 pages, a **paid Gemini API plan** is required

To check current Gemini API limits: [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)

## API

### POST /api/process

| Field | Type | Required |
|-------|------|----------|
| file | PDF | Yes |
| claim_id | string | No(autogenerated if missing) |

`claim_id` is autogenerated as `CLM_XXXXXXXX` if not provided.

**Response**

```json
{
  "claim_id": "CLM_A3F9B21C",
  "total_pages": 8,
  "page_classifications": {
    "0": "identity_document",
    "1": "discharge_summary",
    "2": "itemized_bill",
    "3": "claim_forms"
  },
  "extracted_data": {
    "identity": {
      "patient_name": "John Doe",
      "dob": "1990-01-15",
      "id_number": "ID123456",
      "policy_number": "POL789012",
      "address": "123 Main St, Mumbai",
      "phone": "+91 9876543210",
      "email": "john@example.com"
    },
    "discharge_summary": {
      "diagnosis": "Appendicitis",
      "admit_date": "2024-01-10",
      "discharge_date": "2024-01-15",
      "physician_name": "Dr. Sharma",
      "hospital_name": "City Hospital",
      "treatment_summary": "Laparoscopic appendectomy performed",
      "follow_up_instructions": "Follow up in 2 weeks"
    },
    "itemized_bill": {
      "items": [
        {"name": "Surgery", "quantity": 1, "unit_cost": 35000, "total_cost": 35000},
        {"name": "Room charges", "quantity": 5, "unit_cost": 2000, "total_cost": 10000},
        {"name": "Medicines", "quantity": 1, "unit_cost": 3500, "total_cost": 3500}
      ],
      "subtotal": 48500,
      "taxes": 2425,
      "total_amount": 50925,
      "currency": "INR"
    }
  }
}
```

## Usage

### Local

1. Start the server with `uvicorn main:api --reload`
2. Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI
3. Upload your PDF and optionally provide a `claim_id`
4. Hit Execute and get the extracted JSON back

```bash
# curl --> local
curl -X POST "http://127.0.0.1:8000/api/process" \
  -F "file=@claim.pdf"

# with custom claim id - local
curl -X POST "http://127.0.0.1:8000/api/process" \
  -F "claim_id=CLM001" \
  -F "file=@claim.pdf"
```

### Render (Cloud)

```bash
# curl --> Render
curl -X POST "https://superclaims-assignment-nexv.onrender.com/api/process" \
  -F "file=@claim.pdf"

# with custom claim id --> Render
curl -X POST "https://superclaims-assignment-nexv.onrender.com/api/process" \
  -F "claim_id=CLM001" \
  -F "file=@claim.pdf"
```

```python
import requests

with open("claim.pdf", "rb") as f:
    r = requests.post(
        "https://superclaims-assignment-nexv.onrender.com/api/process",
        files={"file": f}
    )

print(r.json())
```

### GCP / codemos-services.co.in (Cloud)

```bash
# curl --> GCP
curl -X POST "https://codemos-services.co.in/api/process" \
  -F "file=@claim.pdf"

# with custom claim id --> GCP
curl -X POST "https://codemos-services.co.in/api/process" \
  -F "claim_id=CLM001" \
  -F "file=@claim.pdf"
```

```python
import requests

with open("claim.pdf", "rb") as f:
    r = requests.post(
        "https://codemos-services.co.in/api/process",
        files={"file": f}
    )

print(r.json())
```

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Google Gemini](https://aistudio.google.com)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
