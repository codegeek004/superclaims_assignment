import os
import json
import base64
import fitz
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from decouple import config
import os

key = config('GEMINI_API_KEY')
# StateGraph class


class ClaimState(TypedDict):
    claim_id: str
    pdf_path: str
    pages_base64: list[str]
    page_classifications: dict
    id_pages: list[str]
    discharge_pages: list[str]
    bill_pages: list[str]
    id_data: dict | None
    discharge_data: dict | None
    bill_data: dict | None
    final_output: dict | None


# loading pdf to pages using fitz because it loads each page instead of full pdf


def load_pdf_pages(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    pages_base64 = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        pages_base64.append(encoded)
    return pages_base64


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except Exception as e:
        return {"raw": text}


# calling gemini llm api with creds from .env
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=key,
)
# the segregator classifies the pages into 9 doc types and routes them to multiple
# agents as per the classification


def segregator(state: ClaimState) -> ClaimState:
    pages = state["pages_base64"]
    classifications = {}

    for i, page_b64 in enumerate(pages):
        message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{page_b64}"}
            },
            {
                "type": "text",
                "text": """You are a medical claims document classifier.
                Classify this page into exactly one of these categories:
                - claim_forms
                - cheque_or_bank_details
                - identity_document
                - itemized_bill
                - discharge_summary
                - prescription
                - investigation_report
                - cash_receipt
                - other

                Reply with just the category name, nothing else. No explanation."""
            }
        ])

        response = llm.invoke([message])
        label = response.content.strip().lower().replace(" ", "_")
        classifications[i] = label

    id_pages, discharge_pages, bill_pages = [], [], []
    for i, label in classifications.items():
        if label == "identity_document":
            id_pages.append(pages[i])
        elif label == "discharge_summary":
            discharge_pages.append(pages[i])
        elif label == "itemized_bill":
            bill_pages.append(pages[i])


    return {
        "page_classifications": classifications,
        "id_pages": id_pages,
        "discharge_pages": discharge_pages,
        "bill_pages": bill_pages
    }


def id_agent(state: ClaimState) -> ClaimState:
    if not state["id_pages"]:
        return {"id_data": {}}

    content = []
    for page_b64 in state["id_pages"]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{page_b64}"}
        })
    content.append({
        "type": "text",
        "text": """Extract identity information from these document pages.
        Return a JSON object with these fields:
        {
            "patient_name": "",
            "dob": "",
            "id_number": "",
            "policy_number": "",
            "address": "",
            "phone": "",
            "email": ""
        }
        If a field is not found, use null.
        Return only the JSON object, no extra text or markdown."""
    })

    response = llm.invoke([HumanMessage(content=content)])
    data = parse_json_response(response.content)
    return {"id_data": data}


def discharge_agent(state: ClaimState) -> ClaimState:
    if not state["discharge_pages"]:
        return {"discharge_data": {}}

    content = []
    for page_b64 in state["discharge_pages"]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{page_b64}"}
        })
    content.append({
        "type": "text",
        "text": """Extract discharge summary information from these pages.
        Return a JSON object with these fields:
        {
            "diagnosis": "",
            "admit_date": "",
            "discharge_date": "",
            "physician_name": "",
            "hospital_name": "",
            "treatment_summary": "",
            "follow_up_instructions": ""
        }
        If a field is not found, use null.
        Return only the JSON object, no extra text or markdown."""
    })

    response = llm.invoke([HumanMessage(content=content)])
    data = parse_json_response(response.content)
    return {"discharge_data": data}


def bill_agent(state: ClaimState) -> ClaimState:
    if not state["bill_pages"]:
        return {"bill_data": {}}

    content = []
    for page_b64 in state["bill_pages"]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{page_b64}"}
        })
    content.append({
        "type": "text",
        "text": """Extract all billing information from these pages.
        Return a JSON object with these fields:
        {
            "items": [
                {"name": "", "quantity": 1, "unit_cost": 0.0, "total_cost": 0.0}
            ],
            "subtotal": 0.0,
            "taxes": 0.0,
            "total_amount": 0.0,
            "currency": "INR"
        }
        If a field is not found, use null.
        Return only the JSON object, no extra text or markdown."""
    })

    response = llm.invoke([HumanMessage(content=content)])
    data = parse_json_response(response.content)
    return {"bill_data": data}


def aggregator(state: ClaimState) -> ClaimState:
    final = {
        "claim_id": state["claim_id"],
        "total_pages": len(state["pages_base64"]),
        "page_classifications": state["page_classifications"],
        "extracted_data": {
            "identity": state.get("id_data") or {},
            "discharge_summary": state.get("discharge_data") or {},
            "itemized_bill": state.get("bill_data") or {}
        }
    }
    return {"final_output": final}



def build_pipeline():
    builder = StateGraph(ClaimState)
    builder.add_node("segregator", segregator)
    builder.add_node("id_agent", id_agent)
    builder.add_node("discharge_agent", discharge_agent)
    builder.add_node("bill_agent", bill_agent)
    builder.add_node("aggregator", aggregator)

    builder.add_edge(START, "segregator")
    builder.add_edge("segregator", "id_agent")
    builder.add_edge("segregator", "discharge_agent")
    builder.add_edge("segregator", "bill_agent")
    builder.add_edge("id_agent", "aggregator")
    builder.add_edge("discharge_agent", "aggregator")
    builder.add_edge("bill_agent", "aggregator")
    builder.add_edge("aggregator", END)

    return builder.compile()
