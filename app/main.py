from fastapi import FastAPI, UploadFile, Form, HTTPException
from app.models import get_summariser
from app.utils import (
    extract_text_from_pdf,
    extract_text_from_url,
    chunk_text_words,
    clean_text,
)
from app.schemas import SummaryRequest, SummaryResponse

app = FastAPI(title="AI Summariser", version="0.1.0")

summariser = get_summariser()  # loads once


def summarise_long_text(text: str, min_length: int, max_length: int) -> str:
    """
    Handles long inputs by chunking -> summarising each -> merging -> final pass.
    """
    text = clean_text(text)
    if not text:
        raise HTTPException(status_code=400, detail="Empty text after cleaning.")
    chunks = list(chunk_text_words(text, max_words=650))
    partials = []
    for ch in chunks:
        out = summariser(
            ch, max_length=max_length, min_length=min_length, do_sample=False
        )[0]["summary_text"]
        partials.append(out)
    # final pass to compress concatenated partials (keeps quality on very long docs)
    stitched = " ".join(partials)
    final = summariser(
        stitched, max_length=max_length, min_length=min_length, do_sample=False
    )[0]["summary_text"]
    return final


@app.post("/summarise", response_model=SummaryResponse)
async def summarise_text(req: SummaryRequest):
    summary = summarise_long_text(req.text, req.min_length, req.max_length)
    return SummaryResponse(summary=summary)


@app.post("/summarise-pdf", response_model=SummaryResponse)
async def summarise_pdf(file: UploadFile):
    if file.content_type not in {
        "application/pdf"
    } and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    text = extract_text_from_pdf(file.file)
    summary = summarise_long_text(text, 80, 200)
    return SummaryResponse(summary=summary)


@app.post("/summarise-url", response_model=SummaryResponse)
async def summarise_url(url: str = Form(...)):
    text = extract_text_from_url(url)
    summary = summarise_long_text(text, 80, 200)
    return SummaryResponse(summary=summary)
