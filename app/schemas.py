from pydantic import BaseModel, Field


class SummaryRequest(BaseModel):
    text: str = Field(..., description="Raw text to summarise")
    max_length: int = 200
    min_length: int = 80


class SummaryResponse(BaseModel):
    summary: str
