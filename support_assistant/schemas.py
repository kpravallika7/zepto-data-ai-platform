from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)