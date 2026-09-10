from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graph import support_graph
from schemas import SupportResponse


app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
)


class SupportRequest(BaseModel):
    query: str = Field(..., min_length=1)


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=SupportResponse)
def support(request: SupportRequest):
    try:
        result = support_graph.invoke({
            "query": request.query
        })

        return SupportResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result["confidence"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )