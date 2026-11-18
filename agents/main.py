from fastapi import FastAPI

app = FastAPI(title="agent-orchestrator")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(payload: dict):
    # TODO: graphe LangGraph (Analyste -> Cardiologue RAG -> Auditeur)
    return {"received": payload, "note": "stub"}
