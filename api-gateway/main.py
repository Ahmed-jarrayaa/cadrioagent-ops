import os
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

app = FastAPI(title="CardioAgent-Ops API")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
AUDIO = os.getenv("AUDIO_ENGINE_URL", "http://audio-engine:8001")
ORCH = os.getenv("ORCHESTRATOR_URL", "http://agent-orchestrator:8002")
MLFLOW = os.getenv("MLFLOW_URL", os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
ALL_FAMILIES = ("svm", "tree", "cnn", "tl", "hybrid")
DEFAULT_INSTRUCTION = (
    "Analyse ce signal cardiaque et génère un rapport détaillé à partir des résultats réels des modèles."
)


def _unit(value):
    """Garde uniquement les scores valides dans [0, 1] pour le contrat Pydantic."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


def _report_payload(workflow: dict, instruction: str) -> dict:
    """Résumé des vraies prédictions transmis à l'agent générateur de rapport."""
    consensus = workflow.get("consensus") or {}
    rows = []
    for row in workflow.get("analysis") or []:
        rows.append({
            "key": row.get("key"),
            "label": row.get("label"),
            "status": row.get("status"),
            "error": row.get("error"),
            "prediction": row.get("prediction"),
            "prob_abnormal_mean": row.get("prob_abnormal_mean"),
            "prob_abnormal_max": row.get("prob_abnormal_max"),
            "prob_abnormal_min": row.get("prob_abnormal_min"),
            "threshold": row.get("threshold"),
            "training_metrics_reference": row.get("training_metrics_reference"),
        })
    return {
        "mode": "report",
        "question": instruction,
        "top_k": 3,
        "workflow": {
            "instruction": instruction,
            "analysis": rows,
            "consensus": consensus,
            "feature_summary": workflow.get("feature_summary"),
            "duration_s": workflow.get("duration_s"),
            "n_segments": workflow.get("n_segments"),
            "sample_rate_used": workflow.get("sample_rate_used"),
            "preprocessing": workflow.get("preprocessing"),
        },
        "prediction": consensus.get("prediction"),
        "prob_abnormal": _unit(consensus.get("mean_prob_abnormal")),
        "threshold": _unit(consensus.get("mean_threshold")),
        "model": "predict_all (5 modèles)",
    }

@app.get("/")
def root():
    return {
        "service": "CardioAgent-Ops API Gateway",
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/health", "/analyze", "/analyze_pcg", "/predict", "/analyze_all",
                      "/analyze_all_report", "/predict_one", "/models", "/mlflow/experiments"],
    }

@app.get("/health")
async def health():
    services = {"api-gateway": "ok"}
    async with httpx.AsyncClient(timeout=5) as client:
        for name, url in (("audio-engine", f"{AUDIO}/health"), ("orchestrator", f"{ORCH}/health"), ("mlflow", f"{MLFLOW}/health")):
            try:
                r = await client.get(url)
                services[name] = {"status": "ok", "http": r.status_code, "body": (r.json() if "application/json" in r.headers.get("content-type","") else r.text[:120])}
            except Exception as e:
                services[name] = {"status": "down", "error": str(e)}
    return {"status": "ok", "services": services}

@app.post("/analyze")
async def analyze(payload: dict):
    """Proxy vers l'orchestrateur multi-agents (question ou résultat audio)."""
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            r = await client.post(f"{ORCH}/analyze", json=payload)
        except Exception as e:
            raise HTTPException(502, f"agent-orchestrator indisponible: {e}")
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        return r.json()


@app.post("/analyze_pcg")
async def analyze_pcg(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Fichier .wav requis")
    data = await file.read()
    async with httpx.AsyncClient(timeout=120) as client:
        r1 = await client.post(f"{AUDIO}/predict",
                               files={"file": (file.filename, data, "audio/wav")})
        r1.raise_for_status()
        r2 = await client.post(f"{ORCH}/analyze", json=r1.json())
        r2.raise_for_status()
    return r2.json()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Proxy direct vers audio-engine /predict (ONNX)."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Fichier .wav requis")
    data = await file.read()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{AUDIO}/predict", files={"file": (file.filename, data, "audio/wav")})
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        return r.json()


@app.post("/analyze_all")
async def analyze_all(file: UploadFile = File(...), families: str | None = None):
    """Applique les modèles TP3 (SVM, arbre de décision, CNN) enregistrés dans MLflow."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Fichier .wav requis")
    data = await file.read()
    params = {"families": families} if families else None
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{AUDIO}/predict_all", params=params,
                              files={"file": (file.filename, data, "audio/wav")})
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        return r.json()


@app.post("/analyze_all_report")
async def analyze_all_report(
    file: UploadFile = File(...),
    instruction: str = Form(DEFAULT_INSTRUCTION),
):
    """Exécute les 5 modèles puis génère un rapport à partir des résultats réels.

    Le workflow audio est toujours appelé avec les 5 familles explicitement : le
    rapport ne peut donc être produit qu'après les 5 prédictions. Les modèles en
    erreur sont conservés dans le rapport, jamais remplacés par une valeur inventée.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Fichier .wav requis")
    instruction = (instruction or "").strip()
    if not instruction:
        raise HTTPException(400, "Une instruction est requise pour générer le rapport.")
    data = await file.read()
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            r1 = await client.post(
                f"{AUDIO}/predict_all",
                params={"families": ",".join(ALL_FAMILIES)},
                files={"file": (file.filename, data, "audio/wav")},
            )
        except Exception as e:
            raise HTTPException(502, f"audio-engine indisponible: {e}")
        if r1.status_code >= 400:
            raise HTTPException(r1.status_code, r1.text)
        workflow = r1.json()

        executed = {row.get("key") for row in (workflow.get("analysis") or []) if row.get("key")}
        missing = [key for key in ALL_FAMILIES if key not in executed]
        if missing:
            raise HTTPException(
                502,
                "Workflow incomplet : modèles non exécutés (" + ", ".join(missing) + "). "
                "Le rapport n'est pas généré.",
            )

        try:
            r2 = await client.post(f"{ORCH}/analyze", json=_report_payload(workflow, instruction))
        except Exception as e:
            raise HTTPException(502, f"agent-orchestrator indisponible: {e}")
        if r2.status_code >= 400:
            raise HTTPException(r2.status_code, r2.text)

    body = r2.json()
    body["instruction"] = instruction
    body["workflow"] = workflow
    return body


@app.post("/predict_one")
async def predict_one(file: UploadFile = File(...), family: str = "svm"):
    """Proxy vers audio-engine /predict_one : un seul algorithme (svm | tree | cnn)."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Fichier .wav requis")
    data = await file.read()
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{AUDIO}/predict_one", params={"family": family},
                              files={"file": (file.filename, data, "audio/wav")})
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        return r.json()


@app.get("/models")
async def models():
    """Modèles TP3 disponibles (seuil, run MLflow, métriques) via audio-engine."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(f"{AUDIO}/models")
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"audio-engine indisponible: {e}")


@app.get("/mlflow/experiments")
async def mlflow_experiments():
    """Proxy léger vers l'API REST MLflow (liste des expériences)."""
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{MLFLOW}/api/2.0/mlflow/experiments/search", params={"max_results": 50})
            if r.status_code >= 400:
                raise HTTPException(r.status_code, r.text)
            return r.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"MLflow indisponible: {e}")