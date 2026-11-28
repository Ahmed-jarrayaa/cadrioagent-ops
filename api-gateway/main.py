import asyncio
import json
import os
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

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
SERVICE_ORDER = (
    "api-gateway", "audio-engine", "mlflow-server",
    "agent-orchestrator", "vector-db", "llm-server",
)
SERVICE_STEPS = {
    "api-gateway": "Réception du WAV et de l'instruction",
    "audio-engine": "Prétraitement et 5 prédictions",
    "mlflow-server": "Métadonnées de référence des modèles",
    "agent-orchestrator": "Graphe Analyste → Cardiologue → Auditeur",
    "vector-db": "Passages RAG (cardio_docs)",
    "llm-server": "Confirmation bornée du rapport",
}
REQUIRED_SERVICES = frozenset({"api-gateway", "audio-engine", "agent-orchestrator"})
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class WorkflowTracker:
    """Trace le workflow dans MLflow via son API REST (aucun SDK requis).

    Chaque étape réellement traversée est enregistrée comme tag et, lorsqu'une
    durée est mesurée, comme métrique dans un run ``cardio-workflow``. Une panne
    de MLflow est signalée dans le flux mais n'interrompt pas le workflow.
    """

    def __init__(self, client: Any, base: str):
        self.client = client
        self.base = base.rstrip("/")
        self.run_id: str | None = None
        self.available = True
        self.steps: list[dict[str, Any]] = []

    async def start(self, instruction: str) -> bool:
        try:
            response = await self.client.get(
                f"{self.base}/api/2.0/mlflow/experiments/search",
                params={"max_results": 100},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}")
            experiments = (response.json() or {}).get("experiments") or []
            experiment = next(
                (item for item in experiments if item.get("name") == "cardio-workflow"), None
            )
            experiment_id = experiment.get("experiment_id") if experiment else None
            if not experiment_id:
                created = await self.client.post(
                    f"{self.base}/api/2.0/mlflow/experiments/create",
                    json={"name": "cardio-workflow"},
                )
                experiment_id = (created.json() or {}).get("experiment_id")
            if not experiment_id:
                raise RuntimeError("experiment_id indisponible")
            created = await self.client.post(
                f"{self.base}/api/2.0/mlflow/runs/create",
                json={
                    "experiment_id": experiment_id,
                    "run_name": "workflow-rapport",
                    "start_time": int(time.time() * 1000),
                    "tags": [
                        {"key": "workflow", "value": "analyze_all_report_stream"},
                        {"key": "instruction", "value": str(instruction)[:500]},
                    ],
                },
            )
            run = (created.json() or {}).get("run") or {}
            self.run_id = (run.get("info") or {}).get("run_id")
            if not self.run_id:
                raise RuntimeError("run_id indisponible")
        except Exception:
            self.available = False
            self.run_id = None
        return self.available

    async def _tag(self, key: str, value: str) -> None:
        if self.run_id:
            await self.client.post(
                f"{self.base}/api/2.0/mlflow/runs/set-tag",
                json={"run_id": self.run_id, "key": key, "value": str(value)[:500]},
            )

    async def step(
        self, service: str, status: str, detail: str = "", duration_ms: float | None = None
    ) -> None:
        self.steps.append(
            {"service": service, "status": status, "detail": detail, "duration_ms": duration_ms}
        )
        if not self.run_id or not self.available:
            return
        try:
            await self._tag(f"step.{service}", status)
            await self._tag(f"detail.{service}", detail)
            if duration_ms is not None:
                await self.client.post(
                    f"{self.base}/api/2.0/mlflow/runs/log-metric",
                    json={
                        "run_id": self.run_id,
                        "key": f"step.{service}.duration_ms",
                        "value": float(duration_ms),
                        "timestamp": int(time.time() * 1000),
                        "step": len(self.steps),
                    },
                )
        except Exception:
            self.available = False

    async def summary(self, workflow: dict, report: dict) -> None:
        if not self.run_id or not self.available:
            return
        consensus = workflow.get("consensus") or {}
        try:
            for key, value in {
                "prediction": consensus.get("prediction") or "indéterminé",
                "models_ok": consensus.get("models_ok", 0),
                "models_total": consensus.get("models_total", 0),
                "report_status": report.get("status", "inconnu"),
            }.items():
                await self._tag(f"result.{key}", value)
        except Exception:
            self.available = False

    async def finish(self, status: str) -> None:
        if not self.run_id or not self.available:
            return
        try:
            await self.client.post(
                f"{self.base}/api/2.0/mlflow/runs/update",
                json={
                    "run_id": self.run_id,
                    "status": "FINISHED" if status == "terminated" else "FAILED",
                    "end_time": int(time.time() * 1000),
                },
            )
        except Exception:
            self.available = False


def _workflow_trace(workflow: dict, report: dict, audio_ms: float, orch_ms: float) -> list[dict]:
    """Trace réelle des six services traversés pendant la génération du rapport."""
    analysis = workflow.get("analysis") or []
    consensus = workflow.get("consensus") or {}
    ok_count = sum(1 for row in analysis if row.get("status") == "ok")
    metrics_count = sum(1 for row in analysis if row.get("training_metrics_reference"))
    sources = report.get("sources") or []
    audit = report.get("audit") or {}
    warnings = audit.get("warnings") or []

    blocked = (
        report.get("intent") in {"out_of_scope", "diagnosis", "prescription", "emergency"}
        or report.get("status") == "refused"
    )
    llm_status = "non_sollicite"
    llm_detail = "aucun appel LLM"
    if blocked and "llm_out_of_context" not in warnings:
        llm_status, llm_detail = "bloque", "bloquage déterministe avant LLM"
    elif "llm_unavailable" in " ".join(audit.get("issues") or []):
        llm_status, llm_detail = "indisponible", "LLM injoignable, rapport de repli"
    elif "llm_confirmation_rejected" in warnings:
        llm_status, llm_detail = "degrade", "confirmation LLM rejetée"
    elif "llm_out_of_context" in warnings or report.get("status") == "refused":
        llm_status, llm_detail = "bloque", "instruction hors contexte"
    elif report.get("status") == "ok":
        llm_status, llm_detail = "ok", "confirmation structurée validée"

    return [
        {"service": "api-gateway", "step": SERVICE_STEPS["api-gateway"],
         "status": "ok", "duration_ms": None,
         "detail": f"instruction reçue ({len(str(report.get('instruction') or ''))} caractères)"},
        {"service": "audio-engine", "step": SERVICE_STEPS["audio-engine"],
         "status": "ok" if ok_count == len(analysis) and analysis else "degrade",
         "duration_ms": round(audio_ms, 1),
         "detail": f"{ok_count}/{len(analysis)} modèles exécutés · vote {consensus.get('prediction', 'indéterminé')}"},
        {"service": "mlflow-server", "step": SERVICE_STEPS["mlflow-server"],
         "status": "ok" if metrics_count else "sans_donnee",
         "duration_ms": None,
         "detail": f"{metrics_count} modèle(s) avec métriques de référence via audio-engine"},
        {"service": "agent-orchestrator", "step": SERVICE_STEPS["agent-orchestrator"],
         "status": "ok" if report.get("status") == "ok" else str(report.get("status", "inconnu")),
         "duration_ms": round(orch_ms, 1),
         "detail": f"rapport {report.get('status', 'inconnu')} · audit {'OK' if audit.get('passed') else 'avec réserves'}"},
        {"service": "vector-db", "step": SERVICE_STEPS["vector-db"],
         "status": "bloque" if blocked else ("ok" if sources else "indisponible"),
         "duration_ms": None,
         "detail": "instruction bloquée avant recherche" if blocked
                   else f"{len(sources)} passage(s) utilisé(s)"},
        {"service": "llm-server", "step": SERVICE_STEPS["llm-server"],
         "status": llm_status, "duration_ms": None, "detail": llm_detail},
    ]


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
        "endpoints": ["/health", "/workflow/services", "/analyze", "/analyze_pcg", "/predict",
                      "/analyze_all", "/analyze_all_report", "/analyze_all_report/stream",
                      "/predict_one", "/models", "/mlflow/experiments"],
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


@app.get("/workflow/services")
async def workflow_services():
    """État des six services traversés par le workflow de rapport."""
    services = {name: {"status": "inconnu", "step": SERVICE_STEPS[name]}
                for name in SERVICE_ORDER}
    services["api-gateway"] = {"status": "ok", "step": SERVICE_STEPS["api-gateway"]}

    async with httpx.AsyncClient(timeout=5) as client:
        audio, orch, mlflow = await asyncio.gather(
            client.get(f"{AUDIO}/health"),
            client.get(f"{ORCH}/health"),
            client.get(f"{MLFLOW}/health"),
            return_exceptions=True,
        )

    def _simple(result):
        if isinstance(result, Exception):
            return {"status": "indisponible", "error": str(result)}
        return {"status": "ok" if result.status_code < 400 else "degrade",
                "http": result.status_code}

    services["audio-engine"] = _simple(audio)
    services["mlflow-server"] = _simple(mlflow)

    if isinstance(orch, Exception):
        services["agent-orchestrator"] = {"status": "indisponible", "error": str(orch)}
        services["vector-db"] = {"status": "indisponible", "error": "orchestrateur indisponible"}
        services["llm-server"] = {"status": "indisponible", "error": "orchestrateur indisponible"}
    else:
        services["agent-orchestrator"] = _simple(orch)
        nested = {}
        try:
            nested = orch.json().get("services") or {}
        except Exception:
            nested = {}
        qdrant = nested.get("qdrant") or {}
        llm = nested.get("llm") or {}
        services["vector-db"] = {
            "status": qdrant.get("status", "inconnu"),
            "points": qdrant.get("points"),
            "collection": qdrant.get("collection"),
        }
        services["llm-server"] = {
            "status": llm.get("status", "inconnu"),
            "model": llm.get("model"),
        }

    return {"status": "ok", "order": list(SERVICE_ORDER), "services": services}

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
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        audio_started = time.perf_counter()
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
        audio_ms = (time.perf_counter() - audio_started) * 1000

        executed = {row.get("key") for row in (workflow.get("analysis") or []) if row.get("key")}
        missing = [key for key in ALL_FAMILIES if key not in executed]
        if missing:
            raise HTTPException(
                502,
                "Workflow incomplet : modèles non exécutés (" + ", ".join(missing) + "). "
                "Le rapport n'est pas généré.",
            )

        orch_started = time.perf_counter()
        try:
            r2 = await client.post(f"{ORCH}/analyze", json=_report_payload(workflow, instruction))
        except Exception as e:
            raise HTTPException(502, f"agent-orchestrator indisponible: {e}")
        if r2.status_code >= 400:
            raise HTTPException(r2.status_code, r2.text)
        orch_ms = (time.perf_counter() - orch_started) * 1000

    body = r2.json()
    body["instruction"] = instruction
    body["workflow"] = workflow
    body["workflow_trace"] = _workflow_trace(workflow, body, audio_ms, orch_ms)
    body["gateway_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return body


def _sse_frame(event: str, payload: Any) -> str:
    """Formatte un événement Server-Sent Events."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_report(filename: str, data: bytes, instruction: str) -> AsyncIterator[str]:
    """Exécute le workflow séquentiellement et émet chaque étape réelle.

    Les événements ``step`` sont produits avant l'appel (``running``) puis après
    sa réponse (``ok`` / ``erreur`` / état dégradé constaté). Aucune progression
    n'est simulée : les étapes internes de l'orchestrateur sont relayées depuis
    son propre flux SSE.
    """
    out: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    started = time.perf_counter()

    async with httpx.AsyncClient(timeout=300) as client:
        tracker = WorkflowTracker(client, MLFLOW)

        async def emit(event: str, payload: Any) -> None:
            await out.put({"event": event, "data": payload})

        async def mark(service: str, status: str, detail: str = "",
                       duration_ms: float | None = None) -> None:
            step: dict[str, Any] = {
                "service": service, "status": status, "detail": str(detail)[:200],
            }
            if duration_ms is not None:
                step["duration_ms"] = round(float(duration_ms), 1)
            await emit("step", step)
            await tracker.step(service, status, str(detail), duration_ms)

        async def stop(service: str, detail: str) -> str:
            await mark(service, "erreur", detail)
            await emit("error", {"service": service, "detail": str(detail)[:300],
                                 "indispensable": service in REQUIRED_SERVICES})
            return "error"

        async def worker() -> str:
            final_status = "error"
            try:
                await mark("api-gateway", "running", "Réception du WAV et de l'instruction")
                await mark("api-gateway", "ok",
                           f"instruction reçue ({len(instruction)} caractères)")

                await mark("audio-engine", "running", "Prétraitement et 5 prédictions")
                audio_started = time.perf_counter()
                try:
                    r1 = await client.post(
                        f"{AUDIO}/predict_all",
                        params={"families": ",".join(ALL_FAMILIES)},
                        files={"file": (filename, data, "audio/wav")},
                    )
                except Exception as error:
                    return await stop("audio-engine", f"audio-engine injoignable : {error}")
                if r1.status_code >= 400:
                    return await stop("audio-engine", f"HTTP {r1.status_code}")
                workflow = r1.json()
                audio_ms = (time.perf_counter() - audio_started) * 1000
                analysis = workflow.get("analysis") or []
                executed = {row.get("key") for row in analysis if row.get("key")}
                missing = [key for key in ALL_FAMILIES if key not in executed]
                if missing:
                    return await stop(
                        "audio-engine", "modèles non exécutés : " + ", ".join(missing)
                    )
                ok_count = sum(1 for row in analysis if row.get("status") == "ok")
                consensus = workflow.get("consensus") or {}
                await mark(
                    "audio-engine", "ok" if ok_count == len(analysis) else "degrade",
                    f"{ok_count}/{len(analysis)} modèles · vote "
                    f"{consensus.get('prediction', 'indéterminé')}",
                    audio_ms,
                )

                await tracker.start(instruction)
                await mark("mlflow-server", "running", "Tracking des étapes du workflow")
                if tracker.available:
                    for previous in list(tracker.steps[:-1]):
                        await tracker.step(previous["service"], previous["status"],
                                           previous["detail"], previous["duration_ms"])
                    await mark("mlflow-server", "ok", "run cardio-workflow ouvert")
                else:
                    await mark("mlflow-server", "erreur", "MLflow injoignable, tracking ignoré")

                await mark("agent-orchestrator", "running",
                           "Graphe Analyste → Cardiologue → Auditeur")
                orch_started = time.perf_counter()
                payload = _report_payload(workflow, instruction)
                result: dict[str, Any] = {"body": None, "error": None, "fallback": False}

                async def forward(block: list[str]) -> None:
                    event = "message"
                    chunks: list[str] = []
                    for line in block:
                        if line.startswith("event:"):
                            event = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            chunks.append(line.split(":", 1)[1].strip())
                    raw = "\n".join(chunks)
                    try:
                        data_event = json.loads(raw) if raw else {}
                    except ValueError:
                        return
                    if event == "step" and isinstance(data_event, dict):
                        await mark(
                            str(data_event.get("service") or "agent-orchestrator"),
                            str(data_event.get("status") or "running"),
                            str(data_event.get("detail") or ""),
                        )
                    elif event == "result" and isinstance(data_event, dict):
                        result["body"] = data_event
                    elif event == "error" and isinstance(data_event, dict):
                        result["error"] = str(data_event.get("detail") or "erreur orchestrateur")

                try:
                    async with client.stream(
                        "POST", f"{ORCH}/analyze/stream", json=payload
                    ) as r2:
                        if r2.status_code in (404, 405):
                            result["fallback"] = True
                        elif r2.status_code >= 400:
                            return await stop("agent-orchestrator", f"HTTP {r2.status_code}")
                        else:
                            buffer: list[str] = []
                            async for line in r2.aiter_lines():
                                if line.strip() == "":
                                    if buffer:
                                        await forward(buffer)
                                        buffer = []
                                else:
                                    buffer.append(line)
                            if buffer:
                                await forward(buffer)
                except Exception as error:
                    return await stop("agent-orchestrator",
                                      f"flux orchestrateur indisponible : {type(error).__name__}: {error}")

                if result["error"]:
                    return await stop("agent-orchestrator", result["error"])

                if result["fallback"] or result["body"] is None:
                    try:
                        r2 = await client.post(f"{ORCH}/analyze", json=payload)
                    except Exception as error:
                        return await stop("agent-orchestrator", f"orchestrateur injoignable : {error}")
                    if r2.status_code >= 400:
                        return await stop("agent-orchestrator", f"HTTP {r2.status_code}")
                    result["body"] = r2.json()
                    orch_ms = (time.perf_counter() - orch_started) * 1000
                    for row in _workflow_trace(workflow, result["body"], audio_ms, orch_ms)[3:]:
                        await mark(row["service"], row["status"], row["detail"],
                                   row["duration_ms"])

                orch_ms = (time.perf_counter() - orch_started) * 1000
                body = dict(result["body"])
                body["instruction"] = instruction
                body["workflow"] = workflow
                body["workflow_trace"] = _workflow_trace(workflow, body, audio_ms, orch_ms)
                body["gateway_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                await tracker.summary(workflow, body)
                await emit("result", body)
                await emit("status", {"status": "terminated",
                                      "detail": "Workflow terminé", "services": list(SERVICE_ORDER)})
                final_status = "terminated"
                return final_status
            except Exception as error:
                await emit("error", {"service": "api-gateway",
                                     "detail": f"{type(error).__name__}: {error}",
                                     "indispensable": True})
                return "error"
            finally:
                await tracker.finish(final_status)
                await out.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await out.get()
                if item is None:
                    break
                yield _sse_frame(item["event"], item["data"])
        finally:
            if not task.done():
                task.cancel()


@app.post("/analyze_all_report/stream")
async def analyze_all_report_stream(
    file: UploadFile = File(...),
    instruction: str = Form(DEFAULT_INSTRUCTION),
):
    """Workflow 5 modèles + rapport, diffusé étape par étape en SSE."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Fichier .wav requis")
    instruction = (instruction or "").strip()
    if not instruction:
        raise HTTPException(400, "Une instruction est requise pour générer le rapport.")
    data = await file.read()
    return StreamingResponse(
        _stream_report(file.filename, data, instruction),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


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