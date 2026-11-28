FROM python:3.10-slim AS builder
WORKDIR /build
COPY docker/agent-orchestrator/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    HF_HOME=/root/.cache/huggingface
COPY --from=builder /install /usr/local
COPY agents/ /app/agents/
COPY rag_engine/ /app/rag_engine/
EXPOSE 8002
CMD ["uvicorn", "agents.main:app", "--host", "0.0.0.0", "--port", "8002"]
