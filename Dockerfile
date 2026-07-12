# EchoChamber GUI for Cloud Run (or any container host).
#
# Lean by design: installs the gemini/ui/search/env extras only — the heavy
# optional features (pymupdf PDF parsing, chromadb RAG) are omitted; add
# ".[pdf,rag]" below if your deployment needs them.
#
# Auth note: this image has no API keys baked in. On Cloud Run, Gemini works
# through Vertex AI via the runtime service account (set
# GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT).

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY echochamber ./echochamber
COPY cases/example_case ./cases/example_case

RUN pip install --no-cache-dir ".[gemini,ui,search,env,batch]"

ENV ECHOCHAMBER_HOST=0.0.0.0
EXPOSE 8080

CMD ["python", "-m", "echochamber.ui"]
