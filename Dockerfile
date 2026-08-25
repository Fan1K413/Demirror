# syntax=docker/dockerfile:1.7

FROM python:3.10-slim-bookworm AS runtime

ARG INSTALL_OPTIONAL_DETECTORS=1
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.lock \
     requirements-p1-c2pa.lock \
     requirements-p2-ai.lock \
     requirements-p3-pixel.lock \
     pyproject.toml \
     ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.lock \
    && if [ "${INSTALL_OPTIONAL_DETECTORS}" = "1" ]; then \
         if [ "${TARGETARCH}" = "arm64" ]; then \
           apt-get update \
           && apt-get install --yes --no-install-recommends build-essential; \
         fi \
         && python -m pip install -r requirements-p1-c2pa.lock -r requirements-p2-ai.lock \
         && python -m pip install \
              --extra-index-url https://download.pytorch.org/whl/cpu \
              torch==2.13.0+cpu torchvision==0.28.0+cpu \
         && python -m pip install -r requirements-p3-pixel.lock \
         && python -m pip install \
              PyWavelets==1.8.0 onnxruntime==1.23.2 bchlib==2.1.3 \
         && if [ "${TARGETARCH}" = "arm64" ]; then \
              apt-get purge --yes --auto-remove build-essential \
              && rm -rf /var/lib/apt/lists/*; \
            fi; \
       fi

COPY README.md ./
COPY src ./src
COPY models ./models
COPY configs ./configs

RUN python -m pip install --no-deps -e . \
    && groupadd --gid 10001 demirror \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/demirror demirror \
    && mkdir -p /app/weights /app/data /app/.demirror_web_jobs /home/demirror/.cache \
    && chown -R demirror:demirror \
         /app/weights /app/data /app/.demirror_web_jobs /home/demirror

USER demirror

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3).read()"

ENTRYPOINT ["image-trust"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765", "--jobs-root", "/app/.demirror_web_jobs", "--allow-non-loopback"]
