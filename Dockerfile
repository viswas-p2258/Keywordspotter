FROM public.ecr.aws/awslambda-web-adapter:0.9.1 AS lambda-adapter

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HF_HOME=/tmp/huggingface \
    AWS_LWA_PORT=5000 \
    AWS_LWA_READINESS_CHECK_PATH=/health

WORKDIR /app

COPY --from=lambda-adapter /lambda-adapter /opt/extensions/lambda-adapter

RUN apt-get update \
    && apt-get install --no-install-recommends -y libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app app.py ./
COPY --chown=app:app config_dir/ ./config_dir/
COPY --chown=app:app src/ ./src/
COPY --chown=app:app static/ ./static/
COPY --chown=app:app templates/ ./templates/
COPY --chown=app:app artifacts/model/ ./artifacts/model/
COPY --chown=app:app dataset/train/labels.txt ./dataset/train/labels.txt

USER app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '5000'), timeout=3)"]

CMD ["sh", "-c", "exec gunicorn --workers=1 --bind=0.0.0.0:${PORT:-5000} app:app"]
