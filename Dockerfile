FROM dhi.io/pytorch:2.10-cuda13.0-cudnn9-debian13-dev AS dev

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

# RUN python -m venv --system-site-packages /app/venv

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8889", "--no-browser"]

FROM dhi.io/pytorch:2.10-cuda13.0-cudnn9-debian13 AS production

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

RUN python -m venv --system-site-packages /app/venv

COPY requirements-prod.txt /tmp/requirements-prod.txt
RUN pip install --no-cache-dir -r /tmp/requirements-prod.txt

WORKDIR /workspace

CMD ["python", "-c", "import torch; print('Production image ready. Torch:', torch.__version__)"]
