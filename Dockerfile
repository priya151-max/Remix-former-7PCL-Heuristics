FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small; remove this line to use the default (CUDA) wheels.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# The checkpoint must be present at model/RemixFormerPP_FULL_best_model.pth
# (copy it before building, or mount it: -v $(pwd)/model:/app/model:ro)

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
# One worker only: the model lives in memory and inference is serialised.
CMD ["python", "run.py"]
