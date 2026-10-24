# Version-pinned image so the book's examples reproduce years from now.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /synnet-bench

# CPU-only torch from the pinned wheel index, then the rest of the pinned stack.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/synnet-bench

# Default: run the test suite (leakage guards + reference-model smoke tests).
CMD ["python", "-m", "pytest", "tests/", "-q"]
