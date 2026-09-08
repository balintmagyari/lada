FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 lada
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .
USER lada
EXPOSE 8000

# The index page is the cheapest route that proves the app is wired up: it renders a template
# from the catalog and touches no upload path. urllib is in the standard library, so this adds
# nothing to the image (python:3.12-slim has neither curl nor wget).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).read(1)"]

# --proxy-headers/--forwarded-allow-ips: the README recommends running nginx in front, and
#   without these uvicorn reports the proxy's own address and scheme for every request. '*'
#   trusts X-Forwarded-* from any peer, which is only safe because nothing but the proxy can
#   reach this port — do not publish it straight to the internet (see docker-compose.yml).
# --limit-concurrency: the calculations are CPU-bound and each one occupies a worker thread
#   for its whole duration, so past a handful in flight everything just gets slower together.
#   8 per worker (16 across the two) makes the queue answer 503 instead of growing forever.
CMD ["uvicorn", "lada_web.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--limit-concurrency", "8"]
