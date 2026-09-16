# Build context is the REPOSITORY ROOT, not webapp/ (see docker-compose.yml, or run
# `docker build -f webapp/Dockerfile .` from the root). lada-web needs the lada in this repository
# (progress_callback, the *_from_trajectory functions), which is newer than what PyPI publishes, so
# the image installs lada from src/ first and lada-web's `lada>=1.2.0` pin is then already met.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 lada
WORKDIR /build
COPY pyproject.toml README.md ./lada/
COPY src ./lada/src
COPY webapp/pyproject.toml webapp/README.md ./webapp/
COPY webapp/src ./webapp/src
RUN pip install --upgrade pip && pip install ./lada ./webapp && rm -rf /build
WORKDIR /app
USER lada
EXPOSE 8000

# The index page is the cheapest route that proves the app is wired up: it renders a template
# from the catalog and touches no upload path. urllib is in the standard library, so this adds
# nothing to the image (python:3.12-slim has neither curl nor wget).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).read(1)"]

# --workers 1: calculation jobs and their progress live in this process's memory (jobs.JobStore).
#   With a second worker, a job started on one would answer "Job not found" when polled on the
#   other. Parallelism comes from the job threads instead, capped by LADA_WEB_MAX_CONCURRENT_JOBS.
# --proxy-headers/--forwarded-allow-ips: the README recommends running nginx in front, and
#   without these uvicorn reports the proxy's own address and scheme for every request. '*'
#   trusts X-Forwarded-* from any peer, which is only safe because nothing but the proxy can
#   reach this port — do not publish it straight to the internet (see docker-compose.yml).
# --limit-concurrency: in-flight connections past this answer 503 instead of queuing forever.
#   Progress polls are short but count too, hence more headroom than the job cap itself.
CMD ["uvicorn", "lada_web.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--limit-concurrency", "16"]
