# Build context is the REPOSITORY ROOT, not webapp/ (see docker-compose.yml, or run
# `docker build -f webapp/Dockerfile .` from the root). lada-web uses lada's pgfplots writer, and
# PyPI's lada (1.0.x) is older than this repository's, so the image installs lada from src/
# first and lada-web's `lada>=1.2.0` pin is then already met.
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
# matplotlib builds its font cache the first time it draws; build it now, as the user that will
# run the app, so the first exported figure in a fresh container is as quick as the rest.
RUN python -c "import matplotlib.font_manager"
EXPOSE 8000

# The index page is the cheapest route that proves the app is wired up: it renders a template
# from the catalog and touches no upload path. urllib is in the standard library, so this adds
# nothing to the image (python:3.12-slim has neither curl nor wget).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).read(1)"]

# Worker processes: uvicorn reads WEB_CONCURRENCY when --workers is not given. The app keeps no
#   state between requests (imported data lives in the browser), so any worker can answer any
#   request. Each worker draws one figure at a time (matplotlib's style settings are
#   process-wide), so workers are also how exports run in parallel. Budget ~250 MB per worker.
# --proxy-headers/--forwarded-allow-ips: the README recommends a reverse proxy in front, and
#   without these uvicorn reports the proxy's own address and scheme for every request. '*'
#   trusts X-Forwarded-* from any peer, which is only safe because nothing but the proxy can
#   reach this port: do not publish it straight to the internet (see docker-compose.yml).
#   On a platform that terminates TLS itself (Render, Fly.io, Cloudflare), this is what it needs.
# --limit-concurrency: in-flight connections past this answer 503 instead of queuing forever.
ENV WEB_CONCURRENCY=2
CMD ["sh", "-c", "exec uvicorn lada_web.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*' --limit-concurrency 32"]
