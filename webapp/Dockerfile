FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 lada
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .
USER lada
EXPOSE 8000
CMD ["uvicorn", "lada_web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
