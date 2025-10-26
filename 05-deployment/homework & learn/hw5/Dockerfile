FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV UV_PROJECT_ENVIRONMENT=/usr/local

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY question_4.py pipeline_v1.bin ./

EXPOSE 9696
CMD ["uvicorn", "question_4:app", "--host=0.0.0.0", "--port=9696"]