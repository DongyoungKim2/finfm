FROM python:3.10-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir yfinance pandas transformers trl fastapi typer uvicorn

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
