FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

# API container just runs the API. Training happens in a separate init/trainer step (clean demo).
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
