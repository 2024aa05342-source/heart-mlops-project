FROM python:3.10-slim

WORKDIR /app

# Faster, cleaner installs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# API uses 8000, Streamlit uses 8501 (compose overrides command)
EXPOSE 8000
EXPOSE 8501

# Default: run FastAPI (compose can override for Streamlit)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
