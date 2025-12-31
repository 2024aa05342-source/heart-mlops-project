#!/bin/bash

echo "Starting FastAPI..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

sleep 3

echo "Starting Streamlit UI..."
streamlit run app.py --server.port 8501
