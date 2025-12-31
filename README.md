## Heart Disease Prediction - MLOps Pipeline

This project builds a Machine Learning workflow for **Heart Disease Prediction** using an automated **MLOps CI pipeline, model training, testing, API deployment & Streamlit UI dashboard**.

---

## 🚀 Project Features

| Component | Status |
|----------|--------|
| Data Preprocessing | ✔ One-Hot Encoding + Scaling + Missing handling |
| Model Training | ✔ Logistic Regression & RandomForest |
| Model Serialization | ✔ Saved in `/models` folder |
| Web API | ✔ FastAPI Endpoint `/predict` |
| UI Dashboard | ✔ Streamlit app for prediction & visualization |
| Docker Deployment | ✔ Ready (image can run API + UI together) |
| CI/CD | ✔ GitHub Actions: test + train + upload model artifact |

---

## 📂 Project Structure

📁 heart-mlops-project
│── models/                  # trained model artifacts
│── notebooks/               # EDA & analysis
│── src/
│   ├── data_loader.py       # reads dataset
│   ├── preprocess.py        # encoding + scaling + split
│   ├── train.py             # trains + saves models + metrics
│   ├── api.py               # FastAPI backend for prediction
│── tests/
│   ├── test_preprocess.py   # preprocessing tests
│   ├── test_training.py     # training test
│── app.py                   # Streamlit UI
│── requirements.txt
│── Dockerfile
│── run_local.sh
│── README.md

##Local Setup & Run
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

## Install dependencies
pip install -r requirements.txt

## Train model
python src/train.py

## Start FastAPI Backend
uvicorn src.api:app --reload --port 8000

## Run StreamLit UI
streamlit run app.py

## Run Tests
pytest -q

## Docker Deployment
docker build -t heart-app .
docker run -p 8000:8000 -p 8501:8501 heart-app


