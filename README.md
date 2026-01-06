## Clean local demo (Task 7 local substitute)

Because cloud CD isn’t possible for this submission, the deployment is demonstrated **locally** using Docker Compose with a one-shot init service.

### Run everything
```bash
docker compose up --build
```

What happens automatically:
1. **trainer** service:
   - downloads the dataset to `data/heart.csv`
   - runs CV + hyperparameter tuning
   - saves a single reproducible pipeline to `models/model_pipeline.joblib` (Task 4)
   - saves metadata to `models/model_meta.json`
2. **api** service starts only after trainer finishes successfully.
3. **streamlit**, **prometheus**, **grafana** start after api.

### URLs
- API docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default login admin/admin unless you changed it)

### Re-run training only
```bash
docker compose run --rm trainer
```

### Clean reset
```bash
docker compose down -v
rm -rf models data mlruns
```

### Monitoring grafana
```bash
Containers should be running 
Prometheus: http://localhost:9090
Grafana: https://localhst:3000
```

### MLFlow Experiment Traking
```bash
mlflow ui --backend-store-uri ./mlruns
```

