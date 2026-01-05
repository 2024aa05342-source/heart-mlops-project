# Kubernetes Deployment (Task 7)

## 1) Build & push image
Replace `YOUR_DOCKER_IMAGE` in manifests with your image (example: docker.io/<user>/heart-mlops).

```bash
docker build -t YOUR_DOCKER_IMAGE:latest .
docker push YOUR_DOCKER_IMAGE:latest
```

Applying kubernets / k8s manifests
```bash
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
kubectl apply -f k8s/ingress.yaml