# Kubernetes
## 1. Win10 local deployment of kubernetes
`Docker Desktop -> Settings -> Kunernetes -> Enable Kubernetes`
## 2. Start pod
```bash
cd kubernetes
# start
kubectl apply -f online-inference-pod.yaml
# pods status
kubectl get pods
# port forwarding
kubectl port-forward pods/online-inference 8000:8000
# stop pod
kubectl delete pods/online-inference
```
