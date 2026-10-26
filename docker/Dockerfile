# Simulator + dashboard in a container:
#   docker build -f docker/Dockerfile -t gdnav .
#   docker run --rm -p 8080:8080 gdnav
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
EXPOSE 8080
CMD ["python", "scripts/run_dashboard.py", "--host", "0.0.0.0", "--port", "8080"]
