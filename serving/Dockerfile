FROM python:3.10.11-slim

LABEL authors="Blazoned"
LABEL owners="Research Center Data Intelligence - Zuyd University of Applied Science, Bryan Kroon"
LABEL nl.blazoned.emotionrecognition.service="serving server"

# Set environment variables:
# - Base directory,
# - Python create binary files (python -b)
# - Python output buffer stream (python -u)
ENV CONTAINER_HOME=/usr/src/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Add dependencies required for onnxruntime and healthcheck
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get upgrade -y && apt-get install -y -q wget protobuf-compiler

# Copy project to container, then install requirements
ADD . $CONTAINER_HOME
WORKDIR $CONTAINER_HOME
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade -r $CONTAINER_HOME/requirements.txt

# Execute service
ENTRYPOINT ["gunicorn", "--worker-class=gunicorn.workers.ggevent.GeventWorker", "--bind=0.0.0.0:8000", "--workers=8", "--threads=4", "main:get_app()"]
