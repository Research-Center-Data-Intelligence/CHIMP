FROM python:3.11.3-slim

LABEL authors="Blazoned"
LABEL owners="Research Center Data Intelligence - Zuyd University of Applied Science, Bryan Kroon"
LABEL nl.blazoned.emotionrecognition.service="frontend application"

# Set environment variables:
# - Base directory,
# - Python create binary files (python -b)
# - Python output buffer stream (python -u)
ENV CONTAINER_HOME=/usr/src/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Add dependencies required for opencv (ffmpeg, libsm6, libxext6)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get upgrade -y && apt-get install -y -q ffmpeg libsm6 libxext6

# Copy project to container, then install requirements
COPY . $CONTAINER_HOME
WORKDIR $CONTAINER_HOME
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade -r $CONTAINER_HOME/requirements.txt

# Execute service
ENTRYPOINT ["gunicorn", "--worker-class=geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "--bind=0.0.0.0:8000", "--workers=1", "--threads=8", "main:get_app()"]
