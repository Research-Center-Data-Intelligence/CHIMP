FROM python:3.11.3-slim

LABEL authors="Blazoned"
LABEL owners="Research Center Data Intelligence - Zuyd University of Applied Science, Bryan Kroon"
LABEL nl.blazoned.emotionrecognition.service="mlflow server"

# Set environment variables:
# - Python create binary files (python -b)
# - Python output buffer stream (python -u)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Add dependencies required for healthcheck
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get upgrade -y && apt-get install -y -q wget

# Install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade mlflow==2.3.0 cryptography
