FROM python:3.10.11-slim

LABEL authors="Research Center Data Intelligence - Zuyd University of Applied Science, Bryan Kroon"
LABEL owners="Research Center Data Intelligence - Zuyd University of Applied Science, Bryan Kroon"

# Setup apt dependencies
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get install -y -q wget protobuf-compiler

# Setup pip dependencies
WORKDIR $CONTAINER_HOME
COPY ./requirements.txt $CONTAINER_HOME/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project to container
COPY . $CONTAINER_HOME

# Entrypoint
ENTRYPOINT ["gunicorn", "--worker-class=gunicorn.workers.ggevent.GeventWorker", "--bind=0.0.0.0:8000", "--workers=8", "--threads=4", "manage:app"]
