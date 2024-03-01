FROM tensorflow/tensorflow:2.6.1-gpu

LABEL authors="Blazoned"
LABEL owners="Research Center Data Intelligence - Zuyd University of Applied Science, Bryan Kroon"
LABEL nl.blazoned.emotionrecognition.service="experimentation"

# Set environment variables:
# - Base directory,
# - Python create binary files (python -b)
# - Python output buffer stream (python -u)
# - Allow deprecated sklearn is important for Talos automl. Leave it enabled!
ENV CONTAINER_HOME=/usr/src/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Update nvidia keyring
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Add dependencies required for healthcheck, opencv (ffmpeg, libsm6, libxext6), mlflow (git) and tensorflow (cuda's cudnn)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get upgrade -y && apt-get install -y -q wget git ffmpeg libsm6 libxext6

# Copy project to container, then install requirements
ADD . $CONTAINER_HOME
WORKDIR $CONTAINER_HOME
RUN pip install --upgrade pip
RUN pip install -r $CONTAINER_HOME/requirements.txt

# Execute service
ENTRYPOINT ["gunicorn", "--worker-class=gunicorn.workers.ggevent.GeventWorker", "--bind=0.0.0.0:8000", "--workers=8", "--threads=128", "--timeout=0", "main:get_app()"]
