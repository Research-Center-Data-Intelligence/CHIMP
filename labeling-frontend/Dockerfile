FROM python:3.11.3-slim

ENV CONTAINER_HOME=/usr/src/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . $CONTAINER_HOME
WORKDIR $CONTAINER_HOME

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 5261

ENTRYPOINT ["python", "main.py"]
