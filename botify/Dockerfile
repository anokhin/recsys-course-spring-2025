FROM python:3.7-bullseye

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data
COPY train ./train

RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"
