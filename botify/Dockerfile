FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data

RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"