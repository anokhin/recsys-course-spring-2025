FROM python:3.7-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY botify ./botify
COPY data ./data
COPY userbody.ft .
COPY userbody.ft.vectors_ngrams.npy .

RUN mkdir -p ./log

ENV PYTHONPATH "${PYTHONPATH}:/app"