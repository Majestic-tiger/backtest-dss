FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir streamlit yfinance pandas numpy optuna

COPY main.py ./
COPY engines ./engines
COPY ui ./ui
COPY pages ./pages

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
