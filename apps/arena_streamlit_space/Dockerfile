FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=127.0.0.1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ARENA_ROOT=/data/arena/results \
    ARENA_API_HOST=127.0.0.1 \
    ARENA_API_PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx \
    && rm -f /etc/nginx/sites-enabled/default \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY api_app.py .
COPY start.sh .
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY agents ./agents
COPY arena ./arena
COPY evaluation ./evaluation
COPY mystery_world ./mystery_world
COPY scripts ./scripts
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
