FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

# アプリケーションの起動前にログを出力
RUN echo "Starting application with environment variables:"
RUN echo "PORT=$PORT"

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
