FROM python:3.12-slim

WORKDIR /app

# torch CPU (large layer — cache separately for fast rebuilds)
RUN pip install --no-cache-dir \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# app deps
RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    "uvicorn[standard]>=0.30" \
    tiktoken>=0.7 \
    transformers>=4.40 \
    python-dotenv>=1.0

COPY . .

EXPOSE 7860

ENV GPT2_MODEL=gpt2
ENV MAX_NEW_TOKENS=256
ENV TEMPERATURE=0.85
ENV TOP_K=50

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
