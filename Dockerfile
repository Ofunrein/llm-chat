FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# torch CPU — install first as its own layer for caching
RUN pip install --no-cache-dir \
    "torch>=2.2,<2.4" \
    --index-url https://download.pytorch.org/whl/cpu

# verify torch is importable before proceeding
RUN python -c "import torch; print('torch', torch.__version__)"

# all other deps
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "tiktoken>=0.7" \
    "transformers>=4.40" \
    "python-dotenv>=1.0"

# verify transformers can see torch
RUN python -c "from transformers import GPT2LMHeadModel; print('transformers ok')"

COPY . .

EXPOSE 7860

ENV GPT2_MODEL=gpt2
ENV MAX_NEW_TOKENS=256
ENV TEMPERATURE=0.85
ENV TOP_K=50
ENV HOST=0.0.0.0
ENV PORT=7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
