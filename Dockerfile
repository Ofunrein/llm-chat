FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# single RUN = single layer = no PATH divergence between build/runtime
RUN pip3 install --no-cache-dir \
    "torch==2.2.2" \
    "torchvision==0.17.2" \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "tiktoken>=0.7" \
    "transformers>=4.40,<5.0" \
    "python-dotenv>=1.0" && \
    python3 -c "import torch; print('torch', torch.__version__)" && \
    python3 -c "from transformers import GPT2LMHeadModel; m=GPT2LMHeadModel.from_pretrained('gpt2'); print('GPT-2 load test OK, params:', sum(p.numel() for p in m.parameters())//1000000, 'M')"

COPY . .

EXPOSE 7860

ENV GPT2_MODEL=gpt2
ENV MAX_NEW_TOKENS=200
ENV TEMPERATURE=0.85
ENV TOP_K=40

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
