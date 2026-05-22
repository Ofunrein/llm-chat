FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

# torch CPU — extra-index keeps PyPI as primary so other deps resolve correctly
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# hard verify — fails build if torch not importable
RUN python -c "import torch; assert torch.__version__.startswith('2'), torch.__version__; print('torch OK:', torch.__version__)"

# rest of deps (PyPI only — torch already installed above)
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "tiktoken>=0.7" \
    "transformers>=4.40" \
    "python-dotenv>=1.0"

# verify transformers sees torch
RUN python -c "from transformers import GPT2LMHeadModel; print('transformers OK')"

COPY . .

EXPOSE 7860

ENV GPT2_MODEL=gpt2
ENV MAX_NEW_TOKENS=200
ENV TEMPERATURE=0.85
ENV TOP_K=40

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
