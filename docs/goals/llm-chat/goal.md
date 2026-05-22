# Goal: LLM Chat App — LLM Chat

## Charter

**Original request:** Build an LLM-powered chat/form app from scratch in Python, init public GitHub repo, make it look sexy and impressive for production-level hiring, use all bells and whistles, one-shot autonomous execution.

**Interpreted outcome:** A publicly available GitHub repo containing a polished Python LLM chat application that impresses a senior engineering recruiter — production-quality code, beautiful UI, streaming responses, multi-model support, conversation history, and a live demo URL.

**Input shape:** specific (stack: Python; goal: portfolio repo; audience: senior engineering recruiters)

**Authority:** approved — user said "do all in one go, don't ask"

**Language:** Python

## Goal Oracle

The goal is complete when:
1. `gh repo view` shows a public repo under `Ofunrein` with a populated README, live demo badge, and clean code structure
2. App runs locally: `uv run python app.py` (or `uvicorn`/`streamlit`) launches without errors and chat works end-to-end
3. Deployed demo URL is accessible and the UI is visually impressive
4. Code quality passes: type hints, linting, tests present

## Constraints

- Python stack (FastAPI + Streamlit OR Gradio — judge picks the most impressive)
- Use Anthropic Claude API (`claude-sonnet-4-6` model) with `sk-proj-...` OpenAI key also available
- Repo name: `llm-chat` (clean, hireable)
- Public GitHub repo under `Ofunrein`
- No co-authored-by lines, no AI tool attribution in commits
- Author: `Martin O. <Ofunrein@users.noreply.github.com>`
- Use `gh` not `git clone` (SSH unavailable)
- Superpowers / obra skills to be used inside `/goal` run as implementation aids

## Likely Misfire

Building a toy demo that looks like tutorial code — no streaming, no error handling, no tests, plain default Gradio theme. Must look like something a senior engineer shipped.

## Tranche

Scout design options → Judge picks stack + design → Worker builds full app (backend + UI + README + tests) → Worker inits and pushes GitHub repo → Worker deploys (Vercel or HuggingFace Spaces) → Judge final audit.

## Non-Goals

- No auth system (keep it demo-friendly)
- No database (in-memory conversation history is fine)
- No Docker (unless it adds polish)
