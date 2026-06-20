# Deployment

Financier runs locally with zero cloud setup. This guide covers local, Docker, and
AWS options.

## Local

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
streamlit run src/web_app/streamlit_app.py
```

Add real models by copying `.env.example` to `.env` and setting
`ANTHROPIC_API_KEY` (or AWS / Gemini credentials).

## Docker

```bash
docker build -t financier .
docker run -p 8501:8501 --env-file .env financier
```

Open http://localhost:8501. The image installs only `requirements.txt`; to
enable semantic RAG, add `requirements-extras.txt` to the build.

## AWS — models via Bedrock

1. `pip install -r requirements.txt` (includes `boto3`).
2. Request access to the Amazon Nova models in the Amazon Bedrock console.
3. Provide AWS credentials (env vars, `aws configure`, or an instance role).
4. In `config.yaml`:
   ```yaml
   llm:
     provider: bedrock
     bedrock:
       engine: converse
       region: us-east-1
   ```
   Defaults to Amazon Nova (`engine: converse`); set ids under
   `llm.bedrock.models`. See the README for details.

## AWS — DynamoDB profile store

1. Create a table (e.g. `financier-profiles`) with partition key `user_id`
   (String).
2. In `config.yaml`:
   ```yaml
   memory:
     backend: dynamodb
     dynamodb:
       table: financier-profiles
       region: us-east-1
   ```
   Financier falls back to the local JSON store if the table or credentials are
   unavailable.

## Hosting options

- **Streamlit Community Cloud** — point it at `src/web_app/streamlit_app.py`
  and add secrets in the dashboard. Easiest free option.
- **AWS App Runner / ECS Fargate** — deploy the Docker image; set environment
  variables for keys. App Runner is the simplest container path.
- **A small EC2 instance** — run the container with `--env-file .env`.

## Cost notes

- yfinance and the mock backends are free.
- Claude usage is billed per token; the router uses the cheaper Haiku model and
  the specialists use Opus. Switch `llm.models.default` to
  `claude-sonnet-4-6` to reduce cost.
