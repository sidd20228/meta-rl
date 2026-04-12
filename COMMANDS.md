# Commands

## Setup

```bash
pip install -r requirements.txt
pip install openenv-core
huggingface-cli login
uv run server
```

## Local Test

```bash
uv run server
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

## Run HF Agent

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=<hf-model-id>
export HF_TOKEN=<your-token>
python inference.py
```

Run one task explicitly:

```bash
python inference.py --task hard
```

## Run Compliance Checks

```bash
python -m unittest
pytest
openenv validate
docker build .
```

## Deploy To HF Space

```bash
huggingface-cli login
uv run server
openenv push --repo-id <username>/<space-name>
```

## Validate Public Space

```bash
./validate-submission.sh <url>
```
