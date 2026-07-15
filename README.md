# Multi-Tenant LLM Gateway

A small FastAPI app for multi-tenant chat with tenant isolation, rate limiting, local LLM calls, and usage tracking.

## Features

- Requires an `X-Tenant-ID` header.
- Limits requests per tenant with a token bucket.
- Pulls only tenant-specific documents from ChromaDB.
- Sends tenant-scoped context to a local Ollama model.
- Tracks token usage in SQLite.
- Provides a usage endpoint for each tenant.

## Files

- `multi_tenant_gateway.py` — main API server.
- `seed_data.py` — sample document seeding.
- `requirements.txt` — dependencies.

## Setup

```bash
pip install -r requirements.txt
python seed_data.py
python multi_tenant_gateway.py
```

## Endpoints

### Chat
`POST /v1/chat/completions`

Header:

```http
X-Tenant-ID: tenant_a
```

Body:

```json
{
  "messages": [
    {"role": "user", "content": "What documents do I have?"}
  ]
}
```

### Usage
`GET /v1/usage/{tenant_id}?hours=24`

Returns token usage for that tenant.

## Sample Data

The seed script adds example documents for:

- `tenant_a`
- `tenant_b`

This lets you test tenant isolation right away.
