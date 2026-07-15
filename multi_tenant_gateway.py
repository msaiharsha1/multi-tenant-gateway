import time
import sqlite3
import requests
from dataclasses import dataclass
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import chromadb
from chromadb.config import Settings

# ============== CONFIGURATION ==============
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# ============== TOKEN BUCKET RATE LIMITER ==============
@dataclass
class RateLimitResult:
    allowed: bool
    remaining_tokens: int
    retry_after: Optional[float] = None

class TokenBucketLimiter:
    """Per-tenant token bucket rate limiter with in-memory state."""

    def __init__(self, capacity: int = 10000, refill_rate: float = 166.67):
        """
        capacity: Max tokens per tenant (burst limit)
        refill_rate: Tokens per second (10,000 tokens / 60 seconds ≈ 166.67)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: dict[str, dict] = {}  # tenant_id -> {tokens, last_update}

    def _get_bucket(self, tenant_id: str) -> dict:
        if tenant_id not in self.buckets:
            self.buckets[tenant_id] = {"tokens": self.capacity, "last_update": time.monotonic()}
        return self.buckets[tenant_id]

    def _refill(self, bucket: dict) -> None:
        now = time.monotonic()
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.refill_rate)
        bucket["last_update"] = now

    def consume(self, tenant_id: str, tokens: int = 1) -> RateLimitResult:
        bucket = self._get_bucket(tenant_id)
        self._refill(bucket)

        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return RateLimitResult(allowed=True, remaining_tokens=int(bucket["tokens"]))
        else:
            wait_time = (tokens - bucket["tokens"]) / self.refill_rate
            return RateLimitResult(allowed=False, remaining_tokens=0, retry_after=wait_time)

# ============== COST ATTRIBUTION LOGGER (SQLite) ==============
class CostAttributionDB:
    """Persistent cost tracking per tenant."""

    def __init__(self, db_path: str = "tenant_costs.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                tenant_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tenant ON token_usage(tenant_id)")
        self.conn.commit()

    def log_usage(self, tenant_id: str, prompt_tokens: int, completion_tokens: int):
        total = prompt_tokens + completion_tokens
        self.conn.execute(
            "INSERT INTO token_usage VALUES (?, ?, ?, ?, ?)",
            (tenant_id, time.time(), prompt_tokens, completion_tokens, total)
        )
        self.conn.commit()

    def get_tenant_usage(self, tenant_id: str, hours: int = 24) -> dict:
        cursor = self.conn.execute(
            """SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens)
               FROM token_usage
               WHERE tenant_id = ? AND timestamp > ?""",
            (tenant_id, time.time() - hours * 3600)
        )
        row = cursor.fetchone()
        return {
            "prompt_tokens": row[0] or 0,
            "completion_tokens": row[1] or 0,
            "total_tokens": row[2] or 0
        }

# ============== INITIALIZATION ==============
app = FastAPI(title="Multi-Tenant LLM Gateway")
chroma_client = chromadb.PersistentClient(path="./chroma_data", settings=Settings(anonymized_telemetry=False))
vector_db = chroma_client.get_or_create_collection(name="tenant_documents")
rate_limiter = TokenBucketLimiter(capacity=10000, refill_rate=166.67)
cost_db = CostAttributionDB()

# ============== MIDDLEWARE: TENANT AUTH & ISOLATION ==============
@app.middleware("http")
async def tenant_isolation_middleware(request: Request, call_next):
    tenant_id = request.headers.get("X-Tenant-ID")

    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID header")

    # Attach tenant_id to request state for downstream use
    request.state.tenant_id = tenant_id

    # Rate limiting check (non-blocking, tokens consumed on success)
    result = rate_limiter.consume(tenant_id, tokens=1)
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {result.retry_after:.1f}s",
            headers={"Retry-After": str(int(result.retry_after or 1))}
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining_tokens)
    return response

# ============== API ENDPOINTS ==============
@app.post("/v1/chat/completions")
async def chat_completion(
    request: Request,
    tenant_id: str = Header(..., alias="X-Tenant-ID"),
):
    body = await request.json()
    messages_input = body.get("messages", [])

    # 1. VECTOR SEARCH WITH DETERMINISTIC TENANT ISOLATION
    user_message = messages_input[-1].get("content", "") if messages_input else ""

    vector_results = vector_db.query(
        query_texts=[user_message],
        n_results=3,
        where={"tenant_id": tenant_id}
    )

    if not vector_results["documents"] or not vector_results["documents"][0]:
        context = "No relevant documents found for your tenant."
    else:
        context = "\n\n".join(vector_results["documents"][0])

    # 2. CONSTRUCT RAG PROMPT
    system_prompt = f"""You are a helpful AI copilot. Use ONLY the following context from your tenant's documents.
If the context doesn't contain the answer, say you don't have access to that information.

CONTEXT:
{context}

ANSWER:"""

    # 3. CALL LLM (Ollama - FREE, runs locally)
    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": system_prompt + " " + user_message,
                "stream": False
            },
            timeout=30
        )
        llm_answer = ollama_response.json()["response"]
    except Exception as e:
        llm_answer = f"[Ollama error: {str(e)}]. This is a mock response to show the gateway works."

    # Mock token counts for cost attribution
    prompt_tokens = len(system_prompt.split()) + len(user_message.split())
    completion_tokens = len(llm_answer.split())

    # 4. COST ATTRIBUTION
    cost_db.log_usage(
        tenant_id=tenant_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )

    # 5. RETURN RESPONSE
    return {
        "id": f"ollama-chatcmpl-{int(time.time())}",
        "choices": [llm_answer],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

@app.get("/v1/usage/{tenant_id}")
async def get_usage(tenant_id: str, hours: int = 24):
    """Get cost attribution report for a tenant."""
    usage = cost_db.get_tenant_usage(tenant_id, hours)
    return {"tenant_id": tenant_id, "time_window_hours": hours, **usage}

# ============== RUN SERVER ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
