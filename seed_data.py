import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_data", settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="tenant_documents")

# Tenant A's documents
collection.add(
    documents=["Acme Corp Q3 2025 revenue: $12.5M", "Acme's secret formula: XYZ-123"],
    metadatas=[{"tenant_id": "tenant_a"}, {"tenant_id": "tenant_a"}],
    ids=["doc_a1", "doc_a2"]
)

# Tenant B's documents
collection.add(
    documents=["Beta Inc. payroll: $2.1M/month", "Beta's unreleased product: Project Phoenix"],
    metadatas=[{"tenant_id": "tenant_b"}, {"tenant_id": "tenant_b"}],
    ids=["doc_b1", "doc_b2"]
)

print("✓ Seeded 4 documents (2 per tenant)")
