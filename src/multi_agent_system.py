"""
multi_agent_system.py — Módulo principal del sistema multi-agente.

Expone funciones de alto nivel para:
- Construir y cargar los vector stores por dominio
- Inicializar los agentes y el orquestador
- Ejecutar consultas y obtener respuestas
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Importar agentes
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agents.hr_agent import HRAgent
from agents.tech_agent import TechAgent
from agents.finance_agent import FinanceAgent
from agents.orchestrator import Orchestrator


# ── Configuración ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


# ── Chunking ───────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, source: str, chunk_size: int = CHUNK_SIZE) -> List[Dict]:
    """
    Divide el texto en chunks con solapamiento.
    
    Estrategia: divide por párrafos y agrupa hasta el tamaño objetivo.
    Conserva el último párrafo del chunk anterior para solapamiento semántico.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_parts: List[str] = []
    current_len = 0
    idx = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current_parts:
            chunk_text = "\n".join(current_parts)
            chunks.append({
                "chunk_id": idx,
                "text": chunk_text,
                "source": source,
                "char_count": len(chunk_text),
                "embedding": None,
            })
            idx += 1
            # Solapamiento: conservar último párrafo
            current_parts = current_parts[-1:] + [para]
            current_len = sum(len(p) for p in current_parts)
        else:
            current_parts.append(para)
            current_len += len(para)

    if current_parts:
        chunk_text = "\n".join(current_parts)
        chunks.append({
            "chunk_id": idx,
            "text": chunk_text,
            "source": source,
            "char_count": len(chunk_text),
            "embedding": None,
        })

    return chunks


def load_domain_docs(domain_dir: Path) -> List[Dict]:
    """
    Carga todos los documentos .txt, .md, .pdf y .csv de un directorio
    y los divide en chunks.
    """
    all_chunks: List[Dict] = []
    chunk_counter = 0

    supported = [".txt", ".md", ".csv"]
    files = [f for f in domain_dir.iterdir() if f.suffix.lower() in supported]

    if not files:
        raise FileNotFoundError(f"No se encontraron documentos en {domain_dir}")

    for doc_path in files:
        text = doc_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(text, source=doc_path.name)
        # Re-numerar para IDs únicos dentro del dominio
        for c in chunks:
            c["chunk_id"] = chunk_counter
            chunk_counter += 1
        all_chunks.extend(chunks)
        print(f"   ✓ {doc_path.name}: {len(chunks)} chunks")

    return all_chunks


# ── Embeddings ─────────────────────────────────────────────────────────────────

def generate_embeddings(chunks: List[Dict], client: OpenAI, batch_size: int = 20) -> List[Dict]:
    """Genera embeddings para todos los chunks en lotes."""
    texts = [c["text"] for c in chunks]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
        if i + batch_size < len(texts):
            time.sleep(0.3)  # Rate limit

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    return chunks


# ── Persistencia ───────────────────────────────────────────────────────────────

def save_vector_store(chunks: List[Dict], path: Path, model: str) -> None:
    """Guarda el vector store en disco como JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "model": model}, f, ensure_ascii=False)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"   ✓ Guardado en {path} ({size_mb:.2f} MB)")


def load_vector_store(path: Path) -> Dict:
    """Carga un vector store desde disco."""
    if not path.exists():
        raise FileNotFoundError(f"Vector store no encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Pipeline principal ────────────────────────────────────────────────────────

def build_all_vector_stores(client: OpenAI, force_rebuild: bool = False) -> Dict[str, Dict]:
    """
    Construye los vector stores para los tres dominios.

    Args:
        client: Instancia de OpenAI (openai_client), NO el string de la API key.
        force_rebuild: Si True, reconstruye aunque ya existan archivos en disco.

    Returns:
        Diccionario {domain: vector_store}

    Raises:
        TypeError: Si `client` es un string en lugar de una instancia de OpenAI.
        TypeError: Si se pasan argumentos inesperados (e.g. typos como 'embeddin').
    """
    # Validación defensiva: detectar error común de pasar la API key en vez del cliente
    if isinstance(client, str):
        raise TypeError(
            "❌ El argumento 'client' debe ser una instancia de OpenAI, no un string.\n"
            "   Correcto:  build_all_vector_stores(client=openai_client)\n"
            "   Incorrecto: build_all_vector_stores(client=OPENAI_API_KEY)\n"
            "   Asegúrate de haber ejecutado: openai_client = OpenAI(api_key=OPENAI_API_KEY)"
        )
    domains = {
        "hr": DATA_DIR / "hr_docs",
        "tech": DATA_DIR / "tech_docs",
        "finance": DATA_DIR / "finance_docs",
    }
    vector_stores = {}

    for domain, doc_dir in domains.items():
        index_path = OUTPUTS_DIR / f"{domain}_index.json"

        if index_path.exists() and not force_rebuild:
            print(f"\n⚡ {domain.upper()}: Cargando índice existente desde {index_path}")
            vector_stores[domain] = load_vector_store(index_path)
            n = len(vector_stores[domain]["chunks"])
            print(f"   ✓ {n} chunks cargados")
            continue

        print(f"\n🔨 {domain.upper()}: Construyendo vector store desde {doc_dir}")
        chunks = load_domain_docs(doc_dir)
        print(f"   → {len(chunks)} chunks en total, generando embeddings...")
        chunks = generate_embeddings(chunks, client)
        save_vector_store(chunks, index_path, EMBEDDING_MODEL)
        vector_stores[domain] = {"chunks": chunks, "model": EMBEDDING_MODEL}
        print(f"   ✅ {domain.upper()} listo: {len(chunks)} chunks indexados")

    return vector_stores


def build_multi_agent_system(
    openai_client: OpenAI,
    langfuse_client=None,
    force_rebuild: bool = False,
) -> Orchestrator:
    """
    Inicializa todo el sistema multi-agente.
    
    Args:
        openai_client: Cliente de OpenAI autenticado.
        langfuse_client: Cliente de Langfuse para observabilidad (opcional).
        force_rebuild: Reconstruir vector stores desde cero.
    
    Returns:
        Orquestador listo para recibir consultas.
    """
    print("\n🚀 Inicializando Sistema Multi-Agente TechCorp...\n")

    # 1. Construir / cargar vector stores
    vector_stores = build_all_vector_stores(openai_client, force_rebuild=force_rebuild)

    # 2. Inicializar agentes
    hr_agent = HRAgent(openai_client, vector_stores["hr"])
    tech_agent = TechAgent(openai_client, vector_stores["tech"])
    finance_agent = FinanceAgent(openai_client, vector_stores["finance"])

    # 3. Inicializar orquestador
    orchestrator = Orchestrator(
        openai_client=openai_client,
        hr_agent=hr_agent,
        tech_agent=tech_agent,
        finance_agent=finance_agent,
        langfuse_client=langfuse_client,
    )

    total_chunks = sum(len(vs["chunks"]) for vs in vector_stores.values())
    print(f"\n✅ Sistema listo. Total chunks indexados: {total_chunks}")
    print("   - HR:", len(vector_stores["hr"]["chunks"]), "chunks")
    print("   - Tech:", len(vector_stores["tech"]["chunks"]), "chunks")
    print("   - Finance:", len(vector_stores["finance"]["chunks"]), "chunks")

    return orchestrator
