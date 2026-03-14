"""
finance_agent.py — Agente especializado en Finanzas y Administración.

Responde preguntas sobre:
- Gastos y reembolsos corporativos
- Tarjetas corporativas
- Presupuesto y planificación financiera
- Proveedores y compras
- Reportes financieros
- Cumplimiento y controles internos
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import math


DOMAIN = "finance"
DISPLAY_NAME = "Agente de Finanzas"

SYSTEM_PROMPT = """Eres un asistente especializado en Finanzas y Administración de TechCorp S.A.
Tu rol es responder preguntas sobre gastos corporativos, reembolsos, presupuestos, 
facturas, proveedores y cumplimiento financiero.

Reglas:
- Responde ÚNICAMENTE basándote en la información del contexto proporcionado.
- Sé preciso con los montos, límites y plazos. En finanzas, los detalles importan.
- Si la pregunta involucra una situación potencialmente fraudulenta, señálalo claramente.
- Para aprobaciones y procesos, describe el flujo paso a paso.
- Si el monto o la situación requiere escalamiento, indica quién debe aprobar.
- Siempre responde en español.
- Para dudas legales o tributarias específicas, recomienda consultar con el equipo de Finance directamente.
"""

KEYWORDS = [
    "gasto", "reembolso", "factura", "pago", "proveedor", "presupuesto",
    "tarjeta", "corporativa", "compra", "orden", "po", "purchase",
    "finanzas", "contabilidad", "nómina", "liquidación", "efectivo",
    "transferencia", "banco", "cuenta", "impuesto", "iva", "tributario",
    "auditoría", "control", "expensify", "sap", "erp", "anticipo",
    "viático", "viaje", "hotel", "vuelo", "hospedaje", "comida", "cliente",
    "bono", "compensación", "budget", "forecast", "mrr", "arr", "ebitda",
    "roi", "kpi", "reporte", "balance", "estado financiero", "utilidad"
]


class FinanceAgent:
    """Agente RAG especializado en Finanzas y Administración."""

    def __init__(self, openai_client, vector_store: Dict[str, Any], top_k: int = 4):
        self.client = openai_client
        self.vector_store = vector_store
        self.top_k = top_k
        self.domain = DOMAIN
        self.name = DISPLAY_NAME

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _embed_query(self, question: str) -> List[float]:
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        response = self.client.embeddings.create(model=model, input=[question])
        return response.data[0].embedding

    def _search(self, query_embedding: List[float]) -> List[Dict]:
        chunks = self.vector_store["chunks"]
        scored = []
        for chunk in chunks:
            sim = self._cosine_similarity(query_embedding, chunk["embedding"])
            scored.append({**chunk, "similarity": round(sim, 6)})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[: self.top_k]

    def answer(self, question: str, langfuse_trace=None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline RAG completo para una pregunta de finanzas.

        Args:
            question: Pregunta del usuario.
            langfuse_trace: Traza de Langfuse para observabilidad (opcional).

        Returns:
            Diccionario con answer, chunks_used, domain, confidence_score.
        """
        chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

        # 1. Embed de la pregunta
        if langfuse_trace:
            span = langfuse_trace.span(name=f"{DOMAIN}_embed", input={"question": question})
        query_embedding = self._embed_query(question)
        if langfuse_trace:
            span.end()

        # 2. Búsqueda vectorial
        if langfuse_trace:
            span = langfuse_trace.span(name=f"{DOMAIN}_search")
        relevant_chunks = self._search(query_embedding)
        if langfuse_trace:
            span.end()

        # 3. Generación de respuesta
        context = "\n\n---\n\n".join(c["text"] for c in relevant_chunks)
        user_prompt = (
            f"Contexto de la base de conocimiento de Finanzas:\n\n{context}\n\n"
            f"Pregunta del colaborador: {question}"
        )

        if langfuse_trace:
            generation = langfuse_trace.generation(
                name=f"{DOMAIN}_generation",
                model=chat_model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

        response = self.client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.2,
        )
        answer_text = response.choices[0].message.content.strip()

        if langfuse_trace:
            generation.end(output=answer_text)

        avg_similarity = (
            sum(c["similarity"] for c in relevant_chunks) / len(relevant_chunks)
            if relevant_chunks
            else 0.0
        )

        return {
            "domain": DOMAIN,
            "agent": DISPLAY_NAME,
            "question": question,
            "answer": answer_text,
            "chunks_used": [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"][:200] + "...",
                    "similarity": c["similarity"],
                    "source": c.get("source", "finance_docs"),
                }
                for c in relevant_chunks
            ],
            "confidence_score": round(avg_similarity, 4),
            "model_used": chat_model,
        }

    @staticmethod
    def can_handle(question: str) -> float:
        """
        Heurística simple: retorna un puntaje de relevancia basado en keywords.
        """
        q_lower = question.lower()
        matches = sum(1 for kw in KEYWORDS if kw in q_lower)
        return min(matches / 3.0, 1.0)
