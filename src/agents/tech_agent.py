"""
tech_agent.py — Agente especializado en Soporte Técnico y Tecnología.

Responde preguntas sobre:
- Infraestructura y accesos a sistemas
- Hardware y equipos corporativos
- Stack tecnológico y estándares de desarrollo
- Seguridad informática
- Troubleshooting y soporte técnico
- CI/CD y procesos de deployment
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import math


DOMAIN = "tech"
DISPLAY_NAME = "Agente de Soporte Técnico"

SYSTEM_PROMPT = """Eres un asistente de soporte técnico e ingeniería de TechCorp S.A.
Tu rol es ayudar a los colaboradores con preguntas sobre tecnología, sistemas, infraestructura,
desarrollo de software, seguridad informática y troubleshooting.

Reglas:
- Responde ÚNICAMENTE basándote en la información del contexto proporcionado.
- Para problemas técnicos, proporciona pasos claros y ordenados numerados.
- Cuando sea relevante, menciona los canales de soporte adecuados (Slack, Jira, email).
- Para incidentes de seguridad, siempre prioriza la acción inmediata y el reporte.
- Usa terminología técnica apropiada pero explica los conceptos cuando sea necesario.
- Si no puedes resolver el problema con la información disponible, escala al canal correcto.
- Siempre responde en español, aunque los términos técnicos pueden quedar en inglés.
"""

KEYWORDS = [
    "vpn", "acceso", "sistema", "contraseña", "password", "github", "repositorio",
    "docker", "kubernetes", "deploy", "deployment", "aws", "cloud", "servidor",
    "base de datos", "database", "api", "endpoint", "bug", "error", "fallo",
    "laptop", "equipo", "hardware", "software", "instalación", "configuración",
    "seguridad", "firewall", "certificado", "ssl", "tls", "código", "código",
    "python", "javascript", "typescript", "node", "react", "git", "ci/cd",
    "jira", "confluence", "slack", "correo", "email", "gmail", "soporte",
    "it", "infraestructura", "red", "network", "wifi", "internet", "monitor"
]


class TechAgent:
    """Agente RAG especializado en Soporte Técnico e Ingeniería."""

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
        Ejecuta el pipeline RAG completo para una pregunta técnica.

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
            f"Contexto de la base de conocimiento técnica:\n\n{context}\n\n"
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
                    "source": c.get("source", "tech_docs"),
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
