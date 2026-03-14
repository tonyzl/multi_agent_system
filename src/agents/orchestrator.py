"""
orchestrator.py — Orquestador del sistema multi-agente.

El orquestador:
1. Recibe la pregunta del usuario
2. Clasifica la intención usando GPT (routing LLM)
3. Delega al agente especializado correspondiente
4. Retorna la respuesta con metadatos de trazabilidad
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI


ROUTING_SYSTEM_PROMPT = """Eres un clasificador de intención para un sistema multi-agente corporativo.
Tu única tarea es determinar a qué dominio pertenece la pregunta de un colaborador.

Dominios disponibles:
- "hr": Preguntas sobre Recursos Humanos — vacaciones, permisos, beneficios, salarios, onboarding, 
  desempeño, evaluaciones, políticas de RRHH, código de conducta, renuncia, contratación.
- "tech": Preguntas sobre Tecnología y Soporte — accesos a sistemas, VPN, hardware, software,
  desarrollo de software, infraestructura, seguridad informática, GitHub, CI/CD, bugs, troubleshooting.
- "finance": Preguntas sobre Finanzas y Administración — gastos, reembolsos, facturas, proveedores,
  presupuesto, tarjetas corporativas, compras, auditoría, métricas financieras.
- "unknown": Si la pregunta no pertenece claramente a ninguno de los tres dominios o es irrelevante.

Responde ÚNICAMENTE con un objeto JSON con la siguiente estructura (sin markdown, sin texto adicional):
{
  "domain": "hr" | "tech" | "finance" | "unknown",
  "confidence": 0.0 a 1.0,
  "reasoning": "una frase breve explicando la clasificación"
}
"""


class Orchestrator:
    """
    Orquestador central del sistema multi-agente.
    
    Responsabilidades:
    - Clasificar la intención de la pregunta mediante un LLM de routing
    - Delegar al agente correcto (HR, Tech, Finance)
    - Integrar con Langfuse para trazabilidad completa
    - Manejar casos borde (unknown domain, baja confianza)
    """

    def __init__(
        self,
        openai_client: OpenAI,
        hr_agent,
        tech_agent,
        finance_agent,
        langfuse_client=None,
    ):
        self.client = openai_client
        self.agents = {
            "hr": hr_agent,
            "tech": tech_agent,
            "finance": finance_agent,
        }
        self.langfuse = langfuse_client
        self.routing_model = os.getenv("ROUTING_MODEL", "gpt-4o-mini")

    def _classify_intent(self, question: str, trace=None) -> Dict[str, Any]:
        """
        Clasifica la intención de la pregunta usando un LLM.
        
        Returns:
            Dict con domain, confidence y reasoning.
        """
        if trace:
            generation = trace.generation(
                name="routing_classification",
                model=self.routing_model,
                input=[
                    {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Pregunta: {question}"},
                ],
            )

        response = self.client.chat.completions.create(
            model=self.routing_model,
            messages=[
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Pregunta: {question}"},
            ],
            max_tokens=150,
            temperature=0.0,  # Determinístico para routing
        )

        raw = response.choices[0].message.content.strip()

        try:
            classification = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback si el LLM no devuelve JSON válido
            classification = {
                "domain": "unknown",
                "confidence": 0.0,
                "reasoning": f"Error al parsear clasificación: {raw[:100]}",
            }

        if trace:
            generation.end(output=classification)

        return classification

    def _handle_unknown(self, question: str) -> Dict[str, Any]:
        """Maneja preguntas que no pertenecen a ningún dominio."""
        return {
            "domain": "unknown",
            "agent": "Orquestador",
            "question": question,
            "answer": (
                "Lo siento, tu pregunta no parece estar relacionada con Recursos Humanos, "
                "Tecnología o Finanzas. Para consultas generales, puedes contactar a:\n\n"
                "- RRHH: hr@techcorp.com | Ext. 2100\n"
                "- IT: it-support@techcorp.com | Ext. 3000\n"
                "- Finanzas: finance@techcorp.com | Ext. 4000"
            ),
            "chunks_used": [],
            "confidence_score": 0.0,
            "routing": {"domain": "unknown", "confidence": 0.0, "reasoning": "Fuera de dominio"},
            "model_used": None,
        }

    def route_and_answer(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Pipeline completo: clasifica la pregunta y delega al agente correcto.

        Args:
            question: Pregunta del usuario.
            session_id: ID de sesión para trazabilidad (opcional).

        Returns:
            Respuesta completa con metadatos.
        """
        # Crear traza de Langfuse si está disponible
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name="multi_agent_query",
                input={"question": question},
                session_id=session_id,
                tags=["multi-agent", "rag"],
            )

        try:
            # 1. Clasificar intención
            classification = self._classify_intent(question, trace)
            domain = classification.get("domain", "unknown")
            confidence = classification.get("confidence", 0.0)

            # 2. Si confianza es muy baja, tratar como unknown
            if confidence < 0.4 and domain != "unknown":
                domain = "unknown"
                classification["reasoning"] += " (confianza baja → escalado a unknown)"

            # 3. Delegar al agente o manejar unknown
            if domain in self.agents:
                agent = self.agents[domain]
                result = agent.answer(question, langfuse_trace=trace)
            else:
                result = self._handle_unknown(question)

            # 4. Enriquecer con metadatos de routing
            result["routing"] = {
                "domain": domain,
                "confidence": confidence,
                "reasoning": classification.get("reasoning", ""),
            }

            # 5. Actualizar traza de Langfuse con el resultado
            if trace:
                trace.update(
                    output={
                        "answer": result.get("answer", ""),
                        "domain": domain,
                        "confidence": confidence,
                    }
                )

            return result

        except Exception as e:
            error_result = {
                "domain": "error",
                "agent": "Orquestador",
                "question": question,
                "answer": f"Ocurrió un error procesando tu consulta: {str(e)}",
                "chunks_used": [],
                "confidence_score": 0.0,
                "routing": {"domain": "error", "confidence": 0.0, "reasoning": str(e)},
                "error": str(e),
            }
            if trace:
                trace.update(output={"error": str(e)})
            return error_result
