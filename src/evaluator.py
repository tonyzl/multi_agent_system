"""
evaluator.py — Evaluador del Sistema Multi-Agente usando la Score API de Langfuse.

Métricas evaluadas por un LLM juez (LLM-as-judge):
1. correctness      — ¿La respuesta es factualmente correcta?
2. relevance        — ¿La respuesta aborda la pregunta del usuario?
3. completeness     — ¿La respuesta es suficientemente completa?
4. routing_accuracy — ¿El agente correcto respondió la pregunta?

Cada métrica se puntúa de 0 a 1 y se registra en Langfuse via Score API.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI


EVALUATOR_SYSTEM_PROMPT = """Eres un evaluador experto de respuestas de sistemas de IA.
Tu tarea es evaluar la calidad de la respuesta de un asistente corporativo
basándote en criterios específicos.

Para cada evaluación responde ÚNICAMENTE con un JSON válido (sin markdown):
{
  "correctness": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "correctness_reason": "<una frase>",
  "relevance_reason": "<una frase>",
  "completeness_reason": "<una frase>",
  "overall_comment": "<comentario general en 1-2 oraciones>"
}

Criterios de evaluación:
- correctness (0-1): ¿La información proporcionada es factualmente precisa y no contiene errores?
  0.0 = Información incorrecta o inventada
  0.5 = Parcialmente correcta con algunos errores
  1.0 = Completamente correcta y precisa

- relevance (0-1): ¿La respuesta aborda directamente lo que el usuario preguntó?
  0.0 = No responde la pregunta en absoluto
  0.5 = Responde parcialmente o de forma indirecta
  1.0 = Responde directamente y de forma completa a la pregunta

- completeness (0-1): ¿La respuesta incluye toda la información necesaria y útil?
  0.0 = Respuesta muy incompleta o superficial
  0.5 = Respuesta básica pero le falta información importante
  1.0 = Respuesta completa con todos los detalles relevantes
"""


class MultiAgentEvaluator:
    """
    Evalúa las respuestas del sistema multi-agente usando LLM-as-judge
    y registra los scores en Langfuse.
    """

    def __init__(
        self,
        openai_client: OpenAI,
        langfuse_client=None,
        judge_model: str = None,
    ):
        self.client = openai_client
        self.langfuse = langfuse_client
        self.judge_model = judge_model or os.getenv("CHAT_MODEL", "gpt-4o-mini")

    def _llm_judge(
        self,
        question: str,
        answer: str,
        context_chunks: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Usa un LLM para evaluar la calidad de la respuesta."""
        context_text = ""
        if context_chunks:
            context_text = "\n\nContexto utilizado por el agente:\n" + "\n---\n".join(
                c.get("text", "")[:300] for c in context_chunks
            )

        user_prompt = (
            f"Pregunta del usuario: {question}\n\n"
            f"Respuesta del asistente: {answer}"
            f"{context_text}"
        )

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "correctness": 0.5,
                "relevance": 0.5,
                "completeness": 0.5,
                "correctness_reason": "Error al parsear evaluación",
                "relevance_reason": "Error al parsear evaluación",
                "completeness_reason": "Error al parsear evaluación",
                "overall_comment": f"Parse error: {raw[:100]}",
            }

    def _evaluate_routing(
        self,
        predicted_domain: str,
        expected_domain: str,
    ) -> float:
        """Evalúa si el routing fue correcto."""
        if expected_domain == "unknown":
            return 1.0 if predicted_domain == "unknown" else 0.0
        return 1.0 if predicted_domain == expected_domain else 0.0

    def evaluate(
        self,
        result: Dict[str, Any],
        expected_domain: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evalúa una respuesta del sistema y registra scores en Langfuse.

        Args:
            result: Respuesta completa del orquestador.
            expected_domain: Dominio esperado para evaluar routing accuracy.
            trace_id: ID de traza de Langfuse para vincular los scores.

        Returns:
            Diccionario con todos los scores de evaluación.
        """
        question = result.get("question", "")
        answer = result.get("answer", "")
        predicted_domain = result.get("domain", "unknown")
        chunks = result.get("chunks_used", [])

        # 1. LLM-as-judge para calidad de respuesta
        llm_scores = self._llm_judge(question, answer, chunks)

        # 2. Routing accuracy
        routing_score = None
        if expected_domain is not None:
            routing_score = self._evaluate_routing(predicted_domain, expected_domain)

        # 3. Score compuesto
        quality_metrics = [
            llm_scores.get("correctness", 0.5),
            llm_scores.get("relevance", 0.5),
            llm_scores.get("completeness", 0.5),
        ]
        overall_score = sum(quality_metrics) / len(quality_metrics)

        evaluation = {
            "question": question,
            "predicted_domain": predicted_domain,
            "expected_domain": expected_domain,
            "scores": {
                "correctness": llm_scores.get("correctness", 0.5),
                "relevance": llm_scores.get("relevance", 0.5),
                "completeness": llm_scores.get("completeness", 0.5),
                "routing_accuracy": routing_score,
                "overall": round(overall_score, 3),
            },
            "reasons": {
                "correctness": llm_scores.get("correctness_reason", ""),
                "relevance": llm_scores.get("relevance_reason", ""),
                "completeness": llm_scores.get("completeness_reason", ""),
            },
            "overall_comment": llm_scores.get("overall_comment", ""),
        }

        # 4. Registrar scores en Langfuse
        if self.langfuse and trace_id:
            self._send_scores_to_langfuse(evaluation, trace_id)

        return evaluation

    def _send_scores_to_langfuse(
        self,
        evaluation: Dict[str, Any],
        trace_id: str,
    ) -> None:
        """Envía todos los scores a Langfuse via Score API."""
        scores_to_send = {
            "correctness": evaluation["scores"]["correctness"],
            "relevance": evaluation["scores"]["relevance"],
            "completeness": evaluation["scores"]["completeness"],
            "overall_quality": evaluation["scores"]["overall"],
        }

        if evaluation["scores"]["routing_accuracy"] is not None:
            scores_to_send["routing_accuracy"] = evaluation["scores"]["routing_accuracy"]

        for score_name, score_value in scores_to_send.items():
            try:
                self.langfuse.score(
                    trace_id=trace_id,
                    name=score_name,
                    value=score_value,
                    comment=evaluation["reasons"].get(score_name, evaluation.get("overall_comment", "")),
                )
            except Exception as e:
                print(f"⚠ Error enviando score '{score_name}' a Langfuse: {e}")

    def batch_evaluate(
        self,
        results: List[Dict[str, Any]],
        test_queries: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Evalúa un lote de resultados y retorna métricas agregadas.

        Args:
            results: Lista de respuestas del orquestador.
            test_queries: Lista de test queries con expected_domain.

        Returns:
            Diccionario con métricas agregadas del lote.
        """
        # Crear mapa de expected_domain por pregunta
        expected_map = {}
        if test_queries:
            for q in test_queries:
                expected_map[q["query"]] = q.get("expected_domain")

        evaluations = []
        print(f"\n🔍 Evaluando {len(results)} respuestas...")

        for i, result in enumerate(results, 1):
            question = result.get("question", "")
            expected = expected_map.get(question)

            print(f"   [{i}/{len(results)}] Evaluando: {question[:60]}...")
            eval_result = self.evaluate(result, expected_domain=expected)
            evaluations.append(eval_result)

        # Calcular métricas agregadas
        def avg(key):
            vals = [e["scores"][key] for e in evaluations if e["scores"].get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        summary = {
            "total_evaluated": len(evaluations),
            "aggregate_scores": {
                "correctness": avg("correctness"),
                "relevance": avg("relevance"),
                "completeness": avg("completeness"),
                "routing_accuracy": avg("routing_accuracy"),
                "overall": avg("overall"),
            },
            "evaluations": evaluations,
        }

        print("\n📊 Resumen de Evaluación:")
        print(f"   Correctness:      {summary['aggregate_scores']['correctness']:.3f}")
        print(f"   Relevance:        {summary['aggregate_scores']['relevance']:.3f}")
        print(f"   Completeness:     {summary['aggregate_scores']['completeness']:.3f}")
        if summary['aggregate_scores']['routing_accuracy'] is not None:
            print(f"   Routing Accuracy: {summary['aggregate_scores']['routing_accuracy']:.3f}")
        print(f"   Overall Quality:  {summary['aggregate_scores']['overall']:.3f}")

        return summary
