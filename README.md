# 🤖 Sistema Multi-Agente TechCorp

Sistema de múltiples agentes especializados con RAG (Retrieval-Augmented Generation) para responder consultas corporativas de Recursos Humanos, Tecnología y Finanzas. Incluye enrutamiento inteligente con LLM, observabilidad con Langfuse y evaluación automática de calidad.

---

## 🏗️ Arquitectura

```
Usuario
  │
  ▼
Orquestador
(Routing LLM → Clasifica intención)
  │
  ├──→ 🧑‍💼 HR Agent      (RAG sobre hr_docs/)
  ├──→ 💻 Tech Agent    (RAG sobre tech_docs/)
  ├──→ 💰 Finance Agent (RAG sobre finance_docs/)
  └──→ ❓ Unknown       (Fuera de dominio)
         │
         ▼
    Langfuse (Trazas, latencias, costos, scores)
         │
         ▼
    Evaluador (LLM-as-Judge → Score API de Langfuse)
```

## 📁 Estructura del Proyecto

```
multi_agent_system/
├── multi_agent_system.ipynb       # Notebook principal (punto de entrada)
├── requirements.txt               # Dependencias
├── .env.example                   # Plantilla de variables de entorno
├── test_queries.json              # 18 consultas de prueba con dominio esperado
├── README.md
│
├── src/
│   ├── multi_agent_system.py      # Módulo central: chunking, embeddings, init
│   ├── evaluator.py               # Evaluador LLM-as-Judge + Langfuse Score API
│   └── agents/
│       ├── orchestrator.py        # Orquestador y clasificador de intención
│       ├── hr_agent.py            # Agente de Recursos Humanos
│       ├── tech_agent.py          # Agente de Tecnología/IT
│       └── finance_agent.py       # Agente de Finanzas
│
├── data/
│   ├── hr_docs/                   # Base de conocimiento de RRHH
│   │   ├── politicas_generales.txt
│   │   ├── faq_rrhh.txt
│   │   └── guia_reclutamiento_capacitacion.txt
│   ├── tech_docs/                 # Base de conocimiento técnica
│   │   ├── it_knowledge_base.txt
│   │   ├── arquitectura_estandares.txt
│   │   └── troubleshooting_faq.txt
│   └── finance_docs/              # Base de conocimiento financiera
│       ├── politicas_gastos.txt
│       ├── faq_finanzas.txt
│       └── contabilidad_inversiones.txt
│
└── outputs/                       # Generado automáticamente
    ├── hr_index.json              # Vector store HR
    ├── tech_index.json            # Vector store Tech
    ├── finance_index.json         # Vector store Finance
    ├── test_results.json          # Resultados de test queries
    └── evaluation_results.json    # Resultados del evaluador
```

---

## ⚙️ Instalación

### 1. Clonar / descomprimir el proyecto

```bash
cd multi_agent_system
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar API Keys

Copia `.env.example` como `.env` y completa los valores:

```bash
cp .env.example .env
```

Edita `.env`:

```env
# Obligatorio
OPENAI_API_KEY=sk-...

# Opcional — para observabilidad
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> **Langfuse es opcional.** El sistema funciona completamente sin él. Si deseas activarlo, regístrate gratis en [cloud.langfuse.com](https://cloud.langfuse.com).

---

## 🚀 Cómo Ejecutar el Notebook

### Abrir el notebook

```bash
jupyter notebook multi_agent_system.ipynb
# ó
jupyter lab multi_agent_system.ipynb
```

### Orden de ejecución de celdas

Ejecuta las secciones **en orden**:

| Sección | Qué hace | Llama a API |
|---------|----------|-------------|
| **1. Setup e Imports** | Carga `.env`, inicializa cliente OpenAI | ✅ Verifica conexión |
| **2. Carga de Documentos** | Construye vector stores con embeddings | ✅ Solo la primera vez |
| **3. Definición de Agentes** | Inicializa HR, Tech y Finance Agent | ❌ |
| **4. Orquestador** | Inicializa el router + demos de routing | ✅ Por cada query |
| **5. Pruebas** | Ejecuta las 18 test queries | ✅ 18 llamadas |
| **6. Langfuse** | Integra observabilidad | ✅ Si configurado |
| **7. Evaluador (BONUS)** | LLM-as-Judge + Score API | ✅ Por cada evaluación |

> **Tip:** En la **Sección 2**, los vector stores se guardan en `outputs/`. La segunda vez que ejecutes el notebook, se cargan desde disco sin costo adicional de embeddings. Usa `force_rebuild=True` solo si cambias los documentos.

---

## 💡 Ejemplos de Uso

### Desde el notebook

```python
# Consulta de RRHH
result = orchestrator.route_and_answer("¿Cuántos días de vacaciones tengo?")
print(result["answer"])

# Consulta técnica
result = orchestrator.route_and_answer("¿Cómo me conecto a la VPN?")
print(result["answer"])

# Consulta financiera
result = orchestrator.route_and_answer("¿Cómo pido un reembolso en Expensify?")
print(result["answer"])

# Caso fuera de dominio
result = orchestrator.route_and_answer("¿Qué película recomiendas?")
print(result["domain"])  # → "unknown"
```

### Usar agentes directamente

```python
# Sin pasar por el orquestador
hr_result = hr_agent.answer("¿Cómo funciona el bono anual?")
tech_result = tech_agent.answer("¿Qué stack tecnológico usa TechCorp?")
finance_result = finance_agent.answer("¿Cuál es el límite de gasto por hotel?")
```

### Evaluar una respuesta

```python
eval_result = evaluator.evaluate(
    result=result,
    expected_domain="hr",
)
print(eval_result["scores"])
# → {'correctness': 0.9, 'relevance': 0.95, 'completeness': 0.85, 'overall': 0.9}
```

---

## 📊 Test Queries

El archivo `test_queries.json` contiene 18 consultas de prueba que cubren:

- ✅ 6 consultas de **RRHH** (vacaciones, beneficios, desempeño, onboarding, offboarding)
- ✅ 5 consultas de **Tecnología** (VPN, Git, Docker, seguridad, incidentes)
- ✅ 5 consultas de **Finanzas** (reembolsos, proveedores, compliance, gastos)
- ✅ 2 **casos borde** (fuera de dominio, posible soborno)

---

## ⚙️ Notas de Configuración

### Modelos

Los modelos se pueden cambiar en `.env`:

```env
EMBEDDING_MODEL=text-embedding-3-small   # Más económico
CHAT_MODEL=gpt-4o-mini                   # Más económico / gpt-4o para mayor calidad
ROUTING_MODEL=gpt-4o-mini
```

### Parámetros de chunking

```env
CHUNK_SIZE=1000       # Caracteres por chunk (aumentar = más contexto, menos precisión)
CHUNK_OVERLAP=150     # Solapamiento entre chunks
TOP_K=4               # Chunks recuperados por consulta
```

### Reconstruir vector stores

Si modificas los documentos en `data/`, reconstruye los índices:

```python
# En la Sección 2 del notebook
vector_stores = build_all_vector_stores(client=openai_client, force_rebuild=True)
```

---

## ⚠️ Limitaciones Conocidas

1. **Vector store en JSON**: Los embeddings se guardan en archivos JSON locales. Para producción se recomendaría usar una base de datos vectorial como ChromaDB, Pinecone o pgvector.

2. **Sin memoria de conversación**: El sistema responde preguntas individuales sin historial de conversación. Cada consulta es independiente.

3. **Routing basado en texto**: El orquestador clasifica por contenido semántico. Preguntas muy ambiguas pueden ser mal clasificadas.

4. **Latencia**: Cada consulta realiza al menos 2 llamadas a la API de OpenAI (embedding + generación). Considera caching para consultas frecuentes.

5. **Costo de embeddings**: La primera ejecución genera embeddings para todos los chunks (~150+ chunks). Las siguientes ejecuciones cargan desde disco.

6. **Idioma**: El sistema está optimizado para español. Las consultas en inglés funcionan pero con menor calidad de respuesta.

---

## 📞 Contacto

Para consultas sobre el sistema: hr@techcorp.com | it-support@techcorp.com | finance@techcorp.com
