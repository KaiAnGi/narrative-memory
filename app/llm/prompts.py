"""Construccion de prompts para la respuesta.

Ajuste #3: el modelo puede inferir y razonar SOBRE el contexto recuperado,
pero tiene prohibido introducir acontecimientos no respaldados por el.
Flujo: evidencia -> razonamiento -> conclusion, citando capitulos.
"""
from app.models.schemas import SearchHit

SYSTEM_PROMPT = (
    "Eres un analista literario. Solo dispones de los fragmentos de la novela "
    "que se te entregan como contexto.\n\n"
    "Reglas:\n"
    "1. Fundamenta tu respuesta en los fragmentos proporcionados (evidencia).\n"
    "2. Puedes razonar, inferir y comparar acontecimientos de distintos capítulos "
    "siempre que lo apoyes en esa evidencia.\n"
    "3. NO inventes acontecimientos, diálogos, citas ni hechos que no estén "
    "respaldados por los fragmentos.\n"
    "4. Si la evidencia recuperada no basta para responder, dilo explícitamente "
    "y señala qué información faltaría.\n"
    "5. Estructura la respuesta en tres partes: Evidencia, Razonamiento y Conclusión.\n"
    "6. Cita los capítulos de los que tomas cada evidencia."
)


def build_qa_messages(question: str, hits: list[SearchHit]) -> list[dict]:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = (
            f"[Fragmento {i} · Capítulo {chunk.chapter_index} "
            f"· posición global {chunk.global_position}]"
        )
        blocks.append(f"{header}\n{chunk.text}")

    context = "\n\n".join(blocks) if blocks else "(No se recuperó ningún fragmento.)"
    user = f"Contexto recuperado:\n{context}\n\nPregunta: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
