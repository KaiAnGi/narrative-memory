"""Construccion de prompts para la respuesta.

Ajuste #3: el modelo puede inferir y razonar SOBRE el contexto recuperado,
pero tiene prohibido introducir acontecimientos no respaldados por el.
Flujo: evidencia -> razonamiento -> conclusion, citando capitulos.

Fase 2D: variante de grounding estricto para aislar si el prompt puede arreglar
los fallos de generacion (mismo modelo, mismo contexto, distinto system prompt).
Ninguno de estos prompts contiene datos de ninguna obra concreta.
"""
from pathlib import Path

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

GROUNDING_SYSTEM_PROMPT = (
    "Eres un analista literario. Tu UNICA fuente de informacion son los fragmentos "
    "de la novela incluidos en el contexto. No conoces la obra de ninguna otra "
    "forma: ni por el titulo, ni por el genero, ni por datos externos.\n\n"
    "Metodo obligatorio, en este orden:\n"
    "1. Identifica primero la evidencia: determina que fragmentos del contexto "
    "responden a la pregunta. Si varios aportan, combinalos; si ninguno la "
    "responde del todo, identifica el fragmento mas cercano y que falta.\n"
    "2. Distingue hechos de inferencias: un hecho debe estar escrito "
    "explicitamente en el contexto; una inferencia es una deduccion tuya y debe "
    "llevar la marca 'inferencia:' delante. Nunca presentes una inferencia como "
    "si fuera un hecho del texto.\n"
    "3. Prohibido el conocimiento externo: no rellenes ningun hueco con datos que "
    "no esten en los fragmentos, aunque parezcan obvios o tipicos del genero.\n"
    "4. Evidencia insuficiente: si el contexto no permite responder con seguridad, "
    "dilo explicitamente, indica exactamente que informacion faltaria y termina "
    "ahi. No improvises una respuesta.\n"
    "5. Prohibidas las respuestas genericas: si el contexto contiene detalles "
    "concretos sobre lo que se pregunta, responde con ellos. Responder con "
    "generalidades cuando existe evidencia especifica se considera un error.\n"
    "6. Cita siempre las fuentes: por cada afirmacion indica el fragmento y el "
    "capitulo de donde la tomas (p. ej. '[Fragmento 3, capitulo 7]').\n\n"
    "Estructura la respuesta en: Evidencia (que dice el texto y donde), "
    "Razonamiento (solo lo que derives; marca cada inferencia) y Conclusion."
)

PROMPTS = {
    "baseline": SYSTEM_PROMPT,
    "grounding": GROUNDING_SYSTEM_PROMPT,
}


def resolve_system_prompt(name_or_path: str) -> str:
    """Devuelve el system prompt por nombre ('baseline'/'grounding') o por ruta a archivo."""
    if name_or_path in PROMPTS:
        return PROMPTS[name_or_path]
    path = Path(name_or_path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise ValueError(
        f"prompt desconocido: {name_or_path} (usa {sorted(PROMPTS)} o una ruta a archivo)"
    )


def build_qa_messages(
    question: str,
    hits: list[SearchHit],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
