import logging

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(
            model_path="models/mistral-7b-instruct-v0.3-q4_k_m.gguf",
            n_ctx=1024,
            n_threads=8,
            n_gpu_layers=20,
        )
        logger.info("Modelo Mistral cargado")
    return _llm


def _build_prompt(keywords: list, docs: list) -> str:
    keywords_str = ", ".join(
        f"{kw['termino']} ({kw.get('score_tfidf', 0):.2f})"
        if isinstance(kw, dict) else str(kw)
        for kw in keywords[:15]
    )
    docs_str = "\n".join(
        d.get("text", str(d)) if isinstance(d, dict) else str(d)
        for d in docs[:3]
    )
    return (
        "[INST]\n"
        "Eres experto en análisis de tópicos.\n\n"
        "Genera un nombre corto (máximo 5 palabras).\n\n"
        f"Keywords: {keywords_str}\n"
        f"Docs: {docs_str}\n\n"
        "Responde SOLO con el nombre.\n"
        "[/INST]"
    )


def _query_mistral_local(prompt: str, max_tokens: int = 20) -> str:
    llm = _get_llm()
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["\n", "Topic:", "Cluster:"],
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        logger.error("Error en modelo local: %s", e)
        return "unknown_topic"


def name_all_clusters(
    keywords_por_cluster: dict,
    docs_por_cluster: dict,
) -> dict[int, str]:
    cluster_names: dict[int, str] = {}
    for cluster_id, keywords in keywords_por_cluster.items():
        if not keywords:
            cluster_names[cluster_id] = "unknown"
            continue
        docs = docs_por_cluster.get(cluster_id, [])
        prompt = _build_prompt(keywords, docs)
        name = _query_mistral_local(prompt).replace("\n", " ").strip()
        cluster_names[cluster_id] = name
        logger.info("Cluster %d -> %s", cluster_id, name)
    return cluster_names


def name_single_cluster(keywords: list, docs: list) -> str:
    prompt = _build_prompt(keywords, docs)
    return _query_mistral_local(prompt)
