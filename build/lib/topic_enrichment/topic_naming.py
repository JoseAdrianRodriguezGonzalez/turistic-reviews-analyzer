import logging

from config import Params, Paths

logger = logging.getLogger(__name__)

_llm = None


def _resolve_gpu_layers() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return Params.TOPIC_NAMING_N_GPU_LAYERS
        logger.info('GPU no disponible — topic naming usará CPU (n_gpu_layers=0)')
    except ImportError:
        pass
    return 0


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        n_gpu = _resolve_gpu_layers()
        _llm = Llama(
            model_path=str(Paths.LLAMA_MODEL_PATH),
            n_ctx=Params.TOPIC_NAMING_N_CTX,
            n_threads=Params.TOPIC_NAMING_N_THREADS,
            n_gpu_layers=n_gpu,
        )
        logger.info(
            'Modelo Mistral cargado desde %s (n_gpu_layers=%d)',
            Paths.LLAMA_MODEL_PATH, n_gpu,
        )
    return _llm


def _build_prompt(keywords: list, docs: list) -> str:
    keywords_str = ", ".join(
        f"{kw['termino']} ({kw.get('score_tfidf', 0):.2f})"
        if isinstance(kw, dict) else str(kw)
        for kw in keywords[:Params.TOP_N_KEYWORDS]
    )
    docs_str = "\n".join(
        d.get("text", str(d)) if isinstance(d, dict) else str(d)
        for d in docs[:Params.TOPIC_NAMING_TOP_DOCS]
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


def _query_mistral_local(prompt: str) -> str:
    llm = _get_llm()
    try:
        output = llm(
            prompt,
            max_tokens=Params.TOPIC_NAMING_MAX_TOKENS,
            temperature=Params.TOPIC_NAMING_TEMPERATURE,
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
