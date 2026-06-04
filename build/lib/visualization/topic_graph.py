"""
Visualización 1 — Mapa de Tópicos
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Paths, Params

logger = logging.getLogger(__name__)

PATH_PROJ = Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_PROYECCION_FILE
PATH_ETIQ = Paths.CLUSTERING_EMBEDDINGS_DIR / Paths.CLUSTERING_ETIQUETAS_FILE
PATH_KW   = Paths.ENRICHMENT_KEYWORDS_CSV


def run_topic_graph() -> None:
    Paths.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    missing = [p for p in [PATH_PROJ, PATH_ETIQ, PATH_KW] if not p.exists()]
    if missing:
        for p in missing:
            logger.error("Archivo no encontrado: %s", p)
        raise FileNotFoundError(f"Faltan archivos: {[str(p) for p in missing]}")

    k   = Params.CLUSTERING_ACTIVO_K
    key = Params.CLUSTERING_ACTIVO_KEY

    X = np.load(PATH_PROJ)
    with open(PATH_ETIQ, encoding="utf-8") as f:
        etiq = json.load(f)

    if key not in etiq:
        kmeans_keys = [k_ for k_ in etiq if k_.startswith("kmeans")]
        if not kmeans_keys:
            raise KeyError(f"Clave '{key}' no encontrada en etiquetas_mejores.json y no hay modelos kmeans disponibles")
        key = kmeans_keys[0]
        k = int(key.split("k=")[1])
        logger.info("Clave '%s' no en json — usando '%s' (k=%d)", Params.CLUSTERING_ACTIVO_KEY, key, k)

    labels = np.array(etiq[key])

    df_kw = pd.read_csv(PATH_KW, encoding="utf-8-sig")
    cluster_names = {}
    for c in range(k):
        kws = df_kw[df_kw["cluster_id"] == c].head(3)["termino"].tolist()
        cluster_names[c] = " · ".join(kws)

    rng   = np.random.default_rng(Params.RANDOM_STATE)
    per_c = Params.VIZ_SCATTER_SAMPLE_N // k
    idx   = []
    for c in range(k):
        ci = np.where(labels == c)[0]
        idx.extend(rng.choice(ci, min(per_c, len(ci)), replace=False))
    idx = np.array(idx)
    xs = X[idx, 0]
    ys = X[idx, 1]
    ls = labels[idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")

    for c in range(k):
        m = ls == c
        ax.scatter(xs[m], ys[m], c=Params.CLUSTER_COLORS[c % len(Params.CLUSTER_COLORS)], s=6, alpha=0.55,
                   linewidths=0, rasterized=True)

    for c in range(k):
        ci = idx[ls == c]
        if len(ci) == 0:
            continue
        cx, cy = X[ci, 0].mean(), X[ci, 1].mean()
        short = " / ".join(cluster_names[c].split(" · ")[:2])
        ax.annotate(
            f"C{c}: {short}", (cx, cy),
            fontsize=8.5, fontweight="bold", color="white", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=Params.CLUSTER_COLORS[c % len(Params.CLUSTER_COLORS)], ec="none", alpha=0.90),
        )

    ax.set_xlabel("UMAP 1", fontsize=11, color="#555")
    ax.set_ylabel("UMAP 2", fontsize=11, color="#555")
    ax.set_title(
        f"Mapa de Tópicos — Proyección UMAP  (KMeans k={k}, n=54 587 docs)",
        fontsize=14, fontweight="bold", color="#1a1a2e", pad=14,
    )
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ddd")
    ax.grid(True, color="#e5e5e5", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=Params.CLUSTER_COLORS[c % len(Params.CLUSTER_COLORS)], label=f"C{c}: {cluster_names[c]}") for c in range(k)]
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              framealpha=0.95, facecolor="white", edgecolor="#ccc")

    plt.tight_layout()
    out = Paths.VISUALIZATION_DIR / "topic_graph.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("topic_graph.png -> %s", out)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    run_topic_graph()
