"""
Polaridades

4 paneles:
    A (top-left)  : Barras apiladas +/neutro/- por destino
    B (top-right) : Radar de métricas de sentimiento por destino
    C (bot-left)  : Lollipop top-10 tópicos más positivos
    D (bot-right) : Lollipop top-10 tópicos más negativos
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Params, Paths

logger = logging.getLogger(__name__)

PATH_DEST  = Paths.SENTIMIENTO_POR_DESTINO_CSV
PATH_TOPIC = Paths.SENTIMIENTO_POR_TOPICO_CSV

# Params.DEST_ORDER from Params but with a specific display order for this chart
# DEST_LABELS uses \n for multiline chart labels (display-only format)
DEST_LABELS  = ["La Paz", "Huatulco", "Riviera\nNayarit", "Puerto\nVallarta", "Riviera\nMaya"]


def _clean_name(raw: str, max_w: int = 4) -> str:
    parts = str(raw).split("_")
    clean = [p for p in parts[1:] if p and not p.startswith("ent")]
    return " ".join(clean[:max_w]).capitalize()


def run_polarities() -> None:
    Paths.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    df_dest  = pd.read_csv(PATH_DEST,  encoding="utf-8-sig")
    df_topic = pd.read_csv(PATH_TOPIC, encoding="utf-8-sig")
    df_dest["location"] = df_dest["location"].str.strip("'")
    df_dest = df_dest.set_index("location").reindex(Params.DEST_ORDER)

    fig = plt.figure(figsize=(18, 13), facecolor="white")
    fig.suptitle("Análisis de Polaridades — 5 Destinos Turísticos",
                 fontsize=18, fontweight="bold", color="#1a1a2e", y=0.97)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.93, bottom=0.07, left=0.06, right=0.97,
                           hspace=0.50, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#f7f9fc")

    pct_pos = df_dest["pct_positivo"].values
    pct_neu = df_dest["pct_neutro"].values
    pct_neg = df_dest["pct_negativo"].values
    x = np.arange(len(Params.DEST_ORDER))
    w = 0.55

    ax_a.bar(x, pct_pos, w, label="Positivo", color=Params.VIZ_SENTIMENT_POS_COLOR, alpha=0.85)
    ax_a.bar(x, pct_neu, w, bottom=pct_pos, label="Neutro", color=Params.VIZ_SENTIMENT_NEU_COLOR, alpha=0.85)
    ax_a.bar(x, pct_neg, w, bottom=pct_pos + pct_neu, label="Negativo", color=Params.VIZ_SENTIMENT_NEG_COLOR, alpha=0.85)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(DEST_LABELS, fontsize=10, color="#333")
    ax_a.set_ylabel("% comentarios (con rating)", fontsize=9, color="#777")
    ax_a.set_title("Distribución de Sentimiento por Destino",
                   fontsize=12, fontweight="bold", color="#1a1a2e", pad=9)
    ax_a.yaxis.grid(True, color="#ebebeb", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.tick_params(axis="y", colors="#aaa", labelsize=8)
    for sp in ax_a.spines.values():
        sp.set_edgecolor("#ddd")
    ax_a.legend(fontsize=9, facecolor="white", edgecolor="#ccc")
    ax_a.set_ylim(0, 115)

    for i, (p, n, g) in enumerate(zip(pct_pos, pct_neu, pct_neg)):
        ax_a.text(i, p / 2,         f"{p:.0f}%", ha="center", va="center",
                  fontsize=8.5, fontweight="bold", color="white")
        if n > 5:
            ax_a.text(i, p + n / 2,   f"{n:.0f}%", ha="center", va="center",
                      fontsize=7.5, color="#222")
        if g > 3:
            ax_a.text(i, p + n + g / 2, f"{g:.0f}%", ha="center", va="center",
                      fontsize=7.5, fontweight="bold", color="white")

    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    ax_b.set_facecolor("#f7f9fc")

    metrics = ["Sent. Medio", "% Positivo", "Estrellas\n(media)", "% No Negativo"]
    n_m = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
    angles += angles[:1]

    for dest, dlabel, dc in zip(Params.DEST_ORDER, DEST_LABELS, Params.DEST_COLORS):
        row = df_dest.loc[dest]
        v = [
            (row["sentimiento_medio"] + 1) / 2,
            row["pct_positivo"] / 100,
            (row["estrella_media"] - 1) / 4,
            (100 - row["pct_negativo"]) / 100,
        ]
        v += v[:1]
        ax_b.plot(angles, v, "o-", linewidth=1.8, color=dc,
                  label=dlabel.replace("\n", " "), markersize=5)
        ax_b.fill(angles, v, alpha=0.10, color=dc)

    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(metrics, fontsize=9, color="#333")
    ax_b.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_b.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.5, color="#999")
    ax_b.set_title("Radar de Métricas de Sentimiento",
                   fontsize=12, fontweight="bold", color="#1a1a2e", pad=18)
    ax_b.grid(color="#ddd", linewidth=0.6)
    ax_b.spines["polar"].set_edgecolor("#ccc")
    ax_b.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.38, 1.15),
                facecolor="white", edgecolor="#ccc")

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#f7f9fc")

    df_v = df_topic[df_topic["total_con_rating"] >= 10].copy()
    df_v["clean"] = df_v["topic_name"].apply(_clean_name)

    top_pos = df_v.nlargest(10, "sentimiento_medio")[["clean", "sentimiento_medio", "total_con_rating"]]
    lbls_p = [f"{r['clean'][:30]} (n={int(r['total_con_rating'])})" for _, r in top_pos.iterrows()]
    vals_p = top_pos["sentimiento_medio"].tolist()

    y = np.arange(len(vals_p))
    ax_c.hlines(y, 0, vals_p, colors=Params.VIZ_SENTIMENT_POS_COLOR, linewidth=2.2, alpha=0.7)
    ax_c.scatter(vals_p, y, color=Params.VIZ_SENTIMENT_POS_COLOR, s=65, zorder=3)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(lbls_p, fontsize=7.5, color="#333")
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Sentimiento medio", fontsize=9, color="#777")
    ax_c.set_title("Top 10 Tópicos más Positivos",
                   fontsize=12, fontweight="bold", color="#1a1a2e", pad=9)
    ax_c.axvline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax_c.xaxis.grid(True, color="#ebebeb", linewidth=0.5)
    ax_c.set_axisbelow(True)
    ax_c.tick_params(axis="x", colors="#aaa", labelsize=8)
    for sp in ax_c.spines.values():
        sp.set_edgecolor("#ddd")

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#f7f9fc")

    top_neg = df_v.nsmallest(10, "sentimiento_medio")[["clean", "sentimiento_medio", "total_con_rating"]]
    lbls_n = [f"{r['clean'][:30]} (n={int(r['total_con_rating'])})" for _, r in top_neg.iterrows()]
    vals_n = top_neg["sentimiento_medio"].tolist()

    ax_d.hlines(np.arange(len(vals_n)), vals_n, 0, colors=Params.VIZ_SENTIMENT_NEG_COLOR, linewidth=2.2, alpha=0.7)
    ax_d.scatter(vals_n, np.arange(len(vals_n)), color=Params.VIZ_SENTIMENT_NEG_COLOR, s=65, zorder=3)
    ax_d.set_yticks(np.arange(len(lbls_n)))
    ax_d.set_yticklabels(lbls_n, fontsize=7.5, color="#333")
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Sentimiento medio", fontsize=9, color="#777")
    ax_d.set_title("Top 10 Tópicos más Negativos",
                   fontsize=12, fontweight="bold", color="#1a1a2e", pad=9)
    ax_d.axvline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax_d.xaxis.grid(True, color="#ebebeb", linewidth=0.5)
    ax_d.set_axisbelow(True)
    ax_d.tick_params(axis="x", colors="#aaa", labelsize=8)
    for sp in ax_d.spines.values():
        sp.set_edgecolor("#ddd")

    out = Paths.VISUALIZATION_DIR / "polarities.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("polarities.png -> %s", out)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    run_polarities()
