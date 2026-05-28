import logging

import pandas as pd
import plotly.graph_objects as go

from config import Paths

logger = logging.getLogger(__name__)

PATH_TD = Paths.SENTIMIENTO_POR_TOPICO_DESTINO_CSV
PATH_T  = Paths.SENTIMIENTO_POR_TOPICO_CSV
OUTPUT  = Paths.VISUALIZATION_DIR / "polarities_2.html"

TOP_N        = 25
MIN_DOCS_ROW = 10

DEST_LABELS = {
    "huatulco"       : "Huatulco",
    "la_paz"         : "La Paz",
    "puerto_vallarta": "Puerto Vallarta",
    "riviera_maya"   : "Riviera Maya",
    "riviera_nayarit": "Riviera Nayarit",
}


def _clean_name(raw: str) -> str:
    parts = str(raw).split("_")
    tokens = [p for p in parts[1:] if p and not p.startswith("ent")]
    label = " ".join(tokens[:4])
    return label.capitalize() if label else str(raw)


def run_polarities_heatmap() -> None:
    Paths.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    df_td = pd.read_csv(PATH_TD, encoding="utf-8-sig")
    df_t  = pd.read_csv(PATH_T,  encoding="utf-8-sig")
    df_t["clean_name"] = df_t["topic_name"].apply(_clean_name)
    name_map = df_t.set_index("topic")["clean_name"].to_dict()

    topic_docs = df_td.groupby("topic")["n_docs"].sum()
    top_topics = topic_docs[topic_docs >= MIN_DOCS_ROW].nlargest(TOP_N).index.tolist()

    df_filt = df_td[df_td["topic"].isin(top_topics)].copy()
    df_filt["topic_name"] = df_filt["topic"].map(name_map).fillna(df_filt["topic"].astype(str))

    df_pivot = df_filt.pivot(index="topic_name", columns="location", values="sentimiento_medio")
    df_pivot.columns = [DEST_LABELS.get(c, c) for c in df_pivot.columns]

    df_pivot["_mean"] = df_pivot.mean(axis=1, skipna=True)
    df_pivot = df_pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

    df_ndocs = df_filt.pivot(index="topic_name", columns="location", values="n_docs")
    df_ndocs.columns = [DEST_LABELS.get(c, c) for c in df_ndocs.columns]
    df_ndocs = df_ndocs.reindex(df_pivot.index)

    text_matrix = []
    for row_name in df_pivot.index:
        row_vals = []
        for col_name in df_pivot.columns:
            val  = df_pivot.loc[row_name, col_name]
            ndoc = df_ndocs.loc[row_name, col_name] if col_name in df_ndocs.columns else None
            if pd.isna(val):
                row_vals.append("—")
            else:
                n_str = f"<br><span style='font-size:9px'>n={int(ndoc)}</span>" if not pd.isna(ndoc) else ""
                row_vals.append(f"{val:+.2f}{n_str}")
        text_matrix.append(row_vals)

    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns.tolist(),
        y=df_pivot.index.tolist(),
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        colorbar=dict(
            title="Sentimiento<br>Medio",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["−1 Muy neg.", "−0.5", "0 Neutro", "+0.5", "+1 Muy pos."],
            thickness=18,
        ),
        hoverongaps=False,
        hovertemplate=(
            "<b>Tópico:</b> %{y}<br>"
            "<b>Destino:</b> %{x}<br>"
            "<b>Sentimiento:</b> %{z:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Polaridad de Sentimiento: Top {TOP_N} Tópicos × Destino<br>"
                "<sup style='color:#888'>Solo tópicos con ≥10 comentarios con rating | "
                "Celdas '—' = sin datos suficientes</sup>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=16),
        ),
        xaxis=dict(title="Destino Turístico", side="bottom", tickfont=dict(size=12)),
        yaxis=dict(title="Tópico", tickfont=dict(size=11), autorange="reversed"),
        height=780,
        margin=dict(l=280, r=120, t=110, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif"),
    )

    fig.write_html(str(OUTPUT))
    logger.info("polarities_2.html -> %s", OUTPUT)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    run_polarities_heatmap()
