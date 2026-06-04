import logging

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Params, Paths

logger = logging.getLogger(__name__)

INPUT_PATH = Paths.PERFIL_DESTINO_CSV
OUTPUT     = Paths.VISUALIZATION_DIR / "metadata_overview.html"

_DEST_LABELS = dict(zip(Params.DEST_ORDER, Params.DEST_LABELS))


def generar_metadata_overview() -> None:
    Paths.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generando Overview de Metadatos...")
    try:
        df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
        df["location_label"] = df["location"].map(_DEST_LABELS)
        df = df.sort_values(by="n_total", ascending=True)

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=("Volumen Total de Reseñas", "Distribución de Idiomas por Destino"),
        )

        fig.add_trace(go.Pie(
            labels=df["location_label"],
            values=df["n_total"],
            hole=0.4,
            marker=dict(colors=Params.DEST_COLORS, line=dict(color="#FFFFFF", width=2)),
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Comentarios: %{value:,}<extra></extra>",
        ), row=1, col=1)

        idiomas = [("Español", "es", "#5DADE2"), ("Inglés", "en", "#F4D03F"), ("Mixto", "mix", "#AAB7B8")]
        for nombre, col, color in idiomas:
            fig.add_trace(go.Bar(
                name=nombre,
                y=df["location_label"],
                x=df[col],
                orientation="h",
                marker=dict(color=color),
                hovertemplate=f"<b>%{{y}}</b><br>{nombre}: %{{x:,}}<extra></extra>",
            ), row=1, col=2)

        fig.update_layout(
            barmode="stack",
            title=dict(
                text="<b>Resumen de Datos Recolectados</b>",
                x=0.5, xanchor="center",
                font=dict(size=20, color="#1a1a2e"),
            ),
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="#f7f9fc",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        fig.update_xaxes(title_text="Cantidad de Reseñas", row=1, col=2, gridcolor="#e5e5e5")
        fig.update_yaxes(title_text="", row=1, col=2)

        fig.write_html(str(OUTPUT))
        logger.info("metadata_overview.html -> %s", OUTPUT)

    except Exception as e:
        logger.error("Error al generar metadata_overview: %s", e)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    generar_metadata_overview()
