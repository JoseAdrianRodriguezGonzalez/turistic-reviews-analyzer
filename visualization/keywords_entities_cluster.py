import logging

import pandas as pd
import plotly.express as px

from config import Paths

logger = logging.getLogger(__name__)

_RUTA_KW = Paths.ENRICHMENT_KEYWORDS_CSV
_OUTPUT  = Paths.VISUALIZATION_DIR / "keywords_entities_2.html"


def generar_keywords_entities() -> None:
    Paths.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generando Keywords + Entities por cluster...")
    try:
        df_kw = pd.read_csv(_RUTA_KW)
        df_top = df_kw[df_kw["rank"] <= 10].copy()
        df_top = df_top.sort_values(by=["cluster_id", "score_tfidf"], ascending=[True, True])

        fig = px.bar(
            df_top,
            x="score_tfidf",
            y="termino",
            facet_col="cluster_id",
            facet_col_wrap=4,
            color="score_tfidf",
            color_continuous_scale="Viridis",
            title="Top Keywords y Entidades por Tópico (TF-IDF)",
            height=800,
        )
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(showlegend=False)
        fig.write_html(str(_OUTPUT))
        logger.info("keywords_entities_2.html -> %s", _OUTPUT)
    except Exception as e:
        logger.error("Error al generar Keywords + Entities: %s", e)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    generar_keywords_entities()
