import logging

import networkx as nx
import pandas as pd
from pyvis.network import Network

from config import Params, Paths

logger = logging.getLogger(__name__)

INPUT  = Paths.COOCURRENCIA_ENTIDADES_CSV
OUTPUT = Paths.VISUALIZATION_DIR / "topic_graph_2.html"


def run_topic_graph_interactivo() -> None:
    Paths.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    _fp = set(Params.VIZ_FALSE_POSITIVES)
    df_clean = df[
        ~df["entidad_a"].isin(_fp) &
        ~df["entidad_b"].isin(_fp) &
        (df["co_ocurrencias"] > Params.VIZ_COOC_MIN_WEIGHT)
    ].copy()

    if df_clean.empty:
        logger.warning("Sin aristas tras filtrado. Revisa los umbrales.")
        return

    logger.info("Aristas originales: %d | filtradas: %d", len(df), len(df_clean))

    G = nx.Graph()
    for _, row in df_clean.iterrows():
        G.add_edge(
            row["entidad_a"], row["entidad_b"],
            weight=int(row["co_ocurrencias"]),
            pmi=round(float(row["pmi"]), 2),
        )

    freq_map: dict[str, int] = {}
    for _, row in df_clean.iterrows():
        freq_map[row["entidad_a"]] = int(row["doc_freq_a"])
        freq_map[row["entidad_b"]] = int(row["doc_freq_b"])

    logger.info("Nodos: %d | aristas: %d", G.number_of_nodes(), G.number_of_edges())

    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    node_community: dict[str, int] = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    net = Network(
        height="820px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.05,
        damping=0.09,
    )

    min_freq = min(freq_map.values())
    max_freq = max(freq_map.values())

    for node in G.nodes():
        freq  = freq_map.get(node, 10)
        size  = 12 + 33 * (freq - min_freq) / max(max_freq - min_freq, 1)
        comm_idx = node_community.get(node, 0) % len(Params.CLUSTER_COLORS)
        color = Params.CLUSTER_COLORS[comm_idx]
        degree = G.degree(node)
        title = (
            f"<b>{node}</b><br>"
            f"Frecuencia en corpus: {freq}<br>"
            f"Conexiones: {degree}"
        )
        net.add_node(
            node, label=node, size=float(size), color=color,
            title=title, font={"size": 13, "color": "#222222"},
        )

    max_cooc = df_clean["co_ocurrencias"].max()
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        width = 1 + 7 * (w / max_cooc)
        title = f"Co-ocurrencias: {w}<br>PMI: {data['pmi']}"
        net.add_edge(u, v, width=float(width), title=title,
                     color={"color": "#aaaaaa", "opacity": 0.7})

    net.set_options("""
    {
      "interaction": {"hover": true, "tooltipDelay": 100},
      "physics": {"enabled": true, "stabilization": {"iterations": 200}}
    }
    """)

    net.write_html(str(OUTPUT))

    # Inyectar leyenda en el HTML generado
    legend_items = ""
    for i, comm in enumerate(communities):
        color = Params.CLUSTER_COLORS[i % len(Params.CLUSTER_COLORS)]
        top_nodes = sorted(comm, key=lambda n: G.degree(n), reverse=True)[:3]
        legend_items += (
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;background:{color};margin-right:5px"></span>'
            f'Comunidad {i + 1}: {", ".join(top_nodes)}<br>'
        )

    legend_html = f"""
    <div style="position:fixed;top:16px;left:16px;background:rgba(255,255,255,0.93);
                padding:14px 18px;border-radius:10px;border:1px solid #ddd;
                font-family:sans-serif;font-size:12px;max-width:320px;z-index:9999;
                box-shadow:0 2px 8px rgba(0,0,0,0.12)">
      <b style="font-size:14px">Grafo de Co-ocurrencia de Entidades</b><br>
      <span style="color:#777;font-size:11px">
        Nodo: tamaño = frecuencia en corpus<br>
        Arista: grosor = co-ocurrencias<br>
        Color: comunidad detectada
      </span><br><br>
      {legend_items}
    </div>
    """
    html = OUTPUT.read_text(encoding="utf-8")
    html = html.replace("<body>", "<body>\n" + legend_html, 1)
    OUTPUT.write_text(html, encoding="utf-8")

    logger.info("topic_graph_2.html -> %s", OUTPUT)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    run_topic_graph_interactivo()
