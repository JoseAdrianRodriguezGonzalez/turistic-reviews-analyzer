import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px

from config import ACCESSIBLE_PALETTES, Params, Paths

INPUT_DIR   = str(Paths.SENTIMENT_TOPICS_DIR)
OUTPUT_FILE = str(Paths.REPORTE_INTERACTIVO_HTML)


def cargar_datos(base_path):

    def safe_read_csv(path):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return pd.DataFrame() 

    def safe_read_json(path):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    # CSV (ahora opcionales)
    pos_df = safe_read_csv(f"{base_path}/topicos_positivo_scatter.csv")
    neg_df = safe_read_csv(f"{base_path}/topicos_negativo_scatter.csv")
    pvc_df = safe_read_csv(f"{base_path}/precio_valor_costo_scatter.csv")

    # JSON (ya lo tenías bien)
    pos_json = safe_read_json(f"{base_path}/topicos_positivos.json")
    neg_json = safe_read_json(f"{base_path}/topicos_negativos.json")
    pvc_json = safe_read_json(f"{base_path}/precio_valor_costo.json")

    return pos_df, neg_df, pvc_df, pos_json, neg_json, pvc_json

def mapear_keywords(df, json_data):
    lkp = {}
    for t in json_data.get('topicos', []):
        tid = int(t['id'])
        kw = ", ".join(t.get('keywords', [])[:4])
        lkp[tid] = f"Tópico {tid}: {kw}"

    df['nombre_topico'] = df['topico_id'].map(
        lambda x: lkp.get(x, "Comentarios Atípicos / Outliers" if x == -1 else f"Tópico {x}")
    )
    return df


def generar_scatter(df, titulo, color_col, palette_name, is_pvc=False):
    palettes = {
        name: getattr(px.colors.sequential, name.capitalize())
        for name in ACCESSIBLE_PALETTES
    }
    selected_palette = palettes.get(palette_name.lower(), px.colors.sequential.Plasma)

    if not is_pvc:
        hover_data = {
            'x': False,
            'y': False,
            'nombre_topico': True,
            'comentario': True
        }
        fig = px.scatter(
            df, x='x', y='y',
            color=color_col,
            hover_data=hover_data,
            title=titulo,
            color_discrete_sequence=selected_palette,
            opacity=0.75
        )
    else:
        hover_data = {
            'x': False,
            'y': False,
            'similitud_precio': True,
            'comentario': True
        }
        fig = px.scatter(
            df, x='x', y='y',
            color=color_col,
            hover_data=hover_data,
            title=titulo,
            color_continuous_scale=selected_palette,
            opacity=0.75
        )

    fig.update_traces(hoverlabel=dict(namelength=-1))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=30),
        font=dict(family="Poppins, 'Helvetica Neue', Arial, sans-serif"),
        legend=dict(
            title_text="Identificación de Grupos",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    return fig
def generar_barras_frecuencias(json_data, titulo):
    palabras = [item["palabra"] for item in json_data["frecuencias"][:20]]
    freqs = [item["frecuencia"] for item in json_data["frecuencias"][:20]]

    df = pd.DataFrame({
        "palabra": palabras[::-1],
        "frecuencia": freqs[::-1]
    })

    fig = px.bar(
        df,
        x="frecuencia",
        y="palabra",
        orientation="h",
        title=titulo
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=30)
    )

    return fig
def cargar_grafo_html():
    path = Paths.VISUALIZATION_DIR / "topic_graph_2.html"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "<p style='color:#7f8c8d;'>Grafo no disponible</p>"

def generar_html_reporte(fig_pos, fig_neg, fig_pvc, pvc_json, output_file):
    html_pos = fig_pos.to_html(full_html=False, include_plotlyjs='cdn', config={'scrollZoom': True})
    html_neg = fig_neg.to_html(full_html=False, include_plotlyjs=False, config={'scrollZoom': True})
    html_pvc = fig_pvc.to_html(full_html=False, include_plotlyjs=False, config={'scrollZoom': True})
    graph_html = cargar_grafo_html()
    # Extraer y formatear el Top 5
    top5_html = ""
    top5_list = pvc_json.get('comentarios_mas_relevantes', [])
    if top5_list:
        for idx, item in enumerate(top5_list[:5]):
            texto = item.get('comentario', str(item)) if isinstance(item, dict) else str(item)
            similitud = item.get('similitud_precio', item.get('similitud', '')) if isinstance(item, dict) else ""
            sim_badge = f"<span class='badge'>Similitud: {round(float(similitud), 4)}</span>" if similitud else ""

            top5_html += f"""
            <div class="top5-item">
                <strong>#{idx + 1}</strong> {texto} <br> {sim_badge}
            </div>
            """
    else:
        top5_html = "<p style='color: #7f8c8d;'>No se encontraron los comentarios más relevantes en los datos procesados.</p>"

    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Análisis de Sentimientos y Tópicos</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            body {{ 
                font-family: 'Poppins', sans-serif; 
                margin: 0; 
                padding: 40px; 
                background-color: #f4f6f9; 
                color: #2c3e50;
            }}
            .header-container {{
                margin-bottom: 40px;
                border-left: 5px solid #3498db;
                padding-left: 20px;
            }}
            h1 {{ font-weight: 600; color: #1a252f; margin: 0 0 10px 0; }}
            .subtitle {{ color: #7f8c8d; font-size: 1.1rem; margin: 0; }}
            .chart-card {{ 
                background: white; 
                padding: 30px; 
                border-radius: 12px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
                margin-bottom: 40px; 
                border: 1px solid #eef2f5; 
            }}
            h2 {{ font-weight: 500; color: #2c3e50; margin-top: 0; margin-bottom: 20px; font-size: 1.4rem; }}
            .instructions {{ font-size: 0.9rem; color: #95a5a6; margin-top: -15px; margin-bottom: 20px; }}

            /* Estilos para el Top 5 */
            .top5-container {{ margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px; }}
            .top5-item {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin-bottom: 12px;
                border-radius: 0 6px 6px 0;
                font-size: 0.95rem;
            }}
            .badge {{
                display: inline-block;
                background-color: #e8f4f8;
                color: #2980b9;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
                margin-top: 8px;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div class="header-container">
            <h1>Análisis Visual de Comentarios</h1>
            <p class="subtitle">Exploración avanzada de la distribución semántica, clusters de opinión y dimensiones de costo.</p>
        </div>

        <div class="chart-card">
            <h2>Tópicos Positivos</h2>
            <p class="instructions">Tip de navegación: Usa la rueda del ratón para hacer zoom o arrastra para desplazarte por la densidad de comentarios.</p>
            {html_pos}
        </div>

        <div class="chart-card">
            <h2>Tópicos Negativos</h2>
            <p class="instructions">Tip de navegación: Usa la rueda del ratón para hacer zoom o arrastra para desplazarte por la densidad de comentarios.</p>
            {html_neg}
        </div>

        <div class="chart-card">
            <h2>Análisis de "Precio / Valor / Costo"</h2>
            <p class="instructions">Tip de navegación: La escala de color representa la proximidad semántica calculada mediante embeddings.</p>
            {html_pvc}

            <div class="top5-container">
                <h3>Top 5 Comentarios Más Relevantes al Concepto</h3>
                {top5_html}
            </div>
        </div>

        <div class="chart-card">
            <h2>Grafo de Co-ocurrencia de Entidades</h2>
            <p class="instructions">
                Interacción: zoom, drag, hover sobre nodos.
            </p>

            {graph_html}
        </div>
    </body>
    </html>
    """

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"Reporte generado con éxito en:\n{output_file}")

def hay_topicos(json_data):
    return len(json_data.get("topicos", [])) > 0
from visualization.topic_graph_2 import run_topic_graph_interactivo 
def run_reporte_interactivo() -> None:
    palette = Params.COLOR_PALETTE or 'plasma'

    pos_df, neg_df, pvc_df, pos_json, neg_json, pvc_json = cargar_datos(INPUT_DIR)

    if hay_topicos(pos_json):
        fig_pos = generar_scatter(pos_df, "Distribución de Tópicos (Positivos)", "nombre_topico", palette)
        pos_df = mapear_keywords(pos_df, pos_json)
        
    else:
        fig_pos = generar_barras_frecuencias(
        pos_json,
        "Palabras más frecuentes (Positivos)")
    if hay_topicos(neg_json):
        fig_neg = generar_scatter(neg_df, "Distribución de Tópicos (Negativos)", "nombre_topico", palette)
        neg_df = mapear_keywords(neg_df, neg_json)
    else:
        fig_neg = generar_barras_frecuencias(
        neg_json,
        "Palabras más frecuentes (Negativos)"
    )
        
    fig_pvc = generar_scatter(pvc_df, 'Similitud con "Precio/Valor/Costo"', "similitud_precio", palette, is_pvc=True)

    generar_html_reporte(fig_pos, fig_neg, fig_pvc, pvc_json, OUTPUT_FILE)


def main():
    parser = argparse.ArgumentParser(description="Generador de visualizaciones interactivas")
    parser.add_argument(
        '--palette',
        type=str,
        default=Params.COLOR_PALETTE or 'plasma',
        choices=['viridis', 'cividis', 'plasma', 'inferno'],
        help="Nombre de la paleta de colores accesible (ej. viridis, cividis, plasma, inferno)"
    )
    args = parser.parse_args()

    print(f"Buscando datos en: {INPUT_DIR}")
    try:
        pos_df, neg_df, pvc_df, pos_json, neg_json, pvc_json = cargar_datos(INPUT_DIR)
    except FileNotFoundError as e:
        print(f"Error: No se encontraron los archivos en la ruta especificada.\nDetalles: {e}")
        return

    print("Mapeando palabras clave")
    pos_df = mapear_keywords(pos_df, pos_json)
    neg_df = mapear_keywords(neg_df, neg_json)

    print(f"Generando gráficos de dispersión con paleta '{args.palette}'")
    fig_pos = generar_scatter(pos_df, "Distribución de Tópicos (Positivos)", "nombre_topico", args.palette)
    fig_neg = generar_scatter(neg_df, "Distribución de Tópicos (Negativos)", "nombre_topico", args.palette)
    fig_pvc = generar_scatter(pvc_df, 'Similitud con "Precio/Valor/Costo"', "similitud_precio", args.palette,
                              is_pvc=True)

    print("Ensamblando y guardando HTML")
    generar_html_reporte(fig_pos, fig_neg, fig_pvc, pvc_json, OUTPUT_FILE)


if __name__ == "__main__":
    main()
