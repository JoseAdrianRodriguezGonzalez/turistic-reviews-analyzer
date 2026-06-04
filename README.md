# Turistic Reviews Analyzer

Pipeline de análisis NLP de comentarios turísticos, ejecutable desde la línea de comandos. Procesa un archivo CSV con reseñas, aplica limpieza de texto, detección de outliers, análisis de sentimientos, modelado de tópicos y genera visualizaciones interactivas en HTML.

El proceso es completamente autónomo: no requiere APIs externas ni conexión a internet durante la ejecución (los modelos se descargan una sola vez al primer uso y quedan almacenados localmente).

---

## Requisitos del sistema

- Python 3.10 o superior
- pip
- Al menos 8 GB de RAM (recomendado 16 GB para el corpus completo)
- ~5 GB de espacio en disco para modelos y datos intermedios

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/JoseAdrianRodriguezGonzalez/turistic-reviews-analyzer.git
cd turistic-reviews-analyzer
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Para GPU NVIDIA, sustituye la línea de `torch` en `requirements.txt` por:
> ```
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Descargar modelos de lenguaje de spaCy

El pipeline necesita los modelos de spaCy para cada idioma que vayas a analizar:

```bash
python -m spacy download es_core_news_sm   # español (requerido)
python -m spacy download en_core_web_sm    # inglés (si el corpus incluye reseñas en inglés)
python -m spacy download fr_core_news_sm   # francés (si el corpus incluye reseñas en francés)
```

### 5. (Opcional) Topic naming con LLM local

Para generar nombres descriptivos de tópicos automáticamente instala `llama-cpp-python` y descarga el modelo Mistral 7B (~4.1 GB):

```bash
pip install llama-cpp-python
huggingface-cli download bartowski/Mistral-7B-Instruct-v0.3-GGUF \
  --include "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
  --local-dir data/models/
mv "data/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
   "data/models/mistral-7b-instruct-v0.3-q4_k_m.gguf"
```

Sin este paso el pipeline funciona igual pero omite el nombrado automático de tópicos.

### 6. Descarga de modelos de transformers

Los modelos de HuggingFace (análisis de sentimientos, traducción, embeddings) se descargan automáticamente en la primera ejecución. No se requiere ninguna clave de API: los modelos son de código abierto y se ejecutan completamente en local tras la descarga.

Modelos utilizados:
- `pysentimiento/robertuito-sentiment-analysis` (~500 MB) — sentimiento en español
- `cardiffnlp/twitter-xlm-roberta-base-sentiment` (~500 MB) — sentimiento multilingüe (opcional)
- `Helsinki-NLP/opus-mt-en-es` (~300 MB) — traducción inglés→español
- `paraphrase-multilingual-MiniLM-L12-v2` (~270 MB) — embeddings semánticos

---

## Uso

### Sintaxis

```
python main.py INPUT_CSV TEXT_COLUMN LANGUAGE TITLE PALETTE [opciones]
```

### Parámetros posicionales (obligatorios)

| Parámetro    | Descripción                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `INPUT_CSV`  | Ruta al archivo CSV con los datos de reseñas                                |
| `TEXT_COLUMN`| Nombre de la columna que contiene los comentarios textuales                 |
| `LANGUAGE`   | Idioma objetivo del análisis: `es`, `en` o `fr`                             |
| `TITLE`      | Título del reporte usado en las visualizaciones                             |
| `PALETTE`    | Paleta de colores para las gráficas (ver opciones accesibles abajo)         |

**Paletas disponibles:** `viridis`, `cividis`, `plasma`, `inferno` (accesibles para daltonismo) · `sunset` (uso general)

### Parámetros opcionales

| Parámetro                         | Descripción                                                           |
|-----------------------------------|-----------------------------------------------------------------------|
| `--steps STEP [STEP ...]`         | Ejecutar solo los pasos indicados                                     |
| `--force`                         | Re-ejecutar aunque los outputs ya existan                             |
| `--status`                        | Mostrar el estado actual de cada paso y salir                         |
| `--list`                          | Listar los pasos disponibles y salir                                  |
| `--mode automatic`                | Cargar parámetros avanzados desde `pipeline_config.yaml`              |
| `--config-file PATH`              | Ruta al YAML de configuración (default: `pipeline_config.yaml`)       |
| `--sentiment-idioma {es,multi}`   | Modelo de sentimiento: `es` (robertuito) o `multi` (xlm-roberta)      |
| `--sentiment-metodo {rapido,robusto}` | Método de clasificación (default: `rapido`)                       |
| `--sentiment-texto {original,cleaned}` | Texto de entrada al transformer (default: `original`)            |
| `--sentiment-features FEATURE [...]` | Features para el método robusto: `embeddings`, `tfidf`, `yake`    |

### Ejemplos de ejecución

Ejecutar el pipeline completo:
```bash
python main.py data/raw/complete.csv comentario es "Análisis Turístico México" viridis
```

Ejecutar solo el paso de preprocesamiento:
```bash
python main.py data/raw/complete.csv comentario es "Análisis" viridis --steps preprocessing
```

Ejecutar clustering y análisis, forzando re-ejecución:
```bash
python main.py data/raw/complete.csv comentario es "Análisis" viridis --steps clustering analysis --force
```

Usar el modelo multilingüe con método robusto:
```bash
python main.py data/raw/complete.csv comentario es "Análisis" cividis \
    --sentiment-idioma multi --sentiment-metodo robusto
```

Modo automático (parámetros avanzados desde YAML):
```bash
python main.py data/raw/complete.csv comentario es "Análisis" viridis --mode automatic
```

Modo automático con archivo de configuración personalizado:
```bash
python main.py data/raw/complete.csv comentario es "Análisis" viridis \
    --mode automatic --config-file mi_config.yaml
```

Ver el estado actual del pipeline:
```bash
python main.py data/raw/complete.csv comentario es "x" viridis --status
```

---

## Configuración avanzada (modo automático)

El archivo `pipeline_config.yaml` permite ajustar los hiperparámetros de cada paso del pipeline sin modificar el código. Se activa con `--mode automatic`.

Estructura del archivo:

```yaml
global:
  random_state: 42
  embedding_model: "paraphrase-multilingual-MiniLM-L12-v2"

step_05_semantic:
  bertopic:
    umap:
      n_neighbors: 30      # rango típico: 5–50
      n_components: 8      # rango típico: 5–15
      min_dist: 0.0        # 0.0 = clusters compactos
      metric: "cosine"
    hdbscan:
      min_cluster_size: 15 # rango típico: 10–50
      metric: "euclidean"
      cluster_selection_method: "eom"
      calculate_probabilities: true
    batch_size: 64

step_06_clustering:
  svd_components: 50
  umap_components: 2
  jerarquico_knn_vecinos: 10
  alpha_kmeans: 0.7
  max_cluster_pct: 0.80

step_08_analysis:
  sentiment:
    idioma: "es"           # 'es' o 'multi'
    metodo: "rapido"       # 'rapido' o 'robusto'
    texto: "original"      # 'original' o 'cleaned'
    features: ["embeddings"]
    batch_size: 32
    max_length: 512

  outliers:
    contamination: 0.05    # fracción esperada de outliers (0.01–0.20)
    knn_neighbors: 20
    umbral_ensemble: 2     # votos mínimos de 3 modelos (1=sensible, 3=estricto)
    svm_max_train: 8000

  sentiment_topics:
    min_docs: 50
    precio_valor_costo_top_n: 5
```

Solo las claves presentes en el YAML sobreescriben los valores por defecto; el resto del pipeline usa los defaults del sistema.

---

## Pasos del pipeline

| Paso | Nombre           | Descripción                                                                 |
|------|------------------|-----------------------------------------------------------------------------|
| 01   | preprocessing    | Limpieza de texto, detección de idioma, tokenización, lematización          |
| 02   | translation      | Traducción de reseñas en inglés y francés al español (Helsinki-NLP)        |
| 03   | vocabulary       | Generación de rankings de unigramas, bigramas y trigramas                  |
| 04   | features         | Extracción de features NLP: TF-IDF, YAKE, entidades, POS                   |
| 05   | semantic         | Modelado de tópicos con BERTopic + UMAP + HDBSCAN y embeddings semánticos  |
| 06   | clustering       | Grid search de KMeans, jerárquico y HDBSCAN sobre 4 representaciones        |
| 07   | enrichment       | Enriquecimiento de clusters: keywords internas, docs representativos        |
| 08   | analysis         | Sentimientos, entidades, co-ocurrencia, tendencias, outliers, tópicos       |
| 09   | visualization    | Generación de visualizaciones HTML y PNG                                    |

---

## Análisis incluidos

**Detección de outliers**
Ensemble de tres modelos (Isolation Forest, One-Class SVM, Local Outlier Factor). Un comentario se clasifica como outlier si al menos 2 de 3 modelos lo señalan. Los outliers se analizan mediante unigramas, bigramas y trigramas. Los comentarios normales continúan al análisis de sentimientos.

**Análisis de sentimientos**
Clasificación binaria (positivo / negativo) mediante el transformer `robertuito` o `xlm-roberta`. El método `robusto` agrega un clasificador LogisticRegression entrenado sobre embeddings semánticos.

**Modelado de tópicos por sentimiento**
BERTopic se aplica por separado al grupo de comentarios positivos y al grupo de comentarios negativos. Si algún grupo tiene menos de 50 documentos (configurable), se usa conteo de frecuencias como alternativa. Por cada tópico se reportan las 10 palabras clave más representativas y el comentario más cercano al centroide del tópico.

**Análisis de precio / valor / costo**
Se genera un embedding sintético del concepto "precio valor costo dinero pago tarifa caro barato economico" y se calcula la similitud coseno contra todos los documentos del corpus. Se exportan los 5 comentarios más relevantes (configurable) y un scatter plot interactivo con la distribución de similitudes.

---

## Salidas generadas

Todos los archivos se guardan en la carpeta `data/` organizada por etapa:

```
data/
├── analysis/
│   ├── outliers/          # outliers.csv, normales.csv, n-gramas
│   ├── sentiment/         # corpus_con_sentimiento.csv, positivos, negativos
│   ├── sentiment_topics/  # topicos_positivos.json, topicos_negativos.json,
│   │                      # scatter plots CSV, precio_valor_costo.json
│   ├── entities/          # entidades por destino y tópico
│   ├── cooccurrence/      # grafo de co-ocurrencia de entidades
│   └── trends/            # tendencias y perfil por destino
└── visualization/
    ├── topic_graph.png            # mapa de tópicos (estático)
    ├── topic_graph_2.html         # grafo de co-ocurrencia (interactivo)
    ├── keywords_entities.png      # keywords y entidades por destino (estático)
    ├── keywords_entities_2.html   # keywords por cluster (interactivo)
    ├── polarities.png             # distribución de polaridades (estático)
    ├── polarities_2.html          # heatmap de sentimiento × tópico (interactivo)
    └── metadata_overview.html     # resumen de volumen y distribución de datos
```

---

## Estructura del repositorio

```
turistic-reviews-analyzer/
├── main.py                    # punto de entrada del CLI
├── config.py                  # rutas y parámetros centralizados
├── pipeline_config.yaml       # configuración del modo automático
├── requirements.txt
├── pipeline/                  # clases Step para cada etapa
├── preprocessing/             # limpieza y tokenización
├── translation/               # traducción automática
├── feature_engineering/       # extracción de features NLP
├── semantic_expression/       # BERTopic y vectorización
├── clustering/                # KMeans, HDBSCAN, jerárquico
├── topic_enrichment/          # keywords y docs representativos
├── analysis/                  # sentimientos, outliers, entidades
└── visualization/             # generación de gráficas
```

---

## Colaboradores

- Jose Adrian Rodriguez Gonzalez — modelado de tópicos, semántica
- Jorge Barroso Solorzano — análisis de outliers, modo automático
- Marco Illescas — análisis de sentimientos
- Emilio Velazquez — visualizaciones

---

## Notas sobre los modelos

Los modelos de HuggingFace se descargan automáticamente al primer uso mediante la librería `transformers` y `sentence-transformers`. No se requiere clave de API ni registro en ningún servicio. Una vez descargados, el pipeline funciona completamente sin conexión a internet.

La carpeta de caché por defecto es `~/.cache/huggingface/`. Para cambiar la ubicación, configura la variable de entorno `HF_HOME` antes de ejecutar el script.
