"""
config.py
---------
Configuración centralizada del proyecto.
Todas las rutas, constantes y parámetros del pipeline viven aquí.
Ningún módulo define sus propias rutas — solo importan desde este archivo.

Uso:
    from config import Paths, Params, Logging

    ruta = Paths.RAW_GOOGLE / "huatulco.csv"

Los parámetros de entrada del usuario (CSV path, columna, idioma, título,
paleta) se inyectan una sola vez desde main.py mediante:

    Params.set_from_args(args)

Después de esa llamada, cualquier módulo puede leer:

    Params.INPUT_CSV        -> Path al archivo CSV del usuario
    Params.TEXT_COLUMN      -> nombre de la columna de comentarios
    Params.LANGUAGE         -> código de idioma (es | en | fr)
    Params.REPORT_TITLE     -> título del reporte
    Params.COLOR_PALETTE    -> paleta de colores para gráficas
"""

import logging
from pathlib import Path

# RAÍZ DEL PROYECTO
# Todos los paths son relativos a esta carpeta (4_analisis/)

ROOT_DIR = Path(__file__).resolve().parent


# RUTAS — organizadas por sección

class Paths:
    """
    Clase de solo lectura con todas las rutas del proyecto.
    No instanciar — usar directamente como Paths.DATA_DIR
    """

    # Raíz de datos
    DATA_DIR         = ROOT_DIR / "data"
    # Datos crudos
    RAW_DIR          = DATA_DIR / "raw"
    RAW_GOOGLE       = RAW_DIR  / "google"
    RAW_TRIP         = RAW_DIR  / "trip"
    RAW_INSTA        = RAW_DIR  / "insta"
    RAW_COMPLETE_CSV = RAW_DIR  / "complete.csv"
    # Datos por idioma (output del preprocesamiento)
    DATA_SPANISH_DIR  = DATA_DIR / "data_spanish"
    DATA_ENGLISH_DIR  = DATA_DIR / "data_english"
    DATA_MIXED_DIR    = DATA_DIR / "data_mixed"
    DATA_FRENCH_DIR   = DATA_DIR / "data_french"
    SPANISH_CLEAN_CSV    = DATA_SPANISH_DIR / "clean.csv"
    SPANISH_ANALYSIS_CSV = DATA_SPANISH_DIR / "analysis.csv"
    SPANISH_ANALYSIS_JSON= DATA_SPANISH_DIR / "analysis.json"
    ENGLISH_CLEAN_CSV    = DATA_ENGLISH_DIR / "clean.csv"
    ENGLISH_ANALYSIS_CSV = DATA_ENGLISH_DIR / "analysis.csv"
    ENGLISH_ANALYSIS_JSON= DATA_ENGLISH_DIR / "analysis.json"
    MIXED_CLEAN_CSV      = DATA_MIXED_DIR   / "clean.csv"
    MIXED_ANALYSIS_CSV   = DATA_MIXED_DIR   / "analysis.csv"
    MIXED_ANALYSIS_JSON  = DATA_MIXED_DIR   / "analysis.json"
    FRENCH_CLEAN_CSV     = DATA_FRENCH_DIR  / "clean.csv"
    FRENCH_ANALYSIS_CSV  = DATA_FRENCH_DIR  / "analysis.csv"
    FRENCH_ANALYSIS_JSON = DATA_FRENCH_DIR  / "analysis.json"
    # Traducciones y normalización de idioma
    TRANSLATIONS_DIR      = DATA_DIR / "translations"
    JOINED_CSV            = TRANSLATIONS_DIR / "joined.csv"
    NORMALIZED_SPANISH_CSV= TRANSLATIONS_DIR / "normalized_spanish.csv"
    # Datos unificados (merge de los tres idiomas)
    UNIFIED_DIR           = DATA_DIR / "unified"
    UNIFIED_ANALYSIS_CSV  = UNIFIED_DIR / "analysis_unified.csv"
    UNIFIED_ANALYSIS_JSON = UNIFIED_DIR / "analysis_json_unified.json"
    # Vocabulario y n-gramas
    PROCESSED_DIR         = DATA_DIR / "processed"
    RANKINGS_UNIGRAMS_CSV = PROCESSED_DIR / "rankings_unigrams.csv"
    RANKINGS_BIGRAMS_CSV  = PROCESSED_DIR / "rankings_bigrams.csv"
    RANKINGS_TRIGRAMS_CSV = PROCESSED_DIR / "rankings_trigrams.csv"
    # Features NLP
    FEATURES_DIR          = DATA_DIR / "features"
    FEATURES_NLP_CSV      = FEATURES_DIR / "features_nlp.csv"
    DOCS_WITH_TOPICS_NPY  = FEATURES_DIR / "docs_with_topics.npy"
    NER_GROUPS_JSON       = FEATURES_DIR / "ner_groups.json"
    ENTITIES_TOP_WORDS_JSON = FEATURES_DIR / "entities_top_words.json"
    # Modelos entrenados
    MODELS_DIR            = DATA_DIR / "models"
    BERTOPIC_MODEL_DIR    = MODELS_DIR / "bertopic_model"
    TFIDF_PKL             = MODELS_DIR / "tfidf.pkl"
    YAKE_PKL              = MODELS_DIR / "yake_vectorizer.pkl"
    # Resultados de BERTopic
    RESULTS_DIR           = DATA_DIR / "results"
    ENRICHMENT_TEXTS = RESULTS_DIR /"text_enrich.csv"
    DOCS_WITH_TOPICS_CSV  = RESULTS_DIR / "docs_with_topics.csv"
    TOPICS_CSV            = RESULTS_DIR / "topics.csv"
    MICROTOPICS_CSV       = RESULTS_DIR / "microtopics.csv"
    # Clustering
    CLUSTERING_DIR               = DATA_DIR / "clustering"
    CLUSTERING_COMPARACION_CSV   = CLUSTERING_DIR / "comparacion_fuentes.csv"
    CLUSTERING_EMBEDDINGS_DIR    = CLUSTERING_DIR / "embeddings"
    CLUSTERING_FEATURES_DIR      = CLUSTERING_DIR / "features"
    CLUSTERING_TFIDF_DIR         = CLUSTERING_DIR / "tfidf"
    CLUSTERING_YAKE_DIR          = CLUSTERING_DIR / "yake"
    # Archivos estándar dentro de cada subcarpeta de clustering
    CLUSTERING_RANKING_FILE      = Path("ranking_completo.csv")
    CLUSTERING_MEJORES_FILE      = Path("mejores_modelos.csv")
    CLUSTERING_ETIQUETAS_FILE    = Path("etiquetas_mejores.json")
    CLUSTERING_PROYECCION_FILE   = Path("proyeccion_2d.npy")
    # Topic enrichment
    ENRICHMENT_DIR               = DATA_DIR / "topic_enrichment"
    ENRICHMENT_RESUMEN_CSV       = ENRICHMENT_DIR / "resumen_enrichment.csv"
    # Ruta activa usada por las visualizaciones (KMeans k=5 sobre embeddings — mejor silhouette)
    ENRICHMENT_ACTIVE_DIR        = ENRICHMENT_DIR / "embeddings" / "kmeans_k9"
    # Modelo LLM local para topic naming (requiere llama_cpp)
    LLAMA_MODEL_PATH             = MODELS_DIR / "mistral-7b-instruct-v0.3-q4_k_m.gguf"
    ENRICHMENT_KEYWORDS_CSV      = ENRICHMENT_ACTIVE_DIR / "keywords_por_cluster.csv"
    # Análisis
    ANALYSIS_DIR                 = DATA_DIR / "analysis"
    ANALYSIS_RESUMEN_CSV         = ANALYSIS_DIR / "resumen_analysis.csv"
    SENTIMENT_DIR                = ANALYSIS_DIR / "sentiment"
    CORPUS_CON_SENTIMIENTO_CSV   = SENTIMENT_DIR / "corpus_con_sentimiento.csv"
    OUTLIERS_DIR                 = ANALYSIS_DIR / "outliers"
    OUTLIERS_CSV                 = OUTLIERS_DIR  / "outliers.csv"
    NORMALES_CSV                 = OUTLIERS_DIR  / "normales.csv"
    OUTLIERS_NGRAMS_UNI_CSV      = OUTLIERS_DIR  / "outliers_unigrams.csv"
    OUTLIERS_NGRAMS_BI_CSV       = OUTLIERS_DIR  / "outliers_bigrams.csv"
    OUTLIERS_NGRAMS_TRI_CSV      = OUTLIERS_DIR  / "outliers_trigrams.csv"
    OUTLIERS_RESUMEN_CSV         = OUTLIERS_DIR  / "resumen_outliers.csv"
    SENTIMENT_TOPICS_DIR              = ANALYSIS_DIR / "sentiment_topics"
    SENTIMENT_TOPICS_POS_JSON         = SENTIMENT_TOPICS_DIR / "topicos_positivos.json"
    SENTIMENT_TOPICS_NEG_JSON         = SENTIMENT_TOPICS_DIR / "topicos_negativos.json"
    PRECIO_VALOR_COSTO_JSON           = SENTIMENT_TOPICS_DIR / "precio_valor_costo.json"
    PRECIO_VALOR_COSTO_SCATTER_CSV    = SENTIMENT_TOPICS_DIR / "precio_valor_costo_scatter.csv"
    SENTIMENT_TOPICS_RESUMEN_CSV      = SENTIMENT_TOPICS_DIR / "resumen_sentiment_topics.csv"
    SENTIMIENTO_POR_TOPICO_CSV   = SENTIMENT_DIR / "sentimiento_por_topico.csv"
    SENTIMIENTO_POR_DESTINO_CSV  = SENTIMENT_DIR / "sentimiento_por_destino.csv"
    SENTIMIENTO_POR_TOPICO_DESTINO_CSV = SENTIMENT_DIR / "sentimiento_por_topico_destino.csv"
    ENTITIES_DIR                 = ANALYSIS_DIR / "entities"
    ENTIDADES_CON_SENTIMIENTO_CSV= ENTITIES_DIR / "entidades_con_sentimiento.csv"
    ENTIDADES_POR_DESTINO_CSV    = ENTITIES_DIR / "entidades_por_destino.csv"
    ENTIDADES_POR_TOPICO_CSV     = ENTITIES_DIR / "entidades_por_topico.csv"
    COOCCURRENCE_DIR             = ANALYSIS_DIR / "cooccurrence"
    COOCURRENCIA_ENTIDADES_CSV   = COOCCURRENCE_DIR / "coocurrencia_entidades.csv"
    COOCURRENCIA_TERMINOS_CSV    = COOCCURRENCE_DIR / "coocurrencia_terminos.csv"
    COMUNIDADES_ENTIDADES_CSV    = COOCCURRENCE_DIR / "comunidades_entidades.csv"
    TRENDS_DIR                   = ANALYSIS_DIR / "trends"
    TENDENCIAS_TOPICOS_DESTINO_CSV     = TRENDS_DIR / "tendencias_topicos_destino.csv"
    TENDENCIAS_SENTIMIENTO_TOPICO_CSV  = TRENDS_DIR / "tendencias_sentimiento_topico.csv"
    MICROTOPICOS_RESUMEN_CSV           = TRENDS_DIR / "microtopicos_resumen.csv"
    PERFIL_DESTINO_CSV                 = TRENDS_DIR / "perfil_destino.csv"
    # Visualizaciones (output)
    VISUALIZATION_DIR            = DATA_DIR / "visualization"
    REPORTE_INTERACTIVO_HTML     = RESULTS_DIR / "reporte_interactivo.html"


# PARÁMETROS DEL PIPELINE

ACCESSIBLE_PALETTES = {"viridis", "cividis", "plasma", "inferno", "sunset"}
SUPPORTED_LANGUAGES = {"es", "en", "fr","all"}

class Params:
    """
    Constantes y parámetros configurables del pipeline.
    Cambiar aquí afecta a todos los módulos que los usen.

    Los 5 parámetros de entrada del usuario se inyectan desde main.py
    con Params.set_from_args(args) antes de ejecutar cualquier step.
    """

    INPUT_CSV:     Path | None = None
    TEXT_COLUMN:   str  | None = None
    LANGUAGE:      str  | None = None
    REPORT_TITLE:  str  | None = None
    COLOR_PALETTE: str  | None = None
    MODE:          str  | None = None   # 'automatic' | None (estándar)
    CONFIG_FILE:   Path | None = None   # ruta al pipeline_config.yaml

    #  Texto
    COLUMNA_TEXTO        = "comentario_clean"
    COLUMNA_COMENTARIO   = "comentario"
    COLUMNA_LANG         = "lang"

    #  Idiomas
    LANG_SPANISH = "es"
    LANG_ENGLISH = "en"
    LANG_FRENCH  = "fr"
    LANG_MIXED   = "mixed"
    LANG_ALL="all"

    # Dispositivo de cómputo: "auto" | "gpu" | "cpu"
    DEVICE: str = "gpu"

    # --- Traducción ---
    TRANSLATION_LANG_FILTER: list = ["en", "mix", "mixed"]
    TRANSLATION_BATCH_SIZE: int = 32
    TRANSLATION_MAX_LENGTH: int = 512
    TRANSLATION_NUM_BEAMS: int  = 1

    #  Vocabulario
    MIN_FREQ_VOCAB       = 2
    MAX_VOCAB_SIZE       = None     # None = sin límite
    VOCAB_MIN_TOKEN_LEN  = 2        # longitud mínima de token para n-gramas

    # --- Feature engineering ---
    DEMOGRAPHIC_COLUMNS: list = ['edad', 'genero', 'lugar']
    POS_RELEVANT_TAGS: list   = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
    NER_ENTITY_TYPES: list    = ['GPE', 'LOC', 'ORG', 'FAC']
    SPACY_BATCH_SIZE: int     = 64

    # --- Semántica / BERTopic ---
    SEMANTIC_RUN_MICROTOPICS: bool = False
    MICROTOPIC_MIN_DOCS: int       = 30

    # NER (project-specific thresholds)
    NER_MIN_ENTITY_LEN: int   = 3
    NER_TOP_NOUN_PHRASES: int = 5
    NER_MERGE_THRESHOLD: int  = 90

    # TF-IDF (sklearn — hiperparámetros cargados desde YAML)
    TFIDF_MIN_DF: int           = 2
    TFIDF_MAX_DF: float         = 0.9
    TFIDF_NGRAM_MIN: int        = 1
    TFIDF_NGRAM_MAX: int        = 2
    TFIDF_MAX_FEATURES          = None
    SEMANTIC_TFIDF_TOP_WORDS: int = 10

    # YAKE (keyword extraction — hiperparámetros cargados desde YAML)
    YAKE_LANGUAGE: str = "es"
    YAKE_NGRAM: int    = 2
    YAKE_TOP_K: int    = 5

    #  Clustering
    K_RANGO              = range(2, 11)
    ALPHA_KMEANS         = 0.7
    MAX_CLUSTER_PCT      = 0.80
    UMAP_COMPONENTS      = 2
    SVD_COMPONENTS       = 50

    EPSILONS             = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
    MIN_SAMPLES_LIST     = list(range(2, 8))
    METODOS_JERARQUICO   = ["ward", "complete", "average", "single"]

    MIN_CLUSTER_SIZES_HDBSCAN = [5, 10, 15, 20, 30]
    MIN_SAMPLES_HDBSCAN       = [3, 5, 10]

    # Muestreo de silhouette (O(n²) -> O(k²) para corpus grandes)
    SILHOUETTE_SAMPLE_THRESHOLD: int = 10_000
    SILHOUETTE_SAMPLE_SIZE: int      = 5_000

    # Límite de docs para linkage jerárquico (pdist es O(n²) en RAM)
    HIERARCHY_MAX_DOCS: int          = 5_000

    # Umbral para activar grafo kNN sparse en clustering jerárquico
    CLUSTERING_SPARSE_THRESHOLD: int = 10_000

    # Columnas de metadatos a excluir antes de clusterizar features NLP
    CLUSTERING_METADATA_COLS: list = ['indice', 'edad', 'genero', 'lugar', 'index']

    # Umbral mínimo de tamaño de cluster en HDBSCAN
    CLUSTERING_MIN_CLUSTER_PCT: float = 0.05
    CLUSTERING_MIN_CLUSTER_ABS: int   = 3

    # Flags de activación por fuente y algoritmo
    CLUSTERING_RUN_HDBSCAN   : bool = True
    CLUSTERING_RUN_JERARQUICO: bool = True
    CLUSTERING_RUN_EMBEDDINGS: bool = True
    CLUSTERING_RUN_FEATURES  : bool = True
    CLUSTERING_RUN_TFIDF     : bool = True
    CLUSTERING_RUN_YAKE      : bool = True

    # --- Topic enrichment ---
    TOP_N_KEYWORDS       = 15
    TOP_K_DOCS_REPR      = 5
    ENRICHMENT_FUENTES: list = ['embeddings', 'features', 'tfidf', 'yake']

    # Jerarquía de clusters — métodos de scipy (hiperparámetros cargados desde YAML)
    HIERARCHY_LINKAGE_METHOD: str = 'ward'
    HIERARCHY_LINKAGE_METRIC: str = 'euclidean'

    # Topic naming con LLM local (hiperparámetros cargados desde YAML)
    TOPIC_NAMING_TOP_DOCS: int          = 3
    TOPIC_NAMING_MAX_TOKENS: int        = 20
    TOPIC_NAMING_TEMPERATURE: float     = 0.3
    TOPIC_NAMING_N_CTX: int             = 1024
    TOPIC_NAMING_N_THREADS: int         = 8
    TOPIC_NAMING_N_GPU_LAYERS: int      = 20

    # --- Análisis ---
    MIN_DOCS_ENTIDAD     = 5
    MAX_ENTIDADES        = 500
    MIN_DOCS_COOC        = 5
    TOP_VOCAB_TERMINOS   = 200

    # --- LDA ---
    N_TOPICOS_RANGO      = range(2, 11)
    N_PALABRAS_TOP_LDA   = 10
    ALPHA_LDA            = 0.6

    # --- Clustering activo para visualizaciones ---
    CLUSTERING_ACTIVO_FUENTE = "embeddings"
    CLUSTERING_ACTIVO_MODELO = "kmeans"
    CLUSTERING_ACTIVO_K      = 5
    CLUSTERING_ACTIVO_KEY    = "kmeans|k=5"

    # --- Destinos turísticos ---
    DEST_ORDER  = [
        "huatulco",
        "la_paz",
        "riviera_nayarit",
        "puerto_vallarta",
        "riviera_maya",
    ]
    DEST_LABELS = [
        "Huatulco",
        "La Paz",
        "Riviera Nayarit",
        "Puerto Vallarta",
        "Riviera Maya",
    ]
    DEST_COLORS = [
        "#0097A7",
        "#388E3C",
        "#F57C00",
        "#7B1FA2",
        "#C62828",
    ]

    # --- Colores de clusters (KMeans k=5) ---
    CLUSTER_COLORS = [
        "#1565C0",
        "#00838F",
        "#2E7D32",
        "#E65100",
        "#6A1B9A",
        "#C62828",
        "#F9A825",
        "#4E342E",
    ]

    # --- Modelos de embeddings ---
    EMBEDDING_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
    SPACY_MODEL_ES       = "es_core_news_sm"
    SPACY_MODEL_EN       = "en_core_web_sm"
    SPACY_MODEL_FR       = "fr_core_news_sm"
    TRANSLATION_MODEL    = "Helsinki-NLP/opus-mt-en-es"

    # Semilla global de aleatoriedad; afecta UMAP, IsolationForest, SVD y langdetect
    RANDOM_STATE         : int   = 42

    # --- Preprocesamiento ---
    CSV_TEXT_COLUMN_ALIASES: list = ["comment", "comentarios", "comentario", "texto", "review_text"]
    CSV_RATING_COLUMN_ALIASES: list = ["rating", "estrellas", "cantidad de estrellas"]
    CSV_ENCODINGS: tuple = ("utf-8", "cp1252", "latin-1")

    # UMAP interno de BERTopic — valores típicos: n_neighbors 5–50, n_components 5–15
    BERTOPIC_UMAP_N_NEIGHBORS  : int   = 30
    BERTOPIC_UMAP_N_COMPONENTS : int   = 8
    BERTOPIC_UMAP_MIN_DIST     : float = 0.0   # 0.0 produce clusters más compactos
    BERTOPIC_UMAP_METRIC       : str   = 'cosine'  # recomendado: 'cosine' o 'euclidean'

    # HDBSCAN interno de BERTopic — min_cluster_size controla la granularidad de tópicos
    BERTOPIC_HDBSCAN_MIN_CLUSTER_SIZE  : int  = 15   # valores típicos: 10–50
    BERTOPIC_HDBSCAN_METRIC            : str  = 'euclidean'  # recomendado: 'euclidean'
    BERTOPIC_HDBSCAN_CLUSTER_SELECTION : str  = 'eom'  # alternativa: 'leaf' (más tópicos pequeños)
    BERTOPIC_CALCULATE_PROBS           : bool = True

    # Tamaño de lote para SentenceTransformer en BERTopic; aumentar si hay GPU
    BERTOPIC_BATCH_SIZE : int = 64   # valores típicos: 32–256

    # k vecinos para grafo de conectividad en clustering jerárquico (solo corpus > 10 000 docs)
    JERARQUICO_KNN_VECINOS : int = 10   # valores típicos: 5–20

    # Fracción esperada de documentos atípicos (0.0–0.5); afecta IsolationForest, SVM y LOF
    OUTLIER_CONTAMINATION  : float = 0.05
    # Vecinos para Local Outlier Factor; valores típicos: 10–50
    OUTLIER_KNN_NEIGHBORS  : int   = 20
    # Votos mínimos (de 3 modelos) para declarar outlier; 1 = más sensible, 3 = más estricto
    OUTLIER_UMBRAL_ENSEMBLE: int   = 2
    # Máximo de muestras de entrenamiento para One-Class SVM (O(n²)); aumentar solo con RAM suficiente
    OUTLIER_SVM_MAX_TRAIN  : int   = 8_000

    # Detección de idioma (langdetect) — umbrales de confianza
    LANGDETECT_SHORT_THRESHOLD: int = 15
    LANGDETECT_CONF_HIGH: float = 0.95
    LANGDETECT_CONF_NORMAL: float = 0.85
    LANGDETECT_CONF_SECONDARY: float = 0.99

    # Documentos mínimos por grupo de sentimiento para aplicar BERTopic; si es menor usa frecuencias
    SENTIMENT_TOPIC_MIN_DOCS  : int = 50
    # Algoritmo de clustering para el modelado de tópicos por sentimiento
    # "hdbscan" -> encuentra clusters naturales (cantidad variable, puede dar menos de nr_topics)
    # "kmeans"  -> garantiza exactamente nr_topics (crea divisiones aunque no sean naturales)
    SENTIMENT_TOPIC_METODO    : str = "hdbscan"
    # Número de tópicos objetivo por grupo
    # hdbscan: reduce jerárquicamente si encuentra más (0 = sin reducción, usa los naturales)
    # kmeans:  k fijo, siempre produce exactamente este número
    SENTIMENT_TOPIC_NR_TOPICS : int = 5
    # Top-N comentarios más similares al concepto 'precio/valor/costo'
    PRECIO_VALOR_COSTO_TOP_N  : int = 5

    # Modelo transformer: 'es' (robertuito, solo español) o 'multi' (xlm-roberta, multilingüe)
    SENTIMENT_IDIOMA    : str       = 'es'
    # Método de clasificación: 'rapido' (solo transformer) o 'robusto' (transformer + LogisticRegression)
    SENTIMENT_METODO    : str       = 'rapido'
    # Texto de entrada al transformer: 'original' (recomendado) o 'cleaned' (lematizado)
    SENTIMENT_TEXTO     : str       = 'original'
    # Features para el método robusto; combinables: 'embeddings', 'tfidf', 'yake'
    SENTIMENT_FEATURES  : list[str] = ['embeddings']
    # Tamaño de lote para inferencia del transformer; aumentar si hay GPU
    SENTIMENT_BATCH_SIZE: int       = 32   # valores típicos: 16–128
    # Longitud máxima de tokens; 512 es el límite de la mayoría de modelos BERT
    SENTIMENT_MAX_LENGTH: int       = 512
    # Hiperparámetros de LogisticRegression para el método robusto
    SENTIMENT_LR_MAX_ITER: int      = 1000  # aumentar si no converge
    SENTIMENT_LR_C      : float     = 1.0   # inverso de regularización; menor = más regularización
    SENTIMENT_LR_SOLVER : str       = 'lbfgs'  # alternativas: 'saga' (datasets grandes), 'liblinear'

    # --- Análisis: flags de activación por sub-módulo ---
    ANALYSIS_RUN_SENTIMENT        : bool = True
    ANALYSIS_RUN_ENTITIES         : bool = True
    ANALYSIS_RUN_COOCCURRENCE     : bool = True
    ANALYSIS_RUN_TRENDS           : bool = True
    ANALYSIS_RUN_OUTLIERS         : bool = True
    ANALYSIS_RUN_SENTIMENT_TOPICS : bool = True

    # --- Análisis: thresholds específicos ---
    TOP_N_TOPICOS_POR_DESTINO : int = 15
    COOC_MIN_DOCS_ENTIDAD     : int = 10
    COOC_MIN_TERMINOS         : int = 3

    # --- Sentiment topic modeling ---
    SENTIMENT_TOPIC_MIN_CLUSTER_SIZE_FLOOR  : int   = 30
    SENTIMENT_TOPIC_MIN_CLUSTER_SIZE_FACTOR : int   = 200
    SENTIMENT_TOPIC_UMAP2D_MAX_NEIGHBORS    : int   = 50
    SENTIMENT_TOPIC_UMAP2D_NEIGHBORS_FACTOR : int   = 200
    SENTIMENT_TOPIC_UMAP2D_MIN_DIST         : float = 0.1
    SENTIMENT_TOPIC_UMAP2D_METRIC           : str   = 'euclidean'
    SENTIMENT_TOPIC_KEYWORDS_TOP_N          : int   = 10
    SENTIMENT_TOPIC_FRECUENCIA_TOP_N        : int   = 30
    PRECIO_VALOR_COSTO_CONCEPTO             : str   = 'precio valor costo dinero pago tarifa caro barato económico'

    # --- Visualización ---
    VIZ_SCATTER_SAMPLE_N   : int = 4000
    VIZ_COOC_MIN_WEIGHT    : int = 5
    VIZ_KEYWORDS_RANK_LIMIT: int = 10
    VIZ_POLARITIES_TOP_N   : int = 25
    VIZ_HEATMAP_MIN_DOCS   : int = 10
    VIZ_SENTIMENT_POS_COLOR: str = "#2E7D32"
    VIZ_SENTIMENT_NEU_COLOR: str = "#F9A825"
    VIZ_SENTIMENT_NEG_COLOR: str = "#C62828"
    VIZ_FALSE_POSITIVES: list = [
        "fuimos", "ademas", "llegamos", "encima", "falto", "buena", "tome",
        "bikiniselfie bikiniseksi bikinisummer summerbodyready",
        "lapazmx bikinishot", "visitacancun agencia gestion",
        "beachlife", "visithuatulco", "huatulcooaxaca huatulco", "bcs",
    ]

    @classmethod
    def load_from_yaml(cls, path: 'str | Path') -> None:
        """
        Carga parámetros desde un YAML de configuración y los inyecta en Params.
        Solo sobreescribe las claves presentes en el YAML; el resto conserva su default.

        Estructura esperada del YAML (ver pipeline_config.yaml):
            global:
            step_05_semantic.bertopic.{umap, hdbscan}
            step_06_clustering
            step_08_analysis.{sentiment, outliers, sentiment_topics}
        """
        import yaml
        _log = logging.getLogger(__name__)
        cfg_path = Path(path)
        if not cfg_path.exists():
            _log.warning('Config YAML no encontrado: %s — usando defaults', cfg_path)
            return

        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

        _log.info('Cargando configuración automática desde %s', cfg_path)

        def _set(key: str, value) -> None:
            setattr(cls, key, value)
            _log.info('  config: %s = %r', key, value)

        g = cfg.get('global', {})
        if 'random_state'    in g: _set('RANDOM_STATE',    g['random_state'])
        if 'embedding_model' in g: _set('EMBEDDING_MODEL', g['embedding_model'])

        s01 = cfg.get('step_01_preprocessing', {})
        ld  = s01.get('language_detection', {})
        if 'short_text_threshold' in ld: _set('LANGDETECT_SHORT_THRESHOLD', ld['short_text_threshold'])
        if 'confidence_high'      in ld: _set('LANGDETECT_CONF_HIGH',       ld['confidence_high'])
        if 'confidence_normal'    in ld: _set('LANGDETECT_CONF_NORMAL',     ld['confidence_normal'])
        if 'confidence_secondary' in ld: _set('LANGDETECT_CONF_SECONDARY',  ld['confidence_secondary'])

        bt = cfg.get('step_05_semantic', {}).get('bertopic', {})
        u  = bt.get('umap', {})
        if 'n_neighbors'  in u: _set('BERTOPIC_UMAP_N_NEIGHBORS',  u['n_neighbors'])
        if 'n_components' in u: _set('BERTOPIC_UMAP_N_COMPONENTS', u['n_components'])
        if 'min_dist'     in u: _set('BERTOPIC_UMAP_MIN_DIST',     u['min_dist'])
        if 'metric'       in u: _set('BERTOPIC_UMAP_METRIC',       u['metric'])
        h = bt.get('hdbscan', {})
        if 'min_cluster_size'        in h: _set('BERTOPIC_HDBSCAN_MIN_CLUSTER_SIZE',  h['min_cluster_size'])
        if 'metric'                  in h: _set('BERTOPIC_HDBSCAN_METRIC',            h['metric'])
        if 'cluster_selection_method'in h: _set('BERTOPIC_HDBSCAN_CLUSTER_SELECTION', h['cluster_selection_method'])
        if 'calculate_probabilities' in h: _set('BERTOPIC_CALCULATE_PROBS',           h['calculate_probabilities'])
        if 'batch_size'              in bt: _set('BERTOPIC_BATCH_SIZE',               bt['batch_size'])

        tfidf = cfg.get('step_05_semantic', {}).get('tfidf', {})
        if 'min_df'       in tfidf: _set('TFIDF_MIN_DF',       tfidf['min_df'])
        if 'max_df'       in tfidf: _set('TFIDF_MAX_DF',       tfidf['max_df'])
        if 'ngram_min'    in tfidf: _set('TFIDF_NGRAM_MIN',    tfidf['ngram_min'])
        if 'ngram_max'    in tfidf: _set('TFIDF_NGRAM_MAX',    tfidf['ngram_max'])
        if 'max_features' in tfidf: _set('TFIDF_MAX_FEATURES', tfidf['max_features'])

        yk = cfg.get('step_05_semantic', {}).get('yake', {})
        if 'language' in yk: _set('YAKE_LANGUAGE', yk['language'])
        if 'ngram'    in yk: _set('YAKE_NGRAM',    yk['ngram'])
        if 'top_k'    in yk: _set('YAKE_TOP_K',    yk['top_k'])

        s05_ner = cfg.get('step_05_semantic', {}).get('ner', {})
        if 'merge_threshold' in s05_ner: _set('NER_MERGE_THRESHOLD', s05_ner['merge_threshold'])

        s07 = cfg.get('step_07_enrichment', {})
        hier = s07.get('hierarchy', {})
        if 'linkage_method' in hier: _set('HIERARCHY_LINKAGE_METHOD', hier['linkage_method'])
        if 'linkage_metric' in hier: _set('HIERARCHY_LINKAGE_METRIC', hier['linkage_metric'])
        llm = s07.get('llm', {})
        if 'n_ctx'         in llm: _set('TOPIC_NAMING_N_CTX',         llm['n_ctx'])
        if 'n_threads'     in llm: _set('TOPIC_NAMING_N_THREADS',     llm['n_threads'])
        if 'n_gpu_layers'  in llm: _set('TOPIC_NAMING_N_GPU_LAYERS',  llm['n_gpu_layers'])
        if 'temperature'   in llm: _set('TOPIC_NAMING_TEMPERATURE',   llm['temperature'])
        if 'max_tokens'    in llm: _set('TOPIC_NAMING_MAX_TOKENS',    llm['max_tokens'])
        if 'top_docs'      in llm: _set('TOPIC_NAMING_TOP_DOCS',      llm['top_docs'])

        s02 = cfg.get('step_02_translation', {})
        if 'batch_size'  in s02: _set('TRANSLATION_BATCH_SIZE',  s02['batch_size'])
        if 'max_length'  in s02: _set('TRANSLATION_MAX_LENGTH',  s02['max_length'])
        if 'num_beams'   in s02: _set('TRANSLATION_NUM_BEAMS',   s02['num_beams'])

        s04 = cfg.get('step_04_features', {})
        if 'spacy_batch_size' in s04: _set('SPACY_BATCH_SIZE', s04['spacy_batch_size'])

        s06 = cfg.get('step_06_clustering', {})
        if 'svd_components'         in s06: _set('SVD_COMPONENTS',          s06['svd_components'])
        if 'umap_components'        in s06: _set('UMAP_COMPONENTS',         s06['umap_components'])
        if 'jerarquico_knn_vecinos' in s06: _set('JERARQUICO_KNN_VECINOS',  s06['jerarquico_knn_vecinos'])
        if 'alpha_kmeans'           in s06: _set('ALPHA_KMEANS',            s06['alpha_kmeans'])
        if 'max_cluster_pct'        in s06: _set('MAX_CLUSTER_PCT',         s06['max_cluster_pct'])

        s08   = cfg.get('step_08_analysis', {})
        sent  = s08.get('sentiment', {})
        if 'idioma'     in sent: _set('SENTIMENT_IDIOMA',     sent['idioma'])
        if 'metodo'     in sent: _set('SENTIMENT_METODO',     sent['metodo'])
        if 'texto'      in sent: _set('SENTIMENT_TEXTO',      sent['texto'])
        if 'features'   in sent: _set('SENTIMENT_FEATURES',   sent['features'])
        if 'batch_size' in sent: _set('SENTIMENT_BATCH_SIZE', sent['batch_size'])
        if 'max_length' in sent: _set('SENTIMENT_MAX_LENGTH', sent['max_length'])
        if 'lr_max_iter'in sent: _set('SENTIMENT_LR_MAX_ITER',sent['lr_max_iter'])
        if 'lr_c'       in sent: _set('SENTIMENT_LR_C',       sent['lr_c'])
        if 'lr_solver'  in sent: _set('SENTIMENT_LR_SOLVER',  sent['lr_solver'])

        out = s08.get('outliers', {})
        if 'contamination'   in out: _set('OUTLIER_CONTAMINATION',  out['contamination'])
        if 'knn_neighbors'   in out: _set('OUTLIER_KNN_NEIGHBORS',  out['knn_neighbors'])
        if 'umbral_ensemble' in out: _set('OUTLIER_UMBRAL_ENSEMBLE',out['umbral_ensemble'])
        if 'svm_max_train'   in out: _set('OUTLIER_SVM_MAX_TRAIN',  out['svm_max_train'])

        st = s08.get('sentiment_topics', {})
        if 'min_docs'                in st: _set('SENTIMENT_TOPIC_MIN_DOCS',   st['min_docs'])
        if 'precio_valor_costo_top_n'in st: _set('PRECIO_VALOR_COSTO_TOP_N',   st['precio_valor_costo_top_n'])
        if 'metodo'                  in st: _set('SENTIMENT_TOPIC_METODO',      st['metodo'])
        if 'nr_topics'               in st: _set('SENTIMENT_TOPIC_NR_TOPICS',   st['nr_topics'])
        if 'min_cluster_size_floor'  in st: _set('SENTIMENT_TOPIC_MIN_CLUSTER_SIZE_FLOOR',  st['min_cluster_size_floor'])
        if 'min_cluster_size_factor' in st: _set('SENTIMENT_TOPIC_MIN_CLUSTER_SIZE_FACTOR', st['min_cluster_size_factor'])
        umap2d = st.get('umap2d', {})
        if 'max_neighbors'    in umap2d: _set('SENTIMENT_TOPIC_UMAP2D_MAX_NEIGHBORS',    umap2d['max_neighbors'])
        if 'neighbors_factor' in umap2d: _set('SENTIMENT_TOPIC_UMAP2D_NEIGHBORS_FACTOR', umap2d['neighbors_factor'])
        if 'min_dist'         in umap2d: _set('SENTIMENT_TOPIC_UMAP2D_MIN_DIST',         umap2d['min_dist'])
        if 'metric'           in umap2d: _set('SENTIMENT_TOPIC_UMAP2D_METRIC',           umap2d['metric'])

    @classmethod
    def set_from_args(cls, args) -> None:
        """
        Recibe el namespace de argparse (o cualquier objeto con los
        atributos input_csv, text_column, language, title, palette)
        y sobreescribe los parámetros de clase correspondientes.

        También actualiza Paths.RAW_COMPLETE_CSV con la ruta del CSV
        del usuario, de modo que el resto del pipeline lo use
        automáticamente sin tocar otros módulos.

        Llamar una única vez desde main.py, antes de ejecutar steps.
        """
        cls.INPUT_CSV = Path(args.input_csv).resolve()
        cls.TEXT_COLUMN = args.text_column
        cls.LANGUAGE = args.language
        cls.REPORT_TITLE = args.title
        cls.COLOR_PALETTE = args.palette

        # Parámetros opcionales del análisis de sentimiento
        if getattr(args, 'sentiment_idioma', None):
            cls.SENTIMENT_IDIOMA = args.sentiment_idioma
        if getattr(args, 'sentiment_metodo', None):
            cls.SENTIMENT_METODO = args.sentiment_metodo
        if getattr(args, 'sentiment_texto', None):
            cls.SENTIMENT_TEXTO = args.sentiment_texto
        if getattr(args, 'sentiment_features', None):
            cls.SENTIMENT_FEATURES = args.sentiment_features

        # Redirigir el CSV de entrada para que el pipeline lo use
        Paths.RAW_COMPLETE_CSV = cls.INPUT_CSV

        _logger = logging.getLogger(__name__)
        _logger.info("Parámetros de entrada cargados:")
        _logger.info("  CSV         : %s", cls.INPUT_CSV)
        _logger.info("  Columna     : %s", cls.TEXT_COLUMN)
        _logger.info("  Idioma      : %s", cls.LANGUAGE)
        _logger.info("  Título      : %s", cls.REPORT_TITLE)
        _logger.info("  Paleta      : %s", cls.COLOR_PALETTE)
        if cls.LANGUAGE not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Idioma no válido: {cls.LANGUAGE}")
        # Advertir si la paleta no es una de las accesibles recomendadas
        if cls.COLOR_PALETTE not in ACCESSIBLE_PALETTES:
            _logger.warning(
                "La paleta '%s' no está en las opciones accesibles recomendadas "
                "(%s). Considera usar una de ellas para mayor accesibilidad visual.",
                cls.COLOR_PALETTE,
                ", ".join(sorted(ACCESSIBLE_PALETTES)),
            )


def resolve_device(requested: str | None = None) -> str:
    """
    Resuelve el dispositivo de cómputo según Params.DEVICE (o el valor pasado).

    "auto" -> detecta GPU automáticamente; si no hay, usa CPU
    "gpu"  -> verifica que GPU exista y funcione; si falla limpia contexto
              y reintenta; si no puede recuperarse cae a CPU con warning
    "cpu"  -> usa CPU directamente, ignora GPU aunque exista
    """
    import torch
    _log = logging.getLogger(__name__)
    req = (requested or Params.DEVICE).lower()

    if req == "cpu":
        return "cpu"

    if req == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            _log.debug("GPU no disponible — usando CPU")
        return device

    if req == "gpu":
        if not torch.cuda.is_available():
            _log.warning("DEVICE=gpu solicitado pero GPU no disponible — usando CPU")
            return "cpu"
        try:
            _t = torch.zeros(1).cuda()
            del _t
            torch.cuda.empty_cache()
            return "cuda"
        except Exception as exc:
            _log.warning("GPU falló (%s) — limpiando contexto CUDA y usando CPU", exc)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return "cpu"

    _log.warning("DEVICE='%s' no reconocido — usando 'auto'", req)
    return "cuda" if torch.cuda.is_available() else "cpu"


class LoggingConfig:
    """
    Configuración estándar de logging para todo el proyecto.
    Usar setup_logging() al inicio de main.py.
    """

    LEVEL   = logging.INFO
    FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%H:%M:%S"

    @staticmethod
    def setup() -> None:
        """
        Inicializa el logging con el formato estándar del proyecto.
        Llamar una sola vez desde main.py.
        """
        logging.basicConfig(
            level=LoggingConfig.LEVEL,
            format=LoggingConfig.FORMAT,
            datefmt=LoggingConfig.DATEFMT,
        )


def ensure_data_directories() -> None:
    """
    Crea todas las carpetas de datos si no existen.
    Llamar desde main.py antes de ejecutar cualquier step.
    """
    dirs = [
        Paths.RAW_GOOGLE,
        Paths.RAW_TRIP,
        Paths.RAW_INSTA,
        Paths.DATA_SPANISH_DIR,
        Paths.DATA_ENGLISH_DIR,
        Paths.DATA_MIXED_DIR,
        Paths.DATA_FRENCH_DIR,
        Paths.TRANSLATIONS_DIR,
        Paths.UNIFIED_DIR,
        Paths.PROCESSED_DIR,
        Paths.FEATURES_DIR,
        Paths.MODELS_DIR,
        Paths.RESULTS_DIR,
        Paths.CLUSTERING_DIR,
        Paths.CLUSTERING_EMBEDDINGS_DIR,
        Paths.CLUSTERING_FEATURES_DIR,
        Paths.CLUSTERING_TFIDF_DIR,
        Paths.CLUSTERING_YAKE_DIR,
        Paths.ENRICHMENT_DIR,
        Paths.ANALYSIS_DIR,
        Paths.SENTIMENT_DIR,
        Paths.OUTLIERS_DIR,
        Paths.SENTIMENT_TOPICS_DIR,
        Paths.ENTITIES_DIR,
        Paths.COOCCURRENCE_DIR,
        Paths.TRENDS_DIR,
        Paths.VISUALIZATION_DIR,
        Paths.ENRICHMENT_ACTIVE_DIR,
    ]

    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logging.getLogger(__name__).warning(
                "No se pudo crear la carpeta %s: %s", directory, error
            )
