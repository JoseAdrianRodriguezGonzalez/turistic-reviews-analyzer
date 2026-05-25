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

    Params.INPUT_CSV        → Path al archivo CSV del usuario
    Params.TEXT_COLUMN      → nombre de la columna de comentarios
    Params.LANGUAGE         → código de idioma (es | en | fr)
    Params.REPORT_TITLE     → título del reporte
    Params.COLOR_PALETTE    → paleta de colores para gráficas
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
    # Ruta activa usada por las visualizaciones (KMeans k=8 sobre embeddings)
    ENRICHMENT_ACTIVE_DIR        = ENRICHMENT_DIR / "embeddings" / "kmeans_k8"
    ENRICHMENT_KEYWORDS_CSV      = ENRICHMENT_ACTIVE_DIR / "keywords_por_cluster.csv"
    # Análisis
    ANALYSIS_DIR                 = DATA_DIR / "analysis"
    ANALYSIS_RESUMEN_CSV         = ANALYSIS_DIR / "resumen_analysis.csv"
    SENTIMENT_DIR                = ANALYSIS_DIR / "sentiment"
    CORPUS_CON_SENTIMIENTO_CSV   = SENTIMENT_DIR / "corpus_con_sentimiento.csv"
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


# PARÁMETROS DEL PIPELINE

ACCESSIBLE_PALETTES = {"viridis", "cividis", "plasma", "inferno"}
SUPPORTED_LANGUAGES = {"es", "en", "fr"}

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

    #  Texto
    COLUMNA_TEXTO        = "comentario_clean"
    COLUMNA_COMENTARIO   = "comentario"

    #  Idiomas
    LANG_SPANISH = "es"
    LANG_ENGLISH = "en"
    LANG_FRENCH  = "fr"
    LANG_MIXED   = "mixed"

    #  Vocabulario
    MIN_FREQ_VOCAB       = 2
    MAX_VOCAB_SIZE       = None     # None = sin límite

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

    # --- Topic enrichment ---
    TOP_N_KEYWORDS       = 15
    TOP_K_DOCS_REPR      = 5

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
    CLUSTERING_ACTIVO_K      = 8
    CLUSTERING_ACTIVO_KEY    = "kmeans|k=8"

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

    # --- Colores de clusters (KMeans k=8) ---
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

    # --- Análisis de sentimiento (transformer) ---
    # Idioma del modelo: 'es' (español) o 'multi' (multilingüe)
    SENTIMENT_IDIOMA   : str       = 'es'
    # Método: 'rapido' (solo transformer) o 'robusto' (transformer + clasificador supervisado)
    SENTIMENT_METODO   : str       = 'rapido'
    # Texto que se pasa al transformer: 'original' o 'cleaned'
    SENTIMENT_TEXTO    : str       = 'original'
    # Features para el método robusto (cualquier combinación de embeddings, tfidf, yake)
    SENTIMENT_FEATURES : list[str] = ['embeddings']

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

        # Advertir si la paleta no es una de las accesibles recomendadas
        if cls.COLOR_PALETTE not in ACCESSIBLE_PALETTES:
            _logger.warning(
                "La paleta '%s' no está en las opciones accesibles recomendadas "
                "(%s). Considera usar una de ellas para mayor accesibilidad visual.",
                cls.COLOR_PALETTE,
                ", ".join(sorted(ACCESSIBLE_PALETTES)),
            )


# ======================================================
# CONFIGURACIÓN DE LOGGING
# ======================================================

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


# ======================================================
# UTILIDAD: verificar que la caja de datos existe
# ======================================================

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
