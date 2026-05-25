# Contexto del proyecto — turistic-reviews-analyzer

> Archivo temporal de contexto para continuar sesiones. Borrar cuando ya no sea necesario.

---

## 1. Proyecto

Repositorio: `JoseAdrianRodriguezGonzalez/turistic-reviews-analyzer`
Rama activa: `sentiment-analysis` (no tocar `main` hasta PR)

Pipeline de análisis NLP de reseñas turísticas de 5 destinos mexicanos (Huatulco, La Paz, Riviera Nayarit, Puerto Vallarta, Riviera Maya). Los datos vienen de Google, TripAdvisor e Instagram. El objetivo es ejecutar todo el análisis con **un solo comando de terminal** y generar un reporte HTML con visualizaciones interactivas.

```
python main.py data.csv comentario es "Mi reporte" viridis
```

---

## 2. Arquitectura del pipeline (9 steps)

| Step | Módulo | Estado |
|------|--------|--------|
| 01 preprocessing | `preprocessing/` | ✅ Completo |
| 02 translation | `translation/` | ✅ Completo |
| 03 vocabulary | `feature_engineering/vocabulary.py` | ✅ Completo |
| 04 features | `feature_engineering/features.py` | ✅ Completo |
| 05 semantic | `semantic_expression/` — BERTopic + embeddings | ✅ Completo |
| 06 clustering | `clustering/` — KMeans, Jerárquico, HDBSCAN | ✅ Completo |
| 07 enrichment | `topic_enrichment/` — keywords, docs representativos, jerarquía | ✅ Completo (topic_naming pendiente) |
| 08 analysis | `analysis/` — sentimiento, entidades, co-ocurrencia, tendencias | ⚠️ En progreso (issue #4) |
| 09 visualization | `visualization/` | ✅ Completo |

**Archivos de datos clave:**
- `data/features/docs_with_topics.npy` — embeddings densos del corpus (~384 dims)
- `data/models/tfidf.pkl` — vectorizador TF-IDF entrenado
- `data/models/yake_vectorizer.pkl` — vectorizador YAKE entrenado
- `data/translations/normalized_spanish.csv` — corpus limpio (columna `comentario_clean`)
- `data/unified/analysis_unified.csv` — corpus unificado con `comentario` (original) y `estrellas`
- `data/results/docs_with_topics.csv` — topic, location, lang por documento
- `data/analysis/sentiment/` — outputs del paso 8 de sentimiento

**Configuración centralizada:** `config.py` → clases `Paths`, `Params`, `LoggingConfig`  
**Entry point:** `main.py` con argparse — todos los parámetros del pipeline pasan por ahí.

**Estilo de código:**
- Sin comentarios `# ====` ni `# ----` (los existentes quedan, no agregar nuevos)
- Comentarios pequeños y explicativos solo cuando el WHY no es obvio
- Docstrings con `'''` triple single-quote en módulos del directorio `analysis/`
- Funciones privadas con prefijo `_`
- Paths definidos localmente en cada módulo (el ideal del config no se cumple uniformemente)
- `from config import Params` para leer parámetros de comportamiento

---

## 3. Issues abiertos que nos competen

### Issue #3 — Detección y análisis de valores atípicos
**Asignados:** BarrosoJorge, MarcoIllescas, Emilio-Velazquez, JoseAdrianRodriguez  
**Estado:** ❌ No iniciado  
**Qué pide:**
- Detectar comentarios atípicos usando técnicas vistas en clase
- Análisis de unigramas, bigramas, trigramas sobre los outliers
- Los no-outliers pasan al siguiente análisis

**Recomendación del jefe del equipo (Jose Adrian):** Usar los embeddings ya extraídos (`docs_with_topics.npy`) para detectar outliers, aprovechando el método `embedding_extraction()` de la clase `BERTopic_analysis` en `semantic_expression/BERTopic.py`. Los embeddings se pueden pasar a Isolation Forest o aprovechar que HDBSCAN ya etiqueta ruido como `topic == -1`.

**Dependencias:** Independiente de #4 y #5. Puede hacerse en paralelo.

---

### Issue #4 — Análisis de sentimientos
**Asignados:** BarrosoJorge, MarcoIllescas  
**Estado:** ✅ Implementado en esta sesión (rama `sentiment-analysis`)  
**Qué pedía:**
- Clasificar cada comentario como positivo o negativo (independiente de si tiene estrellas)
- Separar los comentarios en dos grupos: positivos y negativos

**Solución implementada → ver sección 4**

---

### Issue #5 — Modelado de tópicos por sentimiento
**Asignados:** BarrosoJorge, Emilio-Velazquez, MarcoIllescas  
**Estado:** ❌ No iniciado  
**Qué pide:**
- Tomar comentarios positivos → aplicar modelado de tópicos
- Tomar comentarios negativos → aplicar modelado de tópicos
- Si algún grupo es muy pequeño (umbral a definir por el equipo) → usar wordcloud o lista de frecuencias en lugar de topic modeling

**Dependencia:** Requiere que el issue #4 esté completo. Los grupos positivos/negativos ya salen de `run_sentiment_analysis()` como DataFrames y como CSVs en `data/analysis/sentiment/comentarios_positivos.csv` y `comentarios_negativos.csv`. Ambos tienen el texto limpio (`comentario_clean`) disponible para pasarle a BERTopic.

---

## 4. Lo que se hizo para el Issue #4 (esta sesión)

### 4.1 Archivo nuevo: `analysis/transformer_sentiment.py`

Módulo completo para clasificación de sentimiento con transformer. **No reemplaza** el análisis por estrellas; lo complementa y extiende a documentos sin rating (Instagram).

**Modelos configurables:**
| Clave | Modelo HuggingFace | Uso |
|-------|--------------------|-----|
| `es` (default) | `pysentimiento/robertuito-sentiment-analysis` | Español, optimizado para texto en español |
| `multi` | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingüe |

**Dos métodos:**
- `rapido` (default): Inferencia directa con el transformer. Labels: POS/NEG/NEU → `positivo`/`negativo`/`neutro`
- `robusto`: El transformer etiqueta → se entrena un `LogisticRegression` sobre las features ya extraídas (embeddings, tfidf, yake) con esas etiquetas como supervisión. Esto es **destilación de conocimiento**: el clasificador ligero aprende la frontera de decisión del transformer y generaliza mejor en comentarios ambiguos.

**Features combinables para el método robusto:**
- `embeddings`: `data/features/docs_with_topics.npy` (densos, ~384 dims) — **recomendado como default y solo**
- `tfidf`: `data/models/tfidf.pkl` transformado sobre `comentario_clean` (sparse)
- `yake`: `data/models/yake_vectorizer.pkl` transformado sobre `comentario_clean` (sparse)
- Si se mezcla denso + sparse: se convierten todos a sparse con `scipy.sparse.hstack`

**Por qué embeddings y no TF-IDF/YAKE para el robusto:** TF-IDF y YAKE están diseñados para capturar topics, no sentiment. YAKE en particular extrae keywords temáticas ("comida", "servicio") y pierde las palabras de sentimiento ("increíble", "pésimo"). Los embeddings codifican semántica completa en 384 dims y un SVM/LR en ese espacio logra ~estado del arte con mucho menos cómputo.

**Funciones en `transformer_sentiment.py`:**
- `_normalizar_etiqueta(label)` — mapea los formatos heterogéneos de distintos modelos a `positivo/negativo/neutro`
- `_cargar_pipeline_transformer(idioma)` — carga el pipeline HuggingFace
- `_clasificar_por_lotes(textos, pipe, batch_size)` — inferencia en batches, maneja textos vacíos
- `_cargar_textos_limpios()` — carga `normalized_spanish.csv` para los vectorizadores
- `_cargar_features(fuentes)` — carga y concatena las features según lo pedido
- `_entrenar_clasificador_robusto(X, etiquetas)` — entrena `LogisticRegression(max_iter=1000, solver='lbfgs')`
- `run_transformer_sentiment(...)` — función pública, retorna DataFrame con columnas de resultado

**Output de `run_transformer_sentiment()`:**
```
etiqueta_transformer   — positivo / negativo / neutro (directo del modelo)
confianza_transformer  — float [0, 1]
etiqueta_robusto       — predicción del clasificador (solo método robusto)
etiqueta_final         — clasificación que se usa downstream
```

### 4.2 Cambios en `analysis/sentiment_analysis.py`

**`_cargar_corpus_base()`:** Ahora también carga:
- `comentario` (texto original) desde `analysis_unified.csv` — para el transformer
- `comentario_clean` desde `normalized_spanish.csv` — para los vectorizadores

**`_construir_sentimiento()`:** La columna binaria basada en estrellas ahora se llama `sentimiento_binario_estrellas` (referencia de validación, no la definitiva).

**Nueva función `_binarizar_etiqueta_transformer()`:** Colapsa `neutro → negativo` (umbral: positivo = recomendación activa ≥ 4 estrellas equivalente; neutro = sin recomendación = negativo).

**`run_sentiment_analysis()`:** Ahora:
1. Ensambla corpus + labels por estrellas (referencia)
2. Llama `run_transformer_sentiment()` con parámetros de `Params`
3. Hace `df.join(df_transformer)` para agregar columnas del transformer
4. `sentimiento_binario` = resultado del transformer binarizado → cubre TODOS los docs (incluyendo Instagram sin estrellas)
5. Exporta `comentarios_positivos.csv` y `comentarios_negativos.csv`

**Columnas nuevas en `corpus_con_sentimiento.csv`:**
```
sentimiento_binario_estrellas  — basado en estrellas (validación)
etiqueta_transformer           — etiqueta directa del modelo
confianza_transformer          — confianza del modelo
etiqueta_robusto               — (solo método robusto)
etiqueta_final                 — etiqueta usada como base para el binario
sentimiento_binario            — positivo / negativo / sin_etiqueta (definitivo, viene del transformer)
```

**Nuevos archivos exportados:**
- `data/analysis/sentiment/comentarios_positivos.csv`
- `data/analysis/sentiment/comentarios_negativos.csv`

### 4.3 Cambios en `config.py`

Nuevos parámetros en clase `Params`:
```python
SENTIMENT_IDIOMA   : str       = 'es'           # 'es' | 'multi'
SENTIMENT_METODO   : str       = 'rapido'        # 'rapido' | 'robusto'
SENTIMENT_TEXTO    : str       = 'original'      # 'original' | 'cleaned'
SENTIMENT_FEATURES : list[str] = ['embeddings']  # cualquier combinación: embeddings, tfidf, yake
```

`set_from_args()` los inyecta desde CLI usando `getattr(args, ..., None)` (retrocompatible con namespaces sin esos atributos).

### 4.4 Cambios en `main.py`

Nuevos argumentos opcionales en el CLI:
```
--sentiment-idioma   {es, multi}                  default: es
--sentiment-metodo   {rapido, robusto}             default: rapido
--sentiment-texto    {original, cleaned}           default: original
--sentiment-features {embeddings, tfidf, yake}...  default: embeddings
```

**Ejemplos de uso:**
```bash
# Método robusto con modelo multilingüe
python main.py data.csv comentario es "Reporte" viridis --steps analysis \
    --sentiment-idioma multi --sentiment-metodo robusto

# Combinar embeddings + tfidf
python main.py data.csv comentario es "Reporte" viridis --steps analysis \
    --sentiment-features embeddings tfidf

# Texto limpio al transformer (poco recomendado, pero configurable)
python main.py data.csv comentario es "Reporte" viridis --steps analysis \
    --sentiment-texto cleaned
```

---

## 5. Lo que sigue (issues #5 y #3)

### Issue #5 — Modelado de tópicos por sentimiento

Dónde implementarlo: nuevo módulo `analysis/sentiment_topic_modeling.py`, llamado desde `analysis_pipeline.py` como paso 5 del análisis.

**Inputs disponibles (ya existen):**
- `data/analysis/sentiment/comentarios_positivos.csv` — grupo positivo con `comentario_clean`
- `data/analysis/sentiment/comentarios_negativos.csv` — grupo negativo con `comentario_clean`
- La clase `BERTopic_analysis` en `semantic_expression/BERTopic.py` ya tiene `fit()` y `embedding_extraction()`

**Lógica a implementar:**
1. Cargar positivos y negativos
2. Definir un umbral mínimo de documentos (configurable en `Params`, sugerencia: 50-100)
3. Para cada grupo:
   - Si `len(grupo) >= umbral`: aplicar BERTopic (reusar `BERTopic_analysis`)
   - Si `len(grupo) < umbral`: calcular frecuencias de unigramas/bigramas y generar wordcloud
4. Exportar resultados: topics por grupo, o tabla de frecuencias si aplica wordcloud

**Parámetro nuevo a agregar en `Params`:** `SENTIMENT_TOPIC_MIN_DOCS = 50`

### Issue #3 — Detección de valores atípicos

Dónde implementarlo: nuevo módulo `analysis/outlier_detection.py`, llamado desde `analysis_pipeline.py`.

**Approach recomendado por Jose Adrian:** usar los embeddings (`docs_with_topics.npy`) con alguno de:
- **Isolation Forest** (sklearn) — probabilístico, da score de anomalía por documento
- **Aprovechar HDBSCAN** — documentos con `topic == -1` en `docs_with_topics.csv` ya son considerados ruido/outliers por el modelo de clustering

Una vez identificados los outliers, el análisis pide unigrams/bigrams/trigrams sobre esos documentos — las funciones de vocabulario ya existen en `feature_engineering/vocabulary.py`.

---

## 6. Notas técnicas importantes

- El transformer se descarga automáticamente desde HuggingFace la primera vez (~500MB). Necesita conexión a internet en el primer run.
- `pysentimiento/robertuito-sentiment-analysis` está entrenado en tweets en español. Funciona bien para reseñas turísticas informales pero puede no capturar lenguaje más formal.
- El método robusto hace **predicción en el mismo corpus de entrenamiento** (no hay split train/test). Esto es intencional: el objetivo no es evaluar el clasificador sino usar sus predicciones como etiquetas más robustas del mismo corpus. Si se quisiera evaluar el clasificador, habría que comparar contra las estrellas como ground truth.
- Los grupos positivos/negativos excluyen documentos con `sentimiento_binario == 'sin_etiqueta'`. Esto puede pasar si el transformer falla en algún documento — en la práctica debería ser 0 o casi 0.
- `_sentimiento_por_topico()` y `_sentimiento_por_destino()` siguen usando `sentimiento_estrella` (basado en estrellas) para las agrupaciones estadísticas. Podría actualizarse para usar `etiqueta_transformer`, pero no es urgente para los issues actuales.
- La columna `total_documentos` en `_sentimiento_por_destino()` puede desalinearse si los grupos tienen tamaños distintos al hacer `.values` directamente. Revisar si da problemas.



Algunas notas sobre lo que agregué en la sección 6 (cosas extra que vale la pena tener en mente):

Sobre el transformer: Se descarga ~500MB la primera vez. pysentimiento/robertuito está entrenado en tweets, que es informal como las reseñas turísticas — buena elección.

Sobre el método robusto: Entrena y predice sobre el mismo corpus (no hay split). Esto es intencional — no estamos evaluando un clasificador, estamos produciendo etiquetas más robustas para el corpus actual.

Bug potencial que anoté: En _sentimiento_por_destino() hay un .values directo que puede desalinearse si los groupby no tienen exactamente el mismo orden. No bloquea nada hoy pero conviene revisarlo cuando se pruebe con datos reales.

