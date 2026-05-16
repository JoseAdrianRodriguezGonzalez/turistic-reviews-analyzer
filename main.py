"""
main.py
-------
Punto de entrada del pipeline de análisis NLP de turismo.

Orquesta los 9 steps del pipeline con soporte para:
    - Ejecutar todo con un solo comando
    - Seleccionar steps específicos con --steps
    - Forzar re-ejecución con --force
    - Ver el estado actual del pipeline con --status

Uso:
    # Ejecutar todo el pipeline
    python main.py

    # Ejecutar solo pasos específicos
    python main.py --steps preprocessing translation vocabulary

    # Forzar re-ejecución aunque los outputs ya existan
    python main.py --force

    # Forzar solo en pasos específicos
    python main.py --steps clustering --force

    # Ver el estado actual de cada step
    python main.py --status

    # Ver steps disponibles
    python main.py --list
"""

import argparse
import logging
import sys
import time

from config import LoggingConfig, Paths, ensure_data_directories
from pipeline import (
    StepAnalysis,
    StepClustering,
    StepEnrichment,
    StepFeatures,
    StepPreprocessing,
    StepSemantic,
    StepTranslation,
    StepVocabulary,
    StepVisualization,
)

# REGISTRO DE STEPS
# Orden de ejecución del pipeline completo.
# Agregar o quitar steps aquí sin tocar nada más.

STEPS = [
    StepPreprocessing(),
    StepTranslation(),
    StepVocabulary(),
    StepFeatures(),
    StepSemantic(),
    StepClustering(),
    StepEnrichment(),
    StepAnalysis(),
    StepVisualization(),
]

# Índice por nombre para búsqueda rápida desde CLI
STEPS_POR_NOMBRE: dict[str, object] = {step.name: step for step in STEPS}

logger = logging.getLogger(__name__)


# CLI

def construir_parser() -> argparse.ArgumentParser:
    """
    Define los argumentos aceptados por el CLI.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Pipeline de análisis NLP de turismo — ejecuta todos o pasos específicos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                                 # ejecutar todo
  python main.py --steps preprocessing           # solo preprocesamiento
  python main.py --steps clustering analysis     # clustering y análisis
  python main.py --force                         # re-ejecutar todo
  python main.py --steps clustering --force      # re-ejecutar solo clustering
  python main.py --status                        # ver estado del pipeline
  python main.py --list                          # ver steps disponibles
        """,
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="STEP",
        help="Nombres de los steps a ejecutar (separados por espacio).",
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ejecutar aunque los outputs ya existan.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Mostrar el estado actual de cada step y salir.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar los steps disponibles y salir.",
    )

    return parser


# ACCIONES DEL CLI

def mostrar_lista() -> None:
    """
    Imprime los steps disponibles con su descripción.
    """
    print("\nSteps disponibles:\n")
    for i, step in enumerate(STEPS, start=1):
        print(f"  {i:02d}. {step.name}")
    print()


def mostrar_status() -> None:
    """
    Imprime el estado actual de cada step:
    si está listo para correr y si ya fue ejecutado.
    """
    print("\nEstado del pipeline:\n")
    print(f"  {'STEP':<25} {'LISTO PARA CORRER':<22} {'YA EJECUTADO'}")
    print(f"  {'-'*25} {'-'*22} {'-'*15}")

    for step in STEPS:
        try:
            estado = step.status()
            listo  = "✓ sí" if estado["listo_para_correr"] else "✗ faltan inputs"
            ejecutado = "✓ sí" if estado["ya_ejecutado"]      else "✗ pendiente"
            print(f"  {step.name:<25} {listo:<22} {ejecutado}")

            # Mostrar inputs faltantes si los hay
            for faltante in estado["inputs_faltantes"]:
                print(f"    ↳ input faltante: {faltante}")

        except Exception as error:
            print(f"  {step.name:<25} ERROR al obtener estado: {error}")

    print()


def resolver_steps(nombres: list[str] | None) -> list:
    """
    Retorna la lista de objetos Step a ejecutar.
    Si nombres es None, retorna todos los steps en orden.
    Si hay nombres inválidos, los reporta y sale.
    """
    if nombres is None:
        return STEPS

    steps_seleccionados = []
    invalidos = []

    for nombre in nombres:
        if nombre in STEPS_POR_NOMBRE:
            steps_seleccionados.append(STEPS_POR_NOMBRE[nombre])
        else:
            invalidos.append(nombre)

    if invalidos:
        print(f"\nError: steps no reconocidos: {invalidos}")
        print(f"Steps válidos: {list(STEPS_POR_NOMBRE.keys())}")
        sys.exit(1)

    return steps_seleccionados


# EJECUCIÓN DEL PIPELINE

def ejecutar_pipeline(steps: list, force: bool) -> None:
    """
    Ejecuta la lista de steps en orden.
    Registra cuántos tuvieron éxito y cuántos fallaron.
    No detiene la ejecución si un step falla.
    """
    total    = len(steps)
    exitosos = 0
    fallidos = []

    inicio_total = time.time()

    logger.info("Pipeline iniciado — %d step(s) a ejecutar", total)

    for step in steps:
        exito = step.run(force=force)
        if exito:
            exitosos += 1
        else:
            fallidos.append(step.name)

    duracion_total = time.time() - inicio_total

    # --- Resumen final ---
    logger.info("=" * 55)
    logger.info(
        "Pipeline finalizado en %.1f s — %d/%d steps exitosos",
        duracion_total, exitosos, total,
    )

    if fallidos:
        logger.warning("Steps con error: %s", fallidos)
    else:
        logger.info("Todos los steps completados sin errores.")


# PUNTO DE ENTRADA


def main() -> None:
    """
    Función principal. Parsea argumentos y ejecuta la acción correspondiente.
    """
    LoggingConfig.setup()

    parser = construir_parser()
    args   = parser.parse_args()

    # --- Acciones informativas (no ejecutan el pipeline) ---
    if args.list:
        mostrar_lista()
        sys.exit(0)

    if args.status:
        mostrar_status()
        sys.exit(0)

    # --- Preparar la caja de datos ---
    try:
        ensure_data_directories()
    except Exception as error:
        logger.error("No se pudieron crear las carpetas de datos: %s", error)
        sys.exit(1)

    # --- Resolver qué steps correr ---
    steps = resolver_steps(args.steps)

    # --- Ejecutar ---
    ejecutar_pipeline(steps=steps, force=args.force)


if __name__ == "__main__":
    main()