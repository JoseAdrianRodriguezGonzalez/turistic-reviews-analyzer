"""
base_step.py
------------
Clase base abstracta que define el contrato que todo Step del pipeline
debe cumplir.

Cada Step concreto hereda de BaseStep e implementa:
    - input_paths  : lista de rutas que el step necesita para correr
    - output_paths : lista de rutas que el step produce al terminar
    - _run         : lógica de ejecución (llama al módulo correspondiente)

El método público run() orquesta validate() -> can_skip() -> _run()
con manejo de errores y logging incluidos. Los Steps concretos no
necesitan preocuparse por eso.

Uso:
    class StepClustering(BaseStep):
        name = "clustering"

        @property
        def input_paths(self):
            return [Paths.FEATURES_NLP_CSV]

        @property
        def output_paths(self):
            return [Paths.CLUSTERING_COMPARACION_CSV]

        def _run(self):
            from clustering.clustering_pipeline import run_clustering_pipeline
            run_clustering_pipeline()
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """
    Contrato base para todos los pasos del pipeline.

    Atributos de clase que cada Step concreto debe definir:
        name (str) : identificador único del step, usado en logs y flags de CLI
    """

    # Cada subclase define su propio nombre
    name: str = "base"

    # PROPIEDADES ABSTRACTAS
    # Cada Step concreto debe implementarlas

    @property
    @abstractmethod
    def input_paths(self) -> list[Path]:
        """
        Lista de rutas que deben existir para que este step pueda correr.
        Si alguna falta, validate() retorna False y el step no se ejecuta.
        """
        ...

    @property
    @abstractmethod
    def output_paths(self) -> list[Path]:
        """
        Lista de rutas que este step produce al terminar.
        Si todas existen, can_skip() retorna True (ya fue ejecutado antes).
        """
        ...

    @abstractmethod
    def _run(self) -> None:
        """
        Lógica de ejecución del step.
        Solo llama al módulo correspondiente — no define lógica propia.
        """
        ...

    # MÉTODOS PÚBLICOS
    # Orquestan validate -> can_skip -> _run con logging y try/catch

    def validate(self) -> bool:
        """
        Verifica que todos los inputs necesarios existen antes de correr.
        Retorna True si el step puede ejecutarse, False si falta algún input.
        """
        missing = [p for p in self.input_paths if not p.exists()]

        if missing:
            for path in missing:
                logger.warning(
                    "[%s] Input faltante: %s", self.name, path
                )
            return False

        return True

    def can_skip(self) -> bool:
        """
        Retorna True si todos los outputs ya existen (step ya fue ejecutado).
        En ese caso, run() saltará la ejecución a menos que se use --force.
        """
        return all(p.exists() for p in self.output_paths)

    def run(self, force: bool = False) -> bool:
        """
        Punto de entrada público del step.

        Flujo:
            1. Verifica inputs con validate()
            2. Si can_skip() y no force -> omite ejecución
            3. Ejecuta _run() con medición de tiempo
            4. Captura cualquier excepción sin detener el pipeline

        Parámetros:
            force : si True, ejecuta aunque los outputs ya existan

        Retorna:
            True  si el step terminó correctamente (o fue saltado)
            False si falló o los inputs no estaban disponibles
        """
        logger.info("=" * 55)
        logger.info("[%s] Iniciando step", self.name.upper())

        #  Validar inputs 
        if not self.validate():
            logger.error(
                "[%s] Step cancelado — faltan inputs requeridos", self.name
            )
            return False

        #  Verificar si se puede saltar 
        if self.can_skip() and not force:
            logger.info(
                "[%s] Output ya existe — step omitido (usa --force para re-ejecutar)",
                self.name,
            )
            return True

        #  Ejecutar 
        inicio = time.time()
        try:
            self._run()
            duracion = time.time() - inicio
            logger.info(
                "[%s] Step completado en %.1f s", self.name, duracion
            )
            return True

        except Exception as error:
            duracion = time.time() - inicio
            logger.error(
                "[%s] Step falló después de %.1f s — %s: %s",
                self.name,
                duracion,
                type(error).__name__,
                error,
            )
            return False

    # REPRESENTACIÓN
    

    def status(self) -> dict:
        """
        Retorna un diccionario con el estado actual del step.
        Usado por main.py --status para mostrar el estado del pipeline.
        """
        inputs_ok   = all(p.exists() for p in self.input_paths)
        outputs_ok  = all(p.exists() for p in self.output_paths)

        missing_inputs  = [str(p) for p in self.input_paths  if not p.exists()]
        missing_outputs = [str(p) for p in self.output_paths if not p.exists()]

        return {
            "name"           : self.name,
            "listo_para_correr" : inputs_ok,
            "ya_ejecutado"      : outputs_ok,
            "inputs_faltantes"  : missing_inputs,
            "outputs_faltantes" : missing_outputs,
        }

    def __repr__(self) -> str:
        return f"<Step: {self.name}>"