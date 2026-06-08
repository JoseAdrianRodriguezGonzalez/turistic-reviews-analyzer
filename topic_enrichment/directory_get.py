import json
from config import Paths

def get_active_enrichment_dir():
    config_path = Paths.ENRICHMENT_DIR / "active_model.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"No existe {config_path}. Ejecuta primero el clustering (step 6)."
        )

    with open(config_path) as f:
        cfg = json.load(f)
    model_dir=parse_model_dir(cfg["modelo"])
    return Paths.ENRICHMENT_DIR / cfg["fuente"] / model_dir

def parse_model_dir(model_str: str) -> str:
    # Ej: "kmeans|k=9"
    if "|" not in model_str:
        return model_str  # por si viene limpio

    modelo, params = model_str.split("|", 1)

    if modelo == "kmeans":
        # extraer k
        for p in params.split(","):
            if p.startswith("k="):
                k = p.split("=")[1]
                return f"{modelo}_k{k}"

    elif modelo == "hdbscan":
        return "hdbscan"

    elif modelo == "jerarquico":
        return "jerarquico"

    return modelo  # fallback