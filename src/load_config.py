import json
import os


def load_config(config_path: str = "config.json") -> dict:
    """Load and validate configuration parameters for MUST CAS login, CSMAR API, and feature cleaning."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for section in ("school_auth", "csmar_api", "pipeline_settings", "split_settings"):
        if section not in config:
            raise KeyError(f"Missing required section '{section}' in {config_path}")
    return config
