import os
from pathlib import Path
import yaml


def load(env_id):
    env_name = env_id.split("-")[0]
    somogym_env_name = env_name
    if env_name[-6:] == "Cloner":
        somogym_env_name = env_name[:-6]

    config_file_path = (
        Path(os.path.dirname(__file__))
        / ".."
        / somogym_env_name
        / "benchmark_run_config.yaml"
    )

    with open(config_file_path, "r") as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)
            return {}

    if env_name[-6:] == "Cloner":
        cloner_config_file_path = (
            Path(os.path.dirname(__file__))
            / ".."
            / "SomoBehaviorCloning"
            / env_name.split("-")[0]
            / "benchmark_run_config.yaml"
        )

        with open(cloner_config_file_path, "r") as config_file:
            try:
                cloner_data = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                print(exc)
                return {}
            for key in cloner_data:
                config[key] = cloner_data[key]

    return config
