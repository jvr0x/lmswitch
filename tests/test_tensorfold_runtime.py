"""TensorFold runtime registers and the GLM recipe parses."""

from lmswitch.runtimes.base import runtime_registry
from lmswitch.runtimes.tensorfold import (
    TensorFoldRuntime,
    container_name,
    missing_weights,
    repo_dir,
    weight_env,
)
from lmswitch.system.io import CONF_DIR, _load_yaml


def test_tensorfold_registered():
    assert runtime_registry.lookup("tensorfold") is TensorFoldRuntime


def test_glm_tensorfold_recipe_parses():
    path = CONF_DIR / "glm5.3-flash-exl3-tensorfold-dual.yaml"
    cfg = _load_yaml(path)
    assert cfg["runtime"] == "tensorfold"
    assert int(cfg["port"]) == 8890
    assert container_name(cfg) == "glm53-flash-tf"
    assert repo_dir(cfg).joinpath("start.sh").is_file()
    env = weight_env(cfg)
    assert env["MODEL_DIR"].endswith("Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw")
    assert env["WORKER_MODEL_DIR"].endswith("models/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw")
    assert "models-gigabyte" not in env["WORKER_MODEL_DIR"]
    assert env["DFLASH2_DIR"].endswith("incoai/GLM-5.3-Flash-DFlash2")
    assert env["WORKER_DFLASH2_DIR"].startswith("/home/")


def test_missing_weights_names_an_absent_tree(tmp_path):
    present = tmp_path / "ckpt"
    present.mkdir()
    (present / "config.json").write_text("{}")
    absent = tmp_path / "missing"
    yaml = {
        "model_path": str(present),
        "worker_model_path": str(absent),
        "worker": "",
    }
    assert "WORKER_MODEL_DIR" in missing_weights(yaml)
    yaml["worker_model_path"] = str(present)
    assert missing_weights(yaml) == ""
