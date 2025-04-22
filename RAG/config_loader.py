import yaml
import os

# === Load file cấu hình chính ===
with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)

# === Load system prompts (nếu có dùng) ===
with open("system_prompts.yaml", "r") as f:
    system_prompts = yaml.safe_load(f)

# === Đảm bảo mọi đường dẫn là tuyệt đối ===
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resolve_path(key):
    if key in config_data:
        config_data[key] = os.path.abspath(os.path.join(base_path, config_data[key]))

for path_key in ["CSV_PATH", "VECTOR_DB_PATH", "LLM_CACHE_DIR", "SAVE_RESULTS_PATH"]:
    resolve_path(path_key)

__all__ = ["config_data", "system_prompts"]