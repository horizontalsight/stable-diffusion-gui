"""统一的路径解析与环境准备。

仓库结构:
    <repo>/
        stable-diffusion/   官方源码与资源（原样，勿改）
        app/                本项目代码
        vendor/             第三方依赖副本（不入库）
        output/             运行产物（不入库）
"""

import os
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

UPSTREAM_DIR = REPO_ROOT / "stable-diffusion"

CONFIG_PATH = UPSTREAM_DIR / "configs" / "stable-diffusion" / "v1-inference.yaml"
CKPT_PATH = UPSTREAM_DIR / "models" / "ldm" / "stable-diffusion-v1" / "model.ckpt"

OUTPUT_DIR = REPO_ROOT / "output"
STATE_FILE = OUTPUT_DIR / "main_is_working.txt"
PATH_LIST = OUTPUT_DIR / "path_list.txt"


def samples_dir(date_str=None):
    """出图目录按日期分：output/YYYY-MM-DD/。目录不存在会自动创建。"""
    name = date_str or time.strftime("%Y-%m-%d")
    path = OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path

UI_FILE = APP_DIR / "ui" / "ui_main_start.ui"

# 即使没有 pip install -e stable-diffusion，也能 import ldm
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))


def transformers_cache_dir() -> Path:
    """按 transformers 的规则推断缓存目录。"""
    env = os.environ.get("TRANSFORMERS_CACHE")
    if env:
        return Path(env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "transformers"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "huggingface" / "transformers"


def clip_cache_ready() -> bool:
    """缓存里有 >1GB 的文件，就认为 CLIP 权重已经就位。"""
    try:
        for item in transformers_cache_dir().iterdir():
            if item.is_file() and item.stat().st_size > 1_000_000_000:
                return True
    except OSError:
        pass
    return False


# CLIP 权重已缓存时强制离线。
# 原因: transformers 4.19 的 from_pretrained 每次都会先向 HuggingFace 发 HEAD，
# 一旦代理配置不对就会抛 ProxyError，而且该异常不会回退到本地缓存。
# 换到没有缓存的机器时这里不会触发，仍可正常联网下载。
if "TRANSFORMERS_OFFLINE" not in os.environ and clip_cache_ready():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"