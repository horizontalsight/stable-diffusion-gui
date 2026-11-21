"""模型加载与生成的核心封装。

把"加载 4.27 GB 权重"和"采样出图"分开：一个进程里只加载一次，
之后每次生成只花采样时间（约 8-9 秒）。GUI 与命令行共用这一层。
"""

import ast
import json
import os
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from pytorch_lightning import seed_everything

import _paths
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

DEFAULTS = {
    "batch_size": 1,
    "scale": 9.0,
    "ddim_steps": 30,
    "H": 512,
    "W": 512,
    "f": 8,
    "n_iter": 1,
    "prompt": "a cat",
}

USAGE = """用法:
  python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']"

8 个参数依次为:
  batch_size, scale, ddim_steps, H, W, f, n_iter, prompt

注意: 数字要写成数字 1，不要写成字符串 '1'，否则会报 TypeError。
"""


def coerce_params(seq):
    """校验并规范化 8 个参数，返回 dict。参数不合法时抛 SystemExit（带说明）。"""
    if not isinstance(seq, (list, tuple)):
        raise SystemExit("参数必须是一个列表\n\n%s" % USAGE)
    if len(seq) != 8:
        raise SystemExit("需要 8 个参数，实际收到 %d 个\n\n%s" % (len(seq), USAGE))

    def need_int(value, name, low, high):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit("%s 必须是整数，收到 %r\n\n%s" % (name, value, USAGE))
        if not low <= value <= high:
            raise SystemExit("%s 超出范围 [%d, %d]，收到 %d\n\n%s" % (name, low, high, value, USAGE))
        return value

    batch_size, scale, ddim_steps, H, W, f, n_iter, prompt = seq

    batch_size = need_int(batch_size, "batch_size", 1, 16)
    ddim_steps = need_int(ddim_steps, "ddim_steps", 1, 1000)
    H = need_int(H, "H", 64, 1024)
    W = need_int(W, "W", 64, 1024)
    f = need_int(f, "f", 4, 32)
    n_iter = need_int(n_iter, "n_iter", 1, 100)

    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise SystemExit("scale 必须是数字，收到 %r\n\n%s" % (scale, USAGE))
    if scale <= 0:
        raise SystemExit("scale 必须大于 0，收到 %r\n\n%s" % (scale, USAGE))

    if not isinstance(prompt, str) or not prompt.strip():
        raise SystemExit("prompt 必须是非空字符串\n\n%s" % USAGE)

    if H % f or W % f:
        raise SystemExit(
            "H 和 W 必须能被下采样因子 f=%d 整除（当前 H=%d W=%d）\n\n%s" % (f, H, W, USAGE)
        )

    return {
        "batch_size": batch_size,
        "scale": float(scale),
        "ddim_steps": ddim_steps,
        "H": H,
        "W": W,
        "f": f,
        "n_iter": n_iter,
        "prompt": prompt.strip(),
    }


def parse_params(raw):
    """解析命令行参数。

    兼容 JSON 与 Python 字面量两种写法（start.py 历史上传的是 str(list)）。
    只接受字面量，不执行任何表达式。
    """
    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = ast.literal_eval(raw)
        except Exception as exc:
            raise SystemExit("参数解析失败: %s\n\n%s" % (exc, USAGE))
    return coerce_params(data)


def latest_image():
    """返回 output/samples 里最新的 png 路径；没有则返回 None。"""
    try:
        files = [p for p in _paths.SAMPLES_DIR.iterdir() if p.suffix.lower() == ".png"]
    except OSError:
        return None
    if not files:
        return None
    return str(max(files, key=lambda p: p.stat().st_mtime))


class SDGenerator:
    """持有模型与采样器；load() 之后可以反复 generate()。"""

    def __init__(self, config_path=None, ckpt_path=None, device=None, verbose=True):
        self.config_path = str(config_path or _paths.CONFIG_PATH)
        self.ckpt_path = str(ckpt_path or _paths.CKPT_PATH)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.verbose = verbose
        self.model = None
        self.sampler = None
        self._loaded = False
        self._index = None

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        if self._loaded:
            return self

        t0 = time.time()
        print("Loading model from %s" % self.ckpt_path)
        pl_sd = torch.load(self.ckpt_path, map_location="cpu")
        if "global_step" in pl_sd:
            print("Global Step: %s" % pl_sd["global_step"])

        config = OmegaConf.load(self.config_path)
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(pl_sd["state_dict"], strict=False)
        if self.verbose:
            if m:
                print("missing keys: %d" % len(m))
            if u:
                print("unexpected keys: %d" % len(u))

        model = model.to(self.device)
        model.eval()

        self.model = model
        self.sampler = DDIMSampler(model)
        self._loaded = True
        print("模型加载完成，用时 %.1f 秒，设备 %s" % (time.time() - t0, self.device))
        return self

    def _next_index(self):
        if self._index is None:
            try:
                self._index = len(os.listdir(str(_paths.SAMPLES_DIR))) - 1
            except OSError:
                self._index = 0
        self._index += 1
        return self._index

    def generate(self, params, seed=None):
        """按 params 生成图片，返回 (图片路径, 使用的种子)。"""
        if not self._loaded:
            self.load()

        batch_size = params["batch_size"]
        scale = params["scale"]
        ddim_steps = params["ddim_steps"]
        H, W, f = params["H"], params["W"], params["f"]
        n_iter, prompt = params["n_iter"], params["prompt"]

        if seed is None:
            seed = random.randint(1, 999999999)
        seed_everything(seed)

        os.makedirs(str(_paths.SAMPLES_DIR), exist_ok=True)

        C = 4
        n_samples = 1
        ddim_eta = 0.0
        start_code = None
        img_path = ""

        data = [batch_size * [prompt]]

        with torch.no_grad():
            with autocast("cuda"):
                with self.model.ema_scope():
                    for _ in range(n_iter):
                        for prompts in data:
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            shape = [C, H // f, W // f]
                            samples, _ = self.sampler.sample(
                                S=ddim_steps,
                                conditioning=c,
                                batch_size=n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                                eta=ddim_eta,
                                x_T=start_code,
                            )

                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples = x_samples.cpu().numpy()

                            x_sample = 255.0 * x_samples
                            img1 = np.stack(
                                [x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]],
                                axis=2,
                            )
                            img = Image.fromarray(img1.astype(np.uint8))
                            img_path = os.path.join(
                                str(_paths.SAMPLES_DIR), "%04d-%d.png" % (self._next_index(), seed)
                            )
                            img.save(img_path)
                            print("saved:", img_path)

        with open(str(_paths.PATH_LIST), "a") as fp:
            fp.write(img_path + "_new\n")

        return img_path, seed


_GENERATOR = None


def get_generator(**kwargs):
    """进程内单例：同一进程多次生成只加载一次模型。"""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = SDGenerator(**kwargs)
    return _GENERATOR