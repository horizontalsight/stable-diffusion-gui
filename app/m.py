import ast
import json
import os
import random
import sys

import _paths  # noqa: E402  必须在 torch / transformers 之前导入，否则离线标志不生效

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

USAGE = """用法:
  python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']"

8 个参数依次为:
  batch_size, scale, ddim_steps, H, W, f, n_iter, prompt

注意: 数字要写成数字 1，不要写成字符串 '1'，否则会报 TypeError。
"""


def parse_params(raw):
    """解析并校验命令行参数。

    兼容两种写法:
      JSON:          [1, 9, 30, 512, 512, 8, 1, "a cat"]
      Python 字面量:  [1, 9, 30, 512, 512, 8, 1, 'a cat']   (start.py 用 str(list) 传来的格式)
    只接受字面量，不接受任何表达式，避免 eval 的代码执行风险。
    """
    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = ast.literal_eval(raw)
        except Exception as exc:
            raise SystemExit("参数解析失败: %s\n\n%s" % (exc, USAGE))

    if not isinstance(data, (list, tuple)):
        raise SystemExit("参数必须是一个列表\n\n%s" % USAGE)
    if len(data) != 8:
        raise SystemExit("需要 8 个参数，实际收到 %d 个\n\n%s" % (len(data), USAGE))

    def need_int(value, name, low, high):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit("%s 必须是整数，收到 %r\n\n%s" % (name, value, USAGE))
        if not low <= value <= high:
            raise SystemExit("%s 超出范围 [%d, %d]，收到 %d\n\n%s" % (name, low, high, value, USAGE))
        return value

    batch_size, scale, ddim_steps, H, W, f, n_iter, prompt = data

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


def load_model_from_config(config, ckpt, verbose=False):
    print("Loading model from %s" % ckpt)
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print("Global Step: %s" % pl_sd["global_step"])
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    p = parse_params(sys.argv[1])
    print("参数:")
    for key in ("batch_size", "scale", "ddim_steps", "H", "W", "f", "n_iter", "prompt"):
        print("  %-11s %s" % (key, p[key]))

    os.makedirs(_paths.SAMPLES_DIR, exist_ok=True)

    seed = random.randint(1, 999999999)
    seed_everything(seed)
    print("seed: %d" % seed)

    config = OmegaConf.load(str(_paths.CONFIG_PATH))
    model = load_model_from_config(config, str(_paths.CKPT_PATH))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    start_code = None

    C = 4
    n_samples = 1
    ddim_eta = 0.0

    batch_size = p["batch_size"]
    scale = p["scale"]
    ddim_steps = p["ddim_steps"]
    H, W, f = p["H"], p["W"], p["f"]
    n_iter, prompt = p["n_iter"], p["prompt"]

    grid_count = len(os.listdir(str(_paths.SAMPLES_DIR))) - 1
    img_path = ""

    data = [batch_size * [prompt]]

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for _ in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        print("embedding shape:", c.shape)
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(
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
                        print("samples shape:", samples_ddim.shape)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().numpy()
                        print("image shape:", x_samples_ddim.shape)
                        x_sample = 255.0 * x_samples_ddim
                        img1 = np.stack(
                            [x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]], axis=2
                        )
                        img = Image.fromarray(img1.astype(np.uint8))
                        img_path = os.path.join(str(_paths.SAMPLES_DIR), "%04d-%d.png" % (grid_count, seed))
                        img.save(img_path)
                        grid_count += 1

    with open(str(_paths.PATH_LIST), "a") as fp:
        fp.write(img_path + "_new\n")


if __name__ == "__main__":
    os.makedirs(_paths.OUTPUT_DIR, exist_ok=True)
    with open(str(_paths.STATE_FILE), "w") as fp:
        fp.write("main is working")
    try:
        main()
    finally:
        print("main working is over")
        with open(str(_paths.STATE_FILE), "w") as fp:
            fp.write("main's work is done")