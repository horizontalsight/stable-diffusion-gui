import os
import pickle
import random
import sys
from start import array

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


# 从配置文件创建模型
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
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
    array = eval(sys.argv[1])
    # Put array values into d array
    # array ['1', '9', '30', '512', '512', '8', '1', 'a cat']
    d = [None] * 11
    d[0] = array[0]
    d[1] = array[1]
    d[2] = array[2]
    d[3] = 4
    d[4] = array[3]
    d[5] = array[4]
    d[6] = array[5]
    d[7] = 1
    d[8] = 0.0
    d[9] = array[6]
    d[10] = array[7]

    # 相关提示词
    # TODO:  [0]批次大小   [1]CFG缩放    [2]采样步数     [3]通道数    [4]高    [5]宽
    # TODO:  [6]下采样因子    [7]采样数量     [8]确定性因子    [9]迭代次数
    # TODO:  [10]提示词
    # d = [1, 9, 40, 4, 576, 576,
    #      8, 1, 0.0, 1,
    #      "a cat"]
    print(array)
    print(d)

    print("ok")

    # 设置9位随机数种子
    a = random.randint(1, 999999999)
    #a = 476319773
    seed_everything(a)

    # 加载配置文件
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    # 从加载的配置文件创建模型
    model = load_model_from_config(config, "models\ldm\stable-diffusion-v1\model.ckpt")

    # 判断GPU或者CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 将模型送到CUDA设备 或者 CPU
    model = model.to(device)

    # 创建ddim 采样器
    sampler = DDIMSampler(model)
    # 起始编码
    start_code = None


    # 装入变量
    batch_size, scale, ddim_steps, C, H, W, f, n_samples, ddim_eta, n_iter, prompt = d[0], *d[1:]

    outpath = "output/samples"
    grid_count = len(os.listdir(outpath)) - 1

    img_path = ""

    # 批次 x 提示词 = [提示词,提示词 ...]
    data = [batch_size * [prompt]]

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        # cfg scale
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        # 元组转列表
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        # c: embedding
                        c = model.get_learned_conditioning(prompts)
                        print("embedding shape:", c.shape)
                        # 采样
                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)
                        print("samples shape:", samples_ddim.shape)
                        # VAE解码器 输出最终图像
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        # 把图像值域缩放到[0,1]之间
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().numpy()

                        print("image shape:", x_samples_ddim.shape)
                        # 图像值域从[0,1]缩放到[0,255]
                        x_sample = 255. * x_samples_ddim
                        # 保存
                        img1 = np.stack([x_sample[0][0, :, :], x_sample[0][1, :, :], x_sample[0][2, :, :]], axis=2)
                        img = Image.fromarray(img1.astype(np.uint8))
                        img_path = os.path.join(outpath, f'{grid_count:04}-{str(a)}.png')
                        img.save(img_path)
                        grid_count += 1

    path = img_path
    with open('output/path_list.txt', 'a') as f:
        # 写入path
        f.write(path + "_new"+'\n')

if __name__ == "__main__":

    #阻止新的调用
    with open('output/main_is_working.txt', 'w') as f:
        f.write("main is working")

    main()

    print("main working is over")
    with open('output/main_is_working.txt', 'w') as f:
        f.write("main's work is done")
