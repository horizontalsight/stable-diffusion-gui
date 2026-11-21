import os
import random

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

    # 示例
    a = random.randint(1, 999999999)
    print(a)

    # 设置随机数种子
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
    # sampler = DDIMSampler(model)

    #sampler = DPMSolverSampler(model)
    #sampler = PLMSSampler(model)
    sampler = DDIMSampler(model)

    # 批次大小
    batch_size = 1
    # 起始编码 (没用)
    start_code = None
    # CFG 缩放
    scale = 9
    # 采样步数
    ddim_steps = 50
    # 通道数
    C = 4
    # 高
    H = 512
    # 宽
    W = 512
    # 下采样因子
    f = 8
    # 采样数量
    n_samples = 1
    # 确定性因子
    ddim_eta = 0.0
    # 迭代次数
    n_iter = 1


    outpath = "output/ex"
    grid_count = len(os.listdir(outpath)) - 1

    # prompt = "1 girl, 18 years old, on the street, full body," \
    #          "dressing white shirt, wearing black short skirt, good body proportions, " \
    #          "back to the viewer, two arms, two legs, perfect proportion"

    prompt = "class room, desks, chairs"

        # 批次 x 提示词 = [提示词,提示词 ...]
    data = [batch_size * [prompt]]

    #使用PyTorch训练深度学习模型时，以半精度（half-precision）或混合精度（mixed-precision）等方式进行模型推理。

    #precision_scope=autocast是将一个函数autocast()赋值给变量precision_scope，是一个函数别名，可以替代使用autocast()函数。
    #实际上，Python中的函数也可以像变量一样被赋值、传递、接收和返回。所以，可以使用别名的方式来简化函数调用过程。

    #precision_scope=autocast的作用是将autocast()函数赋值给precision_scope变量，这样可以在后续的代码中，
    # 直接使用precision_scope(context)的方式调用autocast()函数的上下文管理器，将指定作用域内的所有运算转换为指定的精度计算。
    # 其中，context参数可以指定为 "cpu"、"cuda" 或是特定的按需指定的硬件设备（如"cuda:0"）。
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                # model.ema_scope()也是一个自定义的上下文管理器，它是为了减少深度学习模型的过拟合风险而设计的。
                # 在进行模型训练和推理过程中，过拟合是一个普遍存在的问题，即模型在训练数据上表现很好，但对于新数据的泛化性能很差。
                # EMA是一种常用的降低过拟合的技术，其原理是通过指数移动平均来平滑模型参数，使得模型更加稳定和泛化。
                #
                # 在代码中，with model.ema_scope()的作用是使用model自身定义的ema_scope()上下文管理器来对模型进行评估，
                # 其中将所有在该上下文管理器中执行的语句都转化为EMA策略下的计算。与使用autocast()类似，
                # 这里的with语句也是用于限定一个代码块，使得其中的计算能够被自动地转换为EMA策略下的计算。
                #
                # 具体来说，在ema_scope()上下文管理器下，对于所有参数更新，都会通过指数移动平均的方式进行平滑处理，
                # 从而可以有效地减小模型参数在训练集上的过拟合风险，提高模型的泛化性能。在推理过程中，使用EMA平滑后的模型，可以更好地适应新数据的分布，以获得更好的结果。
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        # cfg scale
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        # 元组 转 列表
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        # c 就是embedding
                        c = model.get_learned_conditioning(prompts)
                        print("embedding shape:", c.shape)

                        # 采样数据 shape (4, 512/8, 512/8) => (4, 64, 64)
                        shape = [C, H // f, W // f]
                        # 采样
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
                        # VAE 解码器 输出最终图像
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        # 把图像 值域 缩放到 0-1之间
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().numpy()

                        print("image shape:", x_samples_ddim.shape)
                        # 图像 值域从0-1 缩放到 0-255
                        x_sample = 255. * x_samples_ddim
                        # 保存成PNG文件
                        img1 = np.stack([x_sample[0][0,:,:], x_sample[0][1,:,:], x_sample[0][2,:,:]], axis=2)
                        img = Image.fromarray(img1.astype(np.uint8))
                        #img.save(f"0.png")
                        img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1
     
if __name__ == "__main__":
    main()