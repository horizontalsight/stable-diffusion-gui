# stable-diffusion-gui-test

本仓库根据 CompVis 官方 Stable Diffusion v1，实现了一份文生图工具。

本仓库包含上游代码的完整副本一份，自己写的界面和推理脚本一份。

我的工作：自己写了一套图形界面。

仓库地址：<https://github.com/horizontalsight/stable-diffusion-gui-test>

---

## 1. 目录结构

```
stable-diffusion-gui/
├── stable-diffusion/        官方源码与资源（CompVis 原样，勿改）
│   ├── ldm/                 潜空间扩散模型本体（UNet / VAE / CLIP 编码 / 采样器）
│   ├── configs/             训练与推理 yaml
│   ├── scripts/             官方 CLI（txt2img / img2img / inpaint …）
│   ├── data/  assets/       官方示例输入与展示图
│   ├── models/              各模型的 config.yaml；主权重也放这里（不入库）
│   ├── main.py              官方训练入口
│   ├── setup.py             pip install -e stable-diffusion 用
│   ├── environment.yaml     环境定义
│   └── LICENSE, Stable_Diffusion_v1_Model_Card.md
│
├── app/                     本项目自己的代码
│   ├── start.py             图形界面入口
│   ├── m.py                 命令行入口
│   ├── _paths.py            统一路径解析（把 stable-diffusion/ 加入 sys.path）
│   ├── ui/ui_main_start.ui  Qt Designer 界面定义
│   ├── tools/fetch_clip.py  预填 CLIP 缓存
│   ├── start.spec           PyInstaller 配置
│   ├── icon.ico
│   └── legacy/              2023 年的旧脚本（t2img.py / my_txt2img.py，仅留档）
│
├── vendor/                  第三方依赖副本（taming-transformers，不入库）
├── output/                  出图结果与运行状态（不入库）
├── README.md
├── .gitignore
└── .gitattributes
```

三个目录的分工：

| 目录 | 谁的东西 | 是否入库 | 能不能改 |
|---|---|---|---|
| `stable-diffusion/` | CompVis 官方 | 入库 | 不修改；官方更新时整块替换 |
| `app/` | 本项目自己 | 入库 | 随便改 |
| `vendor/` `output/` | 运行时数据 | 不入库 | 可随时删除重建 |
## 2. 项目安装过程

1.CompVis/stable-diffusion ( 2022-11-16 ) 得到主要代码。

2.通过environment.yaml 里面的 pip 安装，得到openai/CLIP ( 2023-02-20 ) 和 CompVis/taming-transformers ( 2022-11-07 ) 。

3.sd-v1-1.ckpt模型，来自CompVis/stable-diffusion-v-1-1-original。

4.openai/clip-vit-large-patch14，来自HuggingFace

5.本地输出output，本地打包产物start.exe

---

## 3. 模型权重

model.ckpt（.stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt，3.97 GiB）

地址：https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt

注意，v1-4是更为推荐的模型版本。

---

## 4. 运行环境

| 项目                    | 版本                      |
| --------------------- | ----------------------- |
| Python                | 3.8.5                   |
| PyTorch / torchvision | 1.11.0 + cu113 / 0.12.0 |
| pytorch-lightning     | 1.4.2                   |
| transformers          | 4.19.2                  |
| PySide2               | 5.15.2.1                |
| GPU                   | RTX 3080 10 GB          |

```powershell
conda env create -f environment.yaml
conda activate ldm
```

---

## 5. 自定义图形界面

**图形界面：**

```powershell
python app/start.py
```

模型在**第一次点击生成时**加载并常驻显存（约 8–12 秒），之后每次生成只需几秒。
生成在后台线程进行，界面不会卡死；生成期间按钮会变为「生成中…」。

**命令行**：

```powershell
python app/m.py "['1', '9', '30', '512', '512', '8', '1', 'a cat']"
```

**参数表格**：

| 参数         | 默认值   | 含义       |
|:---------- | ----- | -------- |
| batch_size | 1     | 批次大小     |
| scale      | 9     | CFG 引导系数 |
| ddim_steps | 30    | 采样步数     |
| H          | 512   | 图像高度     |
| W          | 512   | 图像宽度     |
| f          | 8     | 下采样因子    |
| n_iter     | 1     | 迭代次数     |
| prompt     | a cat | 提示词      |
| C          | 4     | 代码固定值    |
| n_samples  | 1     | 代码固定值    |
| ddim_eta   | 0.0   | 代码固定值    |
| 采样器        | DDIM  | 代码固定值    |

---

## 6. 项目修改记录

| 时间                      | 记录                                                   |
| ----------------------- | ---------------------------------------------------- |
| 2022-11-17              | 源项目仓库                                                |
| 2023-05-11 12:30        | 下载并解压源ZIP，运行                                         |
| 2023-05-11 13:07        | 执行pip，拉取 src、clip、transformers                       |
| 2023-05-11 14:52        | environment.yaml中修改 diffusers==0.12.1、pip install -e |
| 2023-05-23 20:47        | 写t2img.py                                            |
| 2023-06-05 20:41        | 写my_txt2img.py                                       |
| 2023-06-05              | 建立GitHub的仓库                                          |
| 2023-06-08 19:46–20:30  | 写了icon.ico、XXXX.spec、start.spec、dist/start.exe、ui    |
| 2023-06-13 12:11        | 写了start.py、m.py                                      |
| 2023-06-06 21:00        | 上传整体代码                                               |
| 2023-06-07 21:00 +08:00 | 写README                                              |

## 7. 许可与引用

代码与模型遵守 **CreativeML Open RAIL-M**，详见 `LICENSE`。使用时须遵守该许可中的使用限制，并向下游传递同样的条款。

```bibtex
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz
              and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

---

## 8. 命令行参数与运行要点

### 8.1 `m.py` 的参数格式

```powershell
python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']"
```

| 位置 | 参数 | 示例 | 说明 |
|:--:|---|---|---|
| 0 | batch_size | 1 | 一次生成几张（1–16） |
| 1 | scale | 9 | CFG 引导系数，必须 > 0 |
| 2 | ddim_steps | 30 | 采样步数（1–1000） |
| 3 | H | 512 | 高（64–1024，需能被 f 整除） |
| 4 | W | 512 | 宽（同上） |
| 5 | f | 8 | 下采样因子（4–32） |
| 6 | n_iter | 1 | 迭代次数（1–100） |
| 7 | prompt | 'a cat' | 提示词（非空字符串） |

固定值（代码内写死）：通道数 `C=4`、`n_samples=1`、`ddim_eta=0.0`、采样器 `DDIM`。

**数字必须写成数字**：

- 正确：`[1, 9, 30, 512, 512, 8, 1, 'a cat']`
- 错误：`['1', '9', '30', '512', '512', '8', '1', 'a cat']`
  → `TypeError: can't multiply sequence by non-int of type 'list'`

旧版用 `eval()` 解析且不做校验，报错信息不友好；现在改为 `ast.literal_eval` + 类型/范围校验，
参数不对会直接给出可读提示，不再执行任意表达式。

### 8.2 首次运行需要准备的四样东西

1. **conda 环境**（Python 3.8.5 / torch 1.11.0+cu113 / PySide2 5.15.2.1）

   ```powershell
   conda env create -f environment.yaml
   conda activate ldm
   ```

2. **主权重**：`stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt`（4.27 GB，不随仓库分发）

3. **taming-transformers 必须 editable 安装**

   它的 `taming/` 目录没有 `__init__.py`，`setup.py` 里的 `find_packages()` 扫不到任何包，
   非 editable 安装会得到一个空包（`import taming` 失败）。

   ```powershell
   pip install -e vendor\taming-transformers
   ```

4. **CLIP 文本编码器**：老版 `transformers==4.19.2` 无法处理 HuggingFace 现在的
   307 相对跳转，会报 `Can't load tokenizer`。用 `tools/fetch_clip.py` 按它的缓存规则预填：

   ```powershell
   python app\tools\fetch_clip.py
   ```

### 8.3 代理设置

如果本机走 Clash 等代理，需要显式把协议写成 `http://`：

```powershell
$env:HTTP_PROXY  = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

写成 `https://127.0.0.1:7890` 会让客户端去和代理做 TLS 握手，报
`ProxyError: Cannot connect to proxy`。

### 8.4 实测性能（RTX 4070 Ti SUPER 16 GB）

模型常驻显存后（见第 5 节），耗时分布如下：

| 阶段 | 耗时 |
|---|---|
| 首次加载 4.27 GB 权重 | 约 8–12 秒（进程内只做一次） |
| 30 步 DDIM 采样（512×512） | 约 5.5 秒 |
| 单张总计（已加载） | 约 7–9 秒 |
| 显存占用 | 约 3 GB，常驻 |

实测（同一进程连续生成两次，6 步 / 256×256）：首次 12.2 秒，第二次 0.4 秒，**约 30 倍**。