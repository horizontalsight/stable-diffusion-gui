# README（agent 维护部分）

> `README.md` 保留作者手写的内容。本文件收录 agent 新增/修订的结构说明、运行要点与勘误。
> 两份文件的权威性：**目录结构、命令、参数以本文件为准**；叙事与历史以 `README.md` 为准。

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
│   ├── sdgen.py             核心：模型加载 + 生成（进程内单例）
│   ├── _paths.py            统一路径解析（把 stable-diffusion/ 加入 sys.path）
│   ├── ui/ui_main_start.ui  2023 年的旧界面定义（现由代码构建界面，仅留档）
│   ├── tools/fetch_clip.py  预填 CLIP 缓存
│   ├── start.spec           PyInstaller 配置
│   ├── icon.ico
│   └── legacy/              2023 年的旧脚本（t2img.py / my_txt2img.py，仅留档）
│
├── vendor/                  第三方依赖副本（taming-transformers，不入库）
├── output/YYYY-MM-DD/       出图结果，按日期分目录（不入库）
├── README.md                作者手写
├── README_agent.md          本文件
├── .gitignore
└── .gitattributes
```

三个目录的分工：

| 目录 | 谁的东西 | 是否入库 | 能不能改 |
|---|---|---|---|
| `stable-diffusion/` | CompVis 官方 | 入库 | 不修改；官方更新时整块替换 |
| `app/` | 本项目自己 | 入库 | 随便改 |
| `vendor/` `output/` | 运行时数据 | 不入库 | 可随时删除重建 |

---

## 2. 与 `README.md` 的差异（勘误）

`README.md` 写于 2023 年，目录重组后部分路径已变化。下表是现行对应关系：

| README.md 里的写法 | 现在的实际位置 / 用法 |
|---|---|
| `.\ldm`、`.\configs`、`.\scripts`、`.\data`、`.\assets`、`.\models` | 全部移入 `stable-diffusion/` 下 |
| `.\ui` | `app/ui/`（且已不再被程序加载，界面改为代码构建） |
| `.\dist` | 已移除（构建产物不入库） |
| `t2img.py`、`my_txt2img.py` | `app/legacy/`（仅留档，不再维护） |
| `m.py`、`start.py`、`start.spec`、`XXXX.spec`、`icon.ico` | `app/` |
| `python start.py` | `python app/start.py` |
| `python m.py "['1', '9', ...]"` | `python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']"`（**数字不能加引号**） |
| `models/ldm/stable-diffusion-v1/model.ckpt` | `stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt` |
| 采样器固定 DDIM | 现可选 DDIM / PLMS / DPM-Solver（见第 4 节） |

---

## 3. 图形界面

```powershell
python app/start.py
```

| 功能 | 说明 |
|---|---|
| 模型常驻 | 第一次点击生成时加载（约 8–12 秒），之后每次只需几秒 |
| 负面提示词 | 独立输入框，默认 `blurry, lowres, bad anatomy, watermark` |
| 种子 | 可手动填写（可复现），留空则随机；右侧「随机」按钮可一键填充 |
| 采样进度 | 进度条实时显示 `采样 x / y` |
| 采样器 | 下拉选择 DDIM / PLMS / DPM-Solver |
| 输出目录 | `output/YYYY-MM-DD/序号-种子.png`，按日期自动分目录 |
| 后台线程 | 生成在 QThread 里跑，界面不会卡死；按钮变「生成中…」并禁用 |
| 参数控件 | 数字都是 SpinBox，不会输入非法值 |
| 高 DPI | 4K + 系统缩放 100% 时界面自动放大 1.3 倍 |
| 窗口记忆 | 关闭时保存大小与位置，下次打开恢复；首次居中显示 |
| 出错提示 | 弹窗显示完整堆栈，不用去看控制台 |

---

## 4. 命令行参数

```powershell
python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']" [--negative "..."] [--seed N] [--sampler ddim|plms|dpm]
```

**位置参数（必需）**：一个包含 8 个值的列表。

| 位置 | 参数 | 示例 | 范围 / 说明 |
|:--:|---|---|---|
| 0 | batch_size | 1 | 1–16，一次生成几张（**全部保存**） |
| 1 | scale | 9 | CFG 引导系数，> 0 |
| 2 | ddim_steps | 30 | 采样步数，1–1000 |
| 3 | H | 512 | 高，64–1024，需能被 f 整除 |
| 4 | W | 512 | 宽，同上 |
| 5 | f | 8 | 下采样因子，4–32 |
| 6 | n_iter | 1 | 迭代次数，1–100 |
| 7 | prompt | 'a cat' | 提示词，非空 |

**可选参数**：

| 选项 | 默认 | 说明 |
|---|---|---|
| `--negative` / `-n` | 空 | 负面提示词 |
| `--seed` / `-s` | 随机 | 随机种子，填写后可复现 |
| `--sampler` / `-S` | `ddim` | `ddim` / `plms` / `dpm` |

示例：

```powershell
# 最简
python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']"

# 批量 2 张 + DPM-Solver + 固定种子 + 负面提示词
python app/m.py "[2, 7.5, 40, 512, 512, 8, 1, 'a cat']" --sampler dpm --seed 1234 --negative "blurry, lowres"
```

固定值：通道数 `C=4`、`ddim_eta=0.0`。

**参数必须使用正确的类型**：

- 正确：`[1, 9, 30, 512, 512, 8, 1, 'a cat']`
- 错误：`['1', '9', '30', '512', '512', '8', '1', 'a cat']`
  → `TypeError: can't multiply sequence by non-int of type 'list'`

2023 年的版本用 `eval()` 解析且不校验；现在改为 `ast.literal_eval` + 类型/范围校验，
参数不对会直接给出可读提示，且不会执行任意表达式。

> 为什么不用位置参数传负面提示词/种子：PowerShell 会把空字符串参数丢掉，导致后面的参数错位。
> 改用 `--negative` / `--seed` / `--sampler` 后不再有这个问题。

---
## 5. 首次运行需要准备的四样东西

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
   307 相对跳转，会报 `Can't load tokenizer`。用 `app/tools/fetch_clip.py` 按其缓存规则预填：

   ```powershell
   python app\tools\fetch_clip.py
   ```

   缓存就绪后，`app/_paths.py` 会自动设置 `TRANSFORMERS_OFFLINE=1`，运行不再发网络请求。

---

## 6. 代理设置

如果本机走 Clash 等代理，需要显式把协议写成 `http://`：

```powershell
$env:HTTP_PROXY  = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

写成 `https://127.0.0.1:7890` 会让客户端去和代理做 TLS 握手，报
`ProxyError: Cannot connect to proxy`。

---

## 7. 实测性能（RTX 4070 Ti SUPER 16 GB）

| 阶段 | 耗时 |
|---|---|
| 首次加载 4.27 GB 权重 | 约 8–12 秒（进程内只做一次） |
| 30 步 DDIM 采样（512×512） | 约 5.5 秒 |
| 单张总计（已加载） | 约 7–9 秒 |
| 显存占用 | 约 3 GB，常驻 |

对照：2023 年的实现每次都要起新进程重新加载，单张约 16–30 秒。

### 分辨率与显存（16 GB 卡实测）

attention 的显存占用随分辨率**平方**增长，所以分辨率是最敏感的旋钮：

| 宽×高 | 峰值显存 | 结果 |
|---|---:|---|
| 512×512 | 约 7.7 GB | ✅ 推荐 |
| 512×768（竖图） | 约 10.2 GB | ✅ 可用 |
| 768×512（横图） | 约 10.2 GB | ✅ 可用 |
| 576×576 | 约 8.9 GB | ✅ |
| 640×640 | 约 10.6 GB | ✅ 仍安全 |
| 704×704 | 约 12.9 GB | ⚠️ 已接近上限 |
| 768×768 | > 16 GB | ❌ 必然 OOM |

限制是按**面积**算的，不是单边：安全面积上限约 `704 × 704 ≈ 49.6 万像素`。
所以 512×768（39.3 万）可以，768×768（59 万）不行。界面里宽/高各可到 1024，
但只要乘积超限就会在参数区显示红色提示，并在生成前被拒绝。

### 批次也会线性放大显存

512×768 下的实测：

| 批次 | 每步耗时 | 峰值显存 | 结果 |
|---|---:|---:|---|
| 1 | 0.35 秒 | 10.2 GB | ✅ 正常 |
| 2 | **6.7 秒** | 14.8 GB | ❌ 溢出到共享内存，慢约 20 倍 |

注意：慢的不是 PLMS，而是**显存溢出**。批次 2 在 512×768 下总量超过 16 GB，
Windows 退回共享内存（走 PCIe + 系统内存），速度断崖式下跌。

规则：

1. **单张面积** ≤ `704 × 704`（这是硬限制，单张无法再拆）
2. **面积 × 批次** ≤ `62 万`；超出时**自动拆成多次串行**，而不是报错

| 组合 | 行为 |
|---|---|
| 512×768 × 批次 1 | 一次前向，10.2 GB |
| 512×512 × 批次 2 | 一次前向，9.7 GB |
| 512×768 × 批次 2 | 超出预算 → 自动拆成 2 次串行（每次 1 张） |
| 768×768 × 批次 1 | 单张面积超限，直接拒绝 |

自动拆分时的日志：

```
batch 2 exceeds VRAM budget -> split into 2 chunks of 1
saved: ...\0030-21.png
saved: ...\0031-21.png
```

实测：512×768、10 步、批次 2 拆成两次串行，总耗时 23.9 秒（此前不拆的话要 2 分 22 秒）。

**为什么拆分是合理的**：批次只是一次前向里并行算几张，拆成串行后每张的计算量不变，
显存回到单张水平，速度也回到正常。

### 批次 / 拆分 / n_iter 三者的关系（实测）

同一个种子（77）、同一组参数（512×768、6 步）下：

| 写法 | 实际执行 | 产出图哈希 |
|---|---|---|
| `batch=2, n_iter=1`（自动拆分） | 串行 2 次 batch=1 | `5e32fe47bb` / `25bd4b0342` |
| `batch=1, n_iter=2` | 串行 2 次 batch=1 | `5e32fe47bb` / `25bd4b0342` |
| `batch=2, n_iter=1`（显存够、真并行） | 一次前向算 2 张 | `2579a5a2ba` / `3cd5f20d62`（不同） |

结论：

- **自动拆分 ≡ n_iter**：两者都是"连续两次 batch=1"，随机数消耗顺序相同，**逐字节一致**
- **真并行批次 ≠ 串行多次**：一次前向生成 `(2,4,H,W)` 噪声张量，与两次 `(1,4,H,W)` 是不同的抽样
- 所以批次这个旋钮本质只控制**并行度**：显存够就并行（快一点），不够就自动退化成串行（等价 n_iter）

**想一次出多张**：批次和 n_iter 都可以；批次会自动适配显存，n_iter 则永远串行、最稳。

- SD v1 本身就是按 512 训练的，**用 512 出图质量也最稳**；更大分辨率主要带来构图空间，细节不一定更好
- GUI 里宽/高上限已限制为 704，并在参数区下方给出提示
- 发生 OOM 时弹窗会给出降级建议，不再只显示原始堆栈
- `app/_paths.py` 会设置 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` 减少碎片，每次生成前也会 `torch.cuda.empty_cache()`
- OOM 之后进程内可能残留显存，**重启程序**是最干净的恢复方式
同一进程连续生成两次实测（6 步 / 256×256）：首次 12.2 秒，第二次 0.4 秒，**约 30 倍**。

---

## 8. agent 做了哪些改动

| 时间 | 改动 | 提交 |
|---|---|---|
| 2026-11-20 | 从旧仓库复制历史建新仓库；补 `.gitignore` / `.gitattributes`（统一 LF） | `be69437` |
| 2026-11-20 | 停止跟踪 `output/`；加 `vendor/` 忽略；新增 `tools/fetch_clip.py` | `ae6e192` |
| 2026-11-21 | 目录重组：官方代码进 `stable-diffusion/`，自有代码进 `app/` | `e031eac` |
| 2026-11-21 | 抽 `sdgen.py`；模型常驻 + 后台线程生成 | `6d889e5` |
| 2026-11-21 | GUI 重写：深色主题、4K 适配、窗口记忆、负面提示词、种子、进度条、日期目录 | `0f11b39` |