import argparse
import sys

import _paths  # noqa: E402  必须在 torch / transformers 之前导入，否则离线标志不生效

from sdgen import get_generator, normalize_sampler, parse_params

EPILOG = """示例:
  python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']"
  python app/m.py "[2, 7.5, 40, 512, 512, 8, 1, 'a cat']" --sampler dpm --seed 1234
  python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']" --negative "blurry, lowres"
"""


def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion v1 命令行出图",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "params",
        help="8 个参数，JSON 或 Python 列表写法，如 \"[1, 9, 30, 512, 512, 8, 1, 'a cat']\"",
    )
    parser.add_argument("--negative", "-n", default="", help="负面提示词")
    parser.add_argument("--seed", "-s", type=int, default=None, help="随机种子，不填则随机")
    parser.add_argument(
        "--sampler", "-S", default="ddim", help="采样器：ddim（默认）/ plms / dpm"
    )
    args = parser.parse_args()

    params = parse_params(args.params)
    sampler = normalize_sampler(args.sampler)

    print("参数:")
    for key in ("batch_size", "scale", "ddim_steps", "H", "W", "f", "n_iter", "prompt"):
        print("  %-11s %s" % (key, params[key]))
    print("  %-11s %s" % ("negative", args.negative or "(空)"))
    print("  %-11s %s" % ("seed", args.seed if args.seed is not None else "(随机)"))
    print("  %-11s %s" % ("sampler", sampler))

    generator = get_generator()
    path, seed = generator.generate(
        params, seed=args.seed, negative_prompt=args.negative, sampler=sampler
    )

    print("seed: %d" % seed)
    print("output: %s" % path)


if __name__ == "__main__":
    main()