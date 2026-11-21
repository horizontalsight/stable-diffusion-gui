import sys

import _paths  # noqa: E402  必须在 torch / transformers 之前导入，否则离线标志不生效

from sdgen import get_generator, parse_params

USAGE = """用法:
  python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']" ["负面提示词"] [种子]

示例:
  python app/m.py "[1, 9, 30, 512, 512, 8, 1, 'a cat']"
  python app/m.py "[1, 7.5, 30, 512, 512, 8, 1, 'a cat']" "blurry, lowres" 12345
"""


def main():
    if len(sys.argv) < 2:
        raise SystemExit(USAGE)

    params = parse_params(sys.argv[1])
    negative_prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    seed = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None

    print("参数:")
    for key in ("batch_size", "scale", "ddim_steps", "H", "W", "f", "n_iter", "prompt"):
        print("  %-11s %s" % (key, params[key]))
    print("  %-11s %s" % ("negative", negative_prompt or "(空)"))
    print("  %-11s %s" % ("seed", seed if seed is not None else "(随机)"))

    generator = get_generator()
    path, seed = generator.generate(params, seed=seed, negative_prompt=negative_prompt)

    print("seed: %d" % seed)
    print("output: %s" % path)


if __name__ == "__main__":
    main()