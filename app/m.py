import sys

import _paths  # noqa: E402  必须在 torch / transformers 之前导入，否则离线标志不生效

from sdgen import get_generator, parse_params


def main():
    params = parse_params(sys.argv[1])
    print("参数:")
    for key in ("batch_size", "scale", "ddim_steps", "H", "W", "f", "n_iter", "prompt"):
        print("  %-11s %s" % (key, params[key]))

    gen = get_generator()
    path, seed = gen.generate(params)

    print("seed: %d" % seed)
    print("output: %s" % path)


if __name__ == "__main__":
    main()