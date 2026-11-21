"""预填 CLIP 文本编码器缓存，绕过老版 transformers 的下载缺陷。

背景:
    transformers==4.19.2 在 utils/hub.py 的 get_from_cache() 中，遇到 3xx 重定向时
    直接把响应头里的 Location 当作完整 URL 使用。而 HuggingFace 现在返回的是
    307 + 相对路径 (/api/resolve-cache/...)，于是下载失败，最终报
    "OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'"。

做法:
    跳过它的下载逻辑，直接按它自己的缓存命名规则 url_to_filename(url, etag)
    把文件放到 TRANSFORMERS_CACHE 下，之后 from_pretrained 即可命中缓存。

注意:
    计算文件名时必须使用【带引号】的 etag（与 transformers 内部一致），
    去掉引号会得到另一个文件名，缓存对不上。

用法:
    python tools/fetch_clip.py
"""import os
import requests
from transformers.utils.hub import url_to_filename, TRANSFORMERS_CACHE

REPO  = "openai/clip-vit-large-patch14"
FILES = ["config.json", "vocab.json", "merges.txt", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "pytorch_model.bin"]
BASE  = f"https://huggingface.co/{REPO}/resolve/main/"

os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
print("cache:", TRANSFORMERS_CACHE)
s = requests.Session()

for fn in FILES:
    url = BASE + fn
    h = s.head(url, allow_redirects=False, timeout=30)
    etag = h.headers.get("X-Linked-Etag") or h.headers.get("ETag")
    if not etag:
        print(f"[skip] no etag: {fn}"); continue

    new_path = os.path.join(TRANSFORMERS_CACHE, url_to_filename(url, etag))          # 带引号
    old_path = os.path.join(TRANSFORMERS_CACHE, url_to_filename(url, etag.strip('"')))  # 不带引号（我上次算的）

    if os.path.exists(new_path) and os.path.getsize(new_path) > 0:
        print(f"[cached] {fn}"); continue

    # 已有旧名字的文件 → 直接改名，不重新下载
    if os.path.exists(old_path) and os.path.getsize(old_path) > 0:
        os.replace(old_path, new_path)
        print(f"[renamed] {fn}  {os.path.getsize(new_path):,} B")
        continue

    print(f"[download] {fn} ...", flush=True)
    with s.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = new_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        os.replace(tmp, new_path)
    print(f"[ok] {fn}  {os.path.getsize(new_path):,} B")

print("done")