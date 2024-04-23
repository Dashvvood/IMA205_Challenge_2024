from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name",
    type=str,
    default="google/vit-base-patch16-224"
)

parser.add_argument(
    "--cache_dir",
    type=str,
    default="../cache/"
)


opts, _ = parser.parse_known_args()

snapshot_download(opts.name, cache_dir=opts.cache_dir)

print(f"[Done]: {opts.name}")

