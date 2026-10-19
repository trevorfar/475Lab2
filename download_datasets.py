"""
ELEC 475: Lab 2 - Pre-Lab Task 2: Dataset Download
Downloads and extracts CIFAR-10 (./data/cifar-10-batches-py/) and
Tiny ImageNet-200 (./data/tiny-imagenet-200/).

Usage:
    python download_datasets.py                     # both datasets
    python download_datasets.py --dataset cifar10   # CIFAR-10 only
    python download_datasets.py --dataset tiny_imagenet
"""
import argparse
import os
import sys
import urllib.request
import zipfile

from torchvision.datasets import CIFAR10

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total_size / 1e6:.1f} MB ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {done / 1e6:7.1f} MB")
    sys.stdout.flush()


def download_cifar10(data_dir: str):
    print("\n[CIFAR-10]")
    target = os.path.join(data_dir, "cifar-10-batches-py")
    # torchvision verifies MD5 and skips the download if already present
    train_set = CIFAR10(root=data_dir, train=True, download=True)
    test_set = CIFAR10(root=data_dir, train=False, download=True)
    print(f"  Location : {target}")
    print(f"  Train    : {len(train_set):,} images")
    print(f"  Test     : {len(test_set):,} images")


def download_tiny_imagenet(data_dir: str, keep_zip: bool = False):
    print("\n[Tiny ImageNet-200]")
    target = os.path.join(data_dir, "tiny-imagenet-200")
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if os.path.isdir(os.path.join(target, "train")):
        print(f"  Already extracted at {target}, skipping.")
    else:
        if not os.path.exists(zip_path):
            print(f"  Downloading {TINY_IMAGENET_URL} ...")
            tmp_path = zip_path + ".part"
            urllib.request.urlretrieve(TINY_IMAGENET_URL, tmp_path, _progress)
            os.replace(tmp_path, zip_path)
            print("\n  Download complete.")

        print(f"  Extracting {zip_path} ...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
        except zipfile.BadZipFile:
            os.remove(zip_path)
            raise RuntimeError("Corrupt zip removed; re-run the script to download again.")
        print("  Extraction complete.")

        if not keep_zip:
            os.remove(zip_path)

    train_dir = os.path.join(target, "train")
    n_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"  Location : {target}")
    print(f"  Classes  : {n_classes}")


def main():
    parser = argparse.ArgumentParser(description="Download Lab 2 datasets (Pre-Lab Task 2).")
    parser.add_argument("--dataset", choices=["cifar10", "tiny_imagenet", "both"], default="both")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the Tiny ImageNet zip after extracting.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset in ("cifar10", "both"):
        download_cifar10(args.data_dir)
    if args.dataset in ("tiny_imagenet", "both"):
        download_tiny_imagenet(args.data_dir, keep_zip=args.keep_zip)

    print("\nDone.")


if __name__ == "__main__":
    main()
