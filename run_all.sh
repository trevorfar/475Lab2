#!/usr/bin/env bash
# ELEC 475 Lab 2 - full experiment queue (CPU). Each run is skipped if its results JSON already exists,
# so the script can be re-launched after an interruption.
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs results checkpoints plots

run() {
    local name=$1; shift
    if [ -f "results/${name}.json" ]; then
        echo "[skip] ${name}"
        return
    fi
    echo "[$(date '+%F %T')] start ${name}"
    $PY train.py "$@" --save-model "checkpoints/${name}.pth" --results-file "results/${name}.json" > "logs/${name}.log" 2>&1
    echo "[$(date '+%F %T')] done  ${name} (exit $?)"
}

# --- In-Lab baselines (re-run so every row uses the same code & seed) ---
run lenet_baseline  --model lenet        --dataset cifar10 --epochs 5 --seed 42 --no-augment --plot-file plots/loss_lenet.png
run lenet_modern    --model lenet_modern --dataset cifar10 --epochs 5 --seed 42 --no-augment --plot-file plots/loss_lenet_modern.png

# --- Step 2: Normalization study (10 epochs, no augmentation) ---
run alexnet_none    --model alexnet --norm none --dataset cifar10 --epochs 10 --seed 42 --no-augment --plot-file plots/loss_alexnet_none.png
run alexnet_lrn     --model alexnet --norm lrn  --dataset cifar10 --epochs 10 --seed 42 --no-augment --plot-file plots/loss_alexnet_lrn.png
run alexnet_bn      --model alexnet --norm bn   --dataset cifar10 --epochs 10 --seed 42 --no-augment --plot-file plots/loss_alexnet_bn.png

# --- Step 3: Data augmentation (15 epochs) ---
run alexnet_no_aug   --model alexnet --norm bn --dataset cifar10 --epochs 15 --seed 42 --no-augment --plot-file plots/loss_no_aug.png
run alexnet_with_aug --model alexnet --norm bn --dataset cifar10 --epochs 15 --seed 42 --plot-file plots/loss_with_aug.png

# --- Step 4: Tiny ImageNet-200 (10 epochs, augmentation on, as in the manual's command) ---
run alexnet_tiny_imagenet --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 --seed 42 --plot-file plots/loss_tiny_imagenet.png

# --- PyTorch Zoo AlexNet baseline on CIFAR-10 ---
run zoo_alexnet     --model zoo_alexnet --dataset cifar10 --epochs 10 --seed 42 --no-augment --plot-file plots/loss_zoo_alexnet.png

# --- Multi-seed evaluation (Mean +/- Std) ---
for s in 123 999; do
    run lenet_baseline_s$s --model lenet        --dataset cifar10 --epochs 5 --seed $s --no-augment
    run lenet_modern_s$s   --model lenet_modern --dataset cifar10 --epochs 5 --seed $s --no-augment
done
for s in 123 999; do
    run alexnet_bn_s$s --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed $s --no-augment
done

# --- Extended ablation exploration: learning-rate sensitivity of AlexNet-BN ---
run alexnet_bn_lr1e-4 --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed 42 --no-augment --lr 1e-4 --plot-file plots/loss_alexnet_bn_lr1e-4.png
run alexnet_bn_lr3e-3 --model alexnet --norm bn --dataset cifar10 --epochs 10 --seed 42 --no-augment --lr 3e-3 --plot-file plots/loss_alexnet_bn_lr3e-3.png

# --- Tiny ImageNet column for the remaining master-table rows (10 epochs) ---
run tiny_lenet_baseline --model lenet        --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_lenet_modern   --model lenet_modern --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_alexnet_bn     --model alexnet --norm bn   --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_alexnet_none   --model alexnet --norm none --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_zoo_alexnet    --model zoo_alexnet         --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment
run tiny_alexnet_lrn    --model alexnet --norm lrn  --dataset tiny_imagenet --epochs 10 --seed 42 --no-augment

echo "[$(date '+%F %T')] ALL DONE"
