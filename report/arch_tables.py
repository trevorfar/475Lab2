"""
Per-layer architecture tables (output shape, parameters, receptive field, jump) for the report.
Shapes come from a real forward pass with hooks; receptive fields use
    r_out = r_in + (k - 1) * j_in,   j_out = j_in * s.
Run from lab2_template/:  python -m report.arch_tables
"""
import json
import sys
import torch
import torch.nn as nn

from models import LeNet5, LeNetModern, AlexNet64, get_torchvision_alexnet


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def layer_table(model: nn.Module, input_size=(1, 3, 64, 64)):
    rows, hooks = [], []
    rf = {"r": 1, "j": 1}

    def hook(mod, inp, out):
        name = mod._lab_name
        k = s = None
        if isinstance(mod, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            k, s = _pair(mod.kernel_size)[0], _pair(mod.stride)[0]
            rf["r"] = rf["r"] + (k - 1) * rf["j"]
            rf["j"] = rf["j"] * s
        params = sum(p.numel() for p in mod.parameters(recurse=False))
        is_spatial = out.dim() == 4
        rows.append({
            "name": name,
            "type": mod.__class__.__name__,
            "spec": _spec(mod),
            "out": list(out.shape[1:]),
            "params": params,
            "rf": rf["r"] if is_spatial else None,
            "jump": rf["j"] if is_spatial else None,
        })

    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and not isinstance(mod, (nn.Dropout, nn.Identity)):
            mod._lab_name = name
            hooks.append(mod.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_size))
    for h in hooks:
        h.remove()
    return rows


def _spec(mod):
    if isinstance(mod, nn.Conv2d):
        return f"{mod.in_channels}->{mod.out_channels}, k={mod.kernel_size[0]}, s={mod.stride[0]}, p={mod.padding[0]}"
    if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
        return f"k={_pair(mod.kernel_size)[0]}, s={_pair(mod.stride)[0]}"
    if isinstance(mod, nn.Linear):
        return f"{mod.in_features}->{mod.out_features}"
    if isinstance(mod, nn.BatchNorm2d):
        return f"{mod.num_features} ch"
    if isinstance(mod, nn.LocalResponseNorm):
        return f"n={mod.size}, a={mod.alpha}, b={mod.beta}, k={mod.k}"
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return f"out={mod.output_size}"
    return ""


if __name__ == "__main__":
    num_classes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    models = {
        "lenet": LeNet5(num_classes),
        "lenet_modern": LeNetModern(num_classes),
        "alexnet_none": AlexNet64(num_classes, norm_type="none"),
        "alexnet_lrn": AlexNet64(num_classes, norm_type="lrn"),
        "alexnet_bn": AlexNet64(num_classes, norm_type="bn"),
        "zoo_alexnet": get_torchvision_alexnet(num_classes),
    }
    out = {}
    for key, m in models.items():
        rows = layer_table(m)
        out[key] = {"rows": rows, "total_params": sum(p.numel() for p in m.parameters())}
        print(f"\n== {key}: {out[key]['total_params']:,} params")
        for r in rows:
            print(f"  {r['name']:<14} {r['type']:<18} {r['spec']:<28} {str(r['out']):<16} {r['params']:>10,}  RF={r['rf']} J={r['jump']}")
    with open(f"report/arch_tables_{num_classes}.json", "w") as f:
        json.dump(out, f, indent=1)
