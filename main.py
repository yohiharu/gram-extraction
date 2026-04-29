import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

STYLE_LAYERS = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv5_1": 28,
}


def load_image(path):
    preprocess = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0)


def extract_features(model, image, layer_indices):
    features = {}
    hooks = []

    for name, idx in layer_indices.items():

        def make_hook(layer_name):
            def hook(module, input, output):
                features[layer_name] = output.detach()

            return hook

        hooks.append(model.features[idx].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(image)

    for h in hooks:
        h.remove()

    return features


def gram_matrix(feature_map):
    b, c, h, w = feature_map.shape
    F = feature_map.view(c, h * w)
    return torch.mm(F, F.t()) / (c * h * w)


def save_gram_matrices(gram_matrices, output_dir, image_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(image_path).stem

    arrays = {name: G.cpu().numpy() for name, G in gram_matrices.items()}
    out_path = output_dir / f"{source_name}_grams.npz"
    np.savez_compressed(out_path, **arrays)

    for name, arr in arrays.items():
        print(f"  {name}: shape={arr.shape}")
    print(f"Saved -> {out_path}")


def main():
    # ここからparser
    parser = argparse.ArgumentParser(
        description="Extract Gram matrices from an image using pretrained VGG19."
    )
    parser.add_argument("image", help="Path to input image file")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="gram_output",
        help="Directory to write .txt output files (default: ./gram_output)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Compute device (default: auto-select cuda > mps > cpu)",
    )
    args = parser.parse_args()

    # device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 画像のロード
    image = load_image(args.image).to(device)

    weights = models.VGG19_Weights.IMAGENET1K_V1
    model = models.vgg19(weights=weights).to(device)
    model.eval()

    print("Extracting features...")
    features = extract_features(model, image, STYLE_LAYERS)

    print("Computing Gram matrices...")
    gram_matrices = {name: gram_matrix(f) for name, f in features.items()}

    save_gram_matrices(gram_matrices, Path(args.output_dir), args.image)
    print("Done.")


if __name__ == "__main__":
    main()
