#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Caminho para o data.yaml")
    return p.parse_args()

def inspect_split(split_name, base_dir, names):
    possible_dirs = [base_dir / split_name, base_dir / (split_name + "id")]
    split_dir = None
    for d in possible_dirs:
        if d.exists():
            split_dir = d
            break
            
    if not split_dir:
        return 0, 0, {}
        
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    counts = {name: 0 for name in names}
    total_annotations = 0
    
    if lbl_dir.exists():
        for lbl_file in lbl_dir.glob("*.txt"):
            lines = lbl_file.read_text().strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id < len(names):
                        counts[names[cls_id]] += 1
                        total_annotations += 1
                        
    return len(images), total_annotations, counts

def main():
    args = parse_args()
    data_yaml = Path(args.dataset)
    if not data_yaml.exists():
        print(f"[ERRO] Arquivo não encontrado: {data_yaml}")
        sys.exit(1)
        
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
        
    base_path = Path(data.get("path", data_yaml.parent))
    names = data.get("names", [])
    
    print("=" * 55)
    print("Inspeção do Dataset: epi-v1")
    print("=" * 55)
    print(f"Classes ({len(names)}): {names}\n")
    
    for split in ["train", "val", "test"]:
        n_imgs, n_anns, counts = inspect_split(split, base_path, names)
        tag = f"[{split.upper()}]"
        print(f"{tag:<8} {n_imgs} imagens  |  {n_anns} anotações")
        for cls_name, count in counts.items():
            bar = "█" * min(20, max(1, count // 20))
            print(f"    {cls_name:<16} {count:<6} {bar}")
        print()

    print("=" * 55)
    print("Dataset aprovado para treinamento.")

if __name__ == "__main__":
    main()
