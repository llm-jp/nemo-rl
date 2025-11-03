import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_one_dataset(
    name: str,
    examples: List[Dict[str, Any]],
    data_info: Dict[str, Any],
    max_dev_samples_global: int,
    max_dev_ratio_global: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Behavior mirrors the reference logic:
      - dev: take from the head with size = min(cfg.max_dev_samples, int(len * cfg.max_dev_ratio))
      - train: take right after dev for max_train_samples items (-1 means all remaining)
      - upsampling_factor: multiply both train and dev by this factor
      - max_train_samples=0: skip this dataset entirely
    ref: https://github.com/llm-jp/llm-jp-nemo-aligner/blob/9ca83865487a28fff5692a1b5a9fef9c2b9a7287/sft_dataset.py#L168-L256
    """
    if data_info.get("max_train_samples", -1) == 0:
        logging.info(f"[{name}] skip: max_train_samples=0")
        return [], [], {
            "train": 0,
            "dev": 0,
            "original": len(examples),
            "upsampling_factor": data_info.get("upsampling_factor", 1),
        }

    if data_info.get("split_dev", False):
        max_dev = min(
            max_dev_samples_global,
            int(len(examples) * max_dev_ratio_global),
        )
    else:
        max_dev = 0

    mts = data_info.get("max_train_samples", -1)
    if mts == -1:
        mts = max(0, len(examples) - max_dev)

    if mts > len(examples):
        logging.warning(
            f"[{name}] only {len(examples)} examples, but max_train_samples={mts}. Use all available."
        )

    # Slice indices are clipped to valid bounds
    train_slice = examples[max_dev : max_dev + mts]
    dev_slice = examples[:max_dev]

    up = int(data_info.get("upsampling_factor", 1))
    if up < 1:
        up = 1

    train_out = train_slice * up
    dev_out = dev_slice * up

    stats = {
        "train": len(train_out),
        "dev": len(dev_out),
        "original": len(examples),
        "upsampling_factor": up,
    }
    return train_out, dev_out, stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge JSONL datasets per YAML config (split_dev/max_train_samples/upsampling_factor)."
    )
    parser.add_argument("config", type=Path, help="YAML config file path")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base dir containing tuning/train (override YAML). Expected files: <data-dir>/<name>.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./merged_dataset"),
        help="Output directory for merged jsonl files",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default="merged.train.jsonl",
        help="Output filename for merged train jsonl",
    )
    parser.add_argument(
        "--dev-out",
        type=str,
        default="merged.dev.jsonl",
        help="Output filename for merged dev jsonl (written only if non-empty)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_yaml(args.config)
    yaml_data_dir = Path(cfg.get("data_dir", "."))

    data_dir: Path
    if args.data_dir is not None:
        data_dir = args.data_dir
    elif yaml_data_dir is not None:
        data_dir = yaml_data_dir
    else:
        raise ValueError("Input data directory not specified in CLI or YAML.")

    logging.info(f"Input dataset dir: {data_dir}")

    datasets_cfg: Dict[str, Dict[str, Any]] = cfg.get("datasets", {})
    if not datasets_cfg:
        raise ValueError("Could not find 'datasets' config in YAML.")

    max_dev_samples_global = int(cfg.get("max_dev_samples", 0))
    max_dev_ratio_global = float(cfg.get("max_dev_ratio", 0.0))

    total_train: List[Dict[str, Any]] = []
    total_dev: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, int]] = {}

    for name, data_info in datasets_cfg.items():
        jsonl_path = data_dir / f"{name}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"{jsonl_path} does not exist.")

        examples = read_jsonl(jsonl_path)
        train_out, dev_out, stats = process_one_dataset(
            name, examples, data_info, max_dev_samples_global, max_dev_ratio_global
        )

        total_train.extend(train_out)
        total_dev.extend(dev_out)
        summary[name] = stats

        logging.info(
            f"[{name}] {stats['original']} -> train {stats['train']} / dev {stats['dev']} (upsampling {stats['upsampling_factor']})"
        )

    logging.info("-" * 30)
    logging.info(
        f"TOTAL: original {sum(s['original'] for s in summary.values())} "
        f"-> train {len(total_train)} / dev {len(total_dev)}"
    )
    logging.info("-" * 30)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out_path = args.out_dir / args.train_out
    write_jsonl(train_out_path, total_train)
    logging.info(f"Wrote train: {train_out_path} ({len(total_train)} rows)")

    if len(total_dev) > 0:
        dev_out_path = args.out_dir / args.dev_out
        write_jsonl(dev_out_path, total_dev)
        logging.info(f"Wrote dev:   {dev_out_path} ({len(total_dev)} rows)")
    else:
        logging.info("No dev samples -> dev file not written")


if __name__ == "__main__":
    main()
