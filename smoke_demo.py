"""Exercise the training entry point on a disposable synthetic graph."""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def main():
    with TemporaryDirectory(prefix="mmkg-smoke-") as directory:
        dataset = Path(directory) / "datasets" / "DB15K"
        dataset.mkdir(parents=True)

        entities = [f"entity{index}" for index in range(16)]
        (dataset / "entity2id.txt").write_text(
            "".join(f"{entity} {index}\n" for index, entity in enumerate(entities)),
            encoding="utf-8",
        )
        (dataset / "relation2id.txt").write_text("related 0\n", encoding="utf-8")
        for split, offset in (("train", 0), ("valid", 8), ("test", 12)):
            count = 8 if split == "train" else 4
            triples = "".join(
                f"{entities[index]} related {entities[(index + 1) % len(entities)]}\n"
                for index in range(offset, offset + count)
            )
            (dataset / f"{split}.txt").write_text(triples, encoding="utf-8")

        generator = torch.Generator().manual_seed(10010)
        for filename, dimensions in (("img_features.pth", 4096), ("text_features.pth", 768)):
            features = torch.randn(len(entities), dimensions, generator=generator)
            torch.save(features, dataset / filename)

        train = Path(__file__).resolve().with_name("train.py")
        subprocess.run(
            [sys.executable, str(train), "--cuda", "-1", "--epochs", "1",
             "--eval_freq", "1", "--batch_size", "8", "--save", "0"],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
