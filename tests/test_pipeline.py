"""Small functional checks; the data are synthetic, not performance evidence."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import uuid
from contextlib import contextmanager
import unittest
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csds_net.data import CrackDataset
from csds_net.model import MODEL_NAMES, build_model
from csds_net.tolerance_metrics import tolerance_counts, tolerance_metrics
from csds_net.checkpoint import load_model, forward_finite


@contextmanager
def local_temporary_directory():
    parent = (ROOT / "tmp").resolve()
    parent.mkdir(exist_ok=True)
    directory = parent / ("test_" + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != parent:
            raise RuntimeError("Test cleanup path escaped workspace")
        shutil.rmtree(directory)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metric_displacement_and_empty_foreground(self):
        truth = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
        pred = truth.clone()
        truth[:, :, 4:12, 4] = True
        pred[:, :, 4:12, 6] = True
        exact = tolerance_metrics(tolerance_counts(pred, truth, radius=0))
        nearby = tolerance_metrics(tolerance_counts(pred, truth, radius=2))
        self.assertEqual(exact["dice"], 0)
        self.assertEqual(nearby["dice"], 1)
        self.assertLess(nearby["accuracy"], 1)
        pred.zero_(); truth.zero_()
        self.assertEqual(tolerance_metrics(tolerance_counts(pred, truth))["dice"], 1)

    def test_all_ablation_shapes_and_gradient(self):
        for name in MODEL_NAMES:
            model = build_model(name).train()
            output = model(torch.rand(2, 3, 32, 32))
            self.assertEqual(output["logits"].shape, (2, 1, 32, 32))
            (output["logits"].mean() + output["edge_logits"].mean()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_nonfinite_output_retry(self):
        class RetryModel(torch.nn.Module):
            calls = 0
            def forward(self, x):
                self.calls += 1
                return {"logits":x[:, :1] * (float("inf") if self.calls == 1 else 1)}
        output, retries = forward_finite(RetryModel(), torch.ones(1, 3, 4, 4))
        self.assertEqual(retries, 1)
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_train_resume_evaluate_and_overwrite_guard(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with local_temporary_directory() as temporary:
            base = Path(temporary)
            self.assertTrue(base.resolve().is_relative_to((ROOT / "tmp").resolve()))
            dataset = base / "Synthetic"
            for folder in ("images", "masks"):
                (dataset / folder).mkdir(parents=True)
            for i in range(10):
                image = np.full((32, 32, 3), 110+i, dtype=np.uint8)
                mask = np.zeros((32,32), dtype=np.uint8)
                image[5:25, 10+i%3] = 30
                mask[5:25,10+i%3] = 255
                Image.fromarray(image).save(dataset / "images" / f"{i:02d}.png")
                Image.fromarray(mask).save(dataset / "masks" / f"{i:02d}.png")
            sets = [CrackDataset(dataset, split, image_size=32, split_seed=42,
                    train_ratio=0.8, val_ratio=0.2) for split in ("train", "val")]
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"split_seed":42,"train":sets[0].sample_ids,
                                            "val":sets[1].sample_ids,"test":[]}), encoding="utf-8")
            self.assertFalse(set(sets[0].sample_ids) & set(sets[1].sample_ids))
            # Measure a real tiny-input profile; never fabricate complexity values.
            from profile_model import measure_complexity
            profile = base / "profile.json"
            measure_complexity("csds_ablation_m0", 32, torch.device("cpu"), profile, repeats=1, warmup=0)
            out = base / "runs"
            command = [sys.executable, "train.py", "--data-root", str(dataset),
                "--model", "csds_ablation_m0", "--reference-manifest", str(manifest),
                "--output-root", str(out), "--complexity-json", str(profile),
                "--epochs", "2", "--image-size", "32", "--batch-size", "2", "--device", "cpu"]
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            def run(arguments, check=True):
                result = subprocess.run(arguments, cwd=ROOT, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if check and result.returncode:
                    self.fail(result.stdout)
                return result
            run(command)
            directory = out / "Synthetic/csds_ablation_m0/seed_42"
            checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
            saved = {key:(directory/key).read_bytes() for key in ("history.csv", "tolerance2_history.csv", "last.pt", "best.pt")}
            with (directory/"history.csv").open(newline="", encoding="utf-8") as handle:
                before = [float(row["val_dice"]) for row in csv.DictReader(handle)]
            run(command + ["--resume"])
            self.assertEqual(saved["history.csv"], (directory/"history.csv").read_bytes())
            self.assertNotEqual(run(command,check=False).returncode,0)
            self.assertIn("rng_state",checkpoint)
            # Interrupt a separate copy immediately after the first real epoch,
            # then verify resume matches an uninterrupted two-epoch run.
            resumed_command = list(command)
            resumed_command[resumed_command.index("--output-root")+1] = str(base/"resumed")
            process = subprocess.Popen([sys.executable,"-u",*resumed_command[1:]], cwd=ROOT, env=env,
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            seen = False
            try:
                for line in process.stdout:
                    if line.startswith("epoch=001"):
                        seen = True
                        process.terminate()
                        break
                process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                process.stdout.close()
            self.assertTrue(seen)
            resume_directory = base/"resumed/Synthetic/csds_ablation_m0/seed_42"
            run(resumed_command+["--resume"])
            resumed = torch.load(resume_directory/"last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["epoch"],2)
            self.assertTrue(all(torch.equal(value,resumed["model"][key]) for key,value in checkpoint["model"].items()))
            model, config, epoch = load_model(directory/"best.pt")
            self.assertTrue(torch.isfinite(model(torch.rand(1,3,32,32))["logits"]).all())
            run([sys.executable,"evaluate.py","--checkpoint",str(directory/"best.pt"),
                 "--data-root",str(dataset),"--manifest",str(manifest),"--device","cpu",
                 "--output",str(base/"evaluation.json")])
            result = json.loads((base/"evaluation.json").read_text())
            self.assertAlmostEqual(result["strict"]["dice"],max(before),places=6)
            self.assertEqual(result["validation_samples"],2)


if __name__ == "__main__":
    unittest.main()
