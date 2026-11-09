"""Audit-only bookkeeping; never imported by a candidate or by the evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def reset():
    for rel in ("data/riemanline_synthetic.json", "data/riemanline_synthetic_harder.json"):
        shutil.copy2(ROOT / "redteam/original" / rel, ROOT / rel)
    for outdir, suffix in (("outputs", ""), ("outputs/harder", "harder")):
        target = ROOT / outdir
        target.mkdir(parents=True, exist_ok=True)
        for p in target.glob("*.json"):
            p.unlink()
        controls = ROOT / "redteam/control_artifacts" / suffix
        for source in controls.glob("*.json"):
            shutil.copy2(source, target / source.name)
        reference = ROOT / "redteam/reference_seed0/artifacts" / suffix / "struct_riemanline.json"
        if reference.exists():
            shutil.copy2(reference, target / reference.name)
    # A previous score must never stand in for a failed evaluator invocation.
    for name in ("scores_train.json", "scores_test.json"):
        (ROOT / "scoring" / name).unlink(missing_ok=True)


def restore():
    originals = ROOT / "redteam/original"
    for prefix in ("data", "scoring", "method", "outputs"):
        for source in (originals / prefix).rglob("*"):
            if source.is_file():
                destination = ROOT / source.relative_to(originals)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    for directory in (ROOT / "outputs", ROOT / "outputs/harder"):
        original_dir = originals / directory.relative_to(ROOT)
        names = {p.name for p in original_dir.glob("*.json")}
        for p in directory.glob("*.json"):
            if p.name not in names:
                p.unlink()
    (ROOT / "method/audit_candidate.py").unlink(missing_ok=True)
    for p in (ROOT / "method/__pycache__").glob("audit_candidate.*.pyc"):
        p.unlink()


def save(name, train_exit, test_exit, method_exit):
    folder = ROOT / "redteam" / name
    status = {"method_exit": method_exit, "train_exit": train_exit,
              "test_exit": test_exit, "test_called_with_arguments": False,
              "test_fresh_process": True, "test_score_exists": False}
    for phase in ("method", "train", "test"):
        path = folder / f"{phase}.log"
        if path.exists():
            status[f"{phase}_log_tail"] = path.read_text()[-2500:]
    for kind in ("train", "test"):
        source = ROOT / "scoring" / f"scores_{kind}.json"
        if source.exists():
            shutil.copy2(source, folder / source.name)
            if kind == "test":
                status["test_score_exists"] = True
    for outdir, suffix in (("outputs", ""), ("outputs/harder", "harder")):
        target = folder / "artifacts" / suffix
        target.mkdir(parents=True, exist_ok=True)
        for p in (ROOT / outdir).glob("*.json"):
            shutil.copy2(p, target / p.name)
    if name in ("ground_truth_replace_shortcut", "observation_replace_shortcut"):
        target = folder / "modified_data"
        target.mkdir(exist_ok=True)
        for filename in ("riemanline_synthetic.json", "riemanline_synthetic_harder.json"):
            shutil.copy2(ROOT / "data" / filename, target / filename)
    protected = [ROOT / "scripts/evaluate_test.sh"] + list((ROOT / "eval").rglob("*.py"))
    status["evaluator_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=("reset", "restore", "save"))
    p.add_argument("name", nargs="?")
    p.add_argument("train_exit", nargs="?", type=int)
    p.add_argument("test_exit", nargs="?", type=int)
    p.add_argument("method_exit", nargs="?", type=int)
    args = p.parse_args()
    if args.operation == "reset":
        reset()
    elif args.operation == "restore":
        restore()
    else:
        save(args.name, args.train_exit, args.test_exit, args.method_exit)
