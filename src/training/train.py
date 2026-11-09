"""
Fine-tune ONE transformer on Jigsaw (6 labels, multi-label) for ONE seed.
Normally you do not call this directly -- use  python -m src.run_experiments.

Per run it saves, under  <out_dir>/runs/<model>/seed_<seed>/ :
    metrics.json        validation + test metrics (overall and per label), config, timing
    test_outputs.npz    test probabilities + true labels (for ROC curves / confusion matrices)

protocol="common" (preregistration.md v4, Sec. 5.2; the same rules as the proposed recurrent models):
light cleaning, class-weighted BCE, best epoch by validation macro-F1 (patience 3), gradient clipping 1.0;
writes <model>_p3 (threshold 0.5, + val_outputs.npz) and <model>_p4 (per-label thresholds tuned on validation).
"""

import gc
import inspect
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LABELS, MAX_LEN
from src.data import get_splits
from src.metrics import multilabel_metrics, sigmoid
from src.model_registry import MODELS, TRANSFORMER_COMMON, model_dir
from src.preprocessing import clean_text_light


class WeightedTrainer(Trainer):
    """Trainer with class-weighted BCE (common protocol, row 9)."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), pos_weight=self.pos_weight.to(logits.device))
        return (loss, outputs) if return_outputs else loss


class ToxicDataset(Dataset):
    """Tokenised comments (no padding here -- the collator pads each batch dynamically)."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i].tolist()      # 6 floats -> BCE-with-logits loss
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    m = multilabel_metrics(labels, sigmoid(logits))
    return {k: m[k] for k in ("f1_micro", "f1_macro", "precision_macro", "recall_macro",
                              "exact_match_accuracy", "roc_auc_macro")}


def build_training_args(output_dir, epochs, batch_size, lr, seed, protocol="standard"):
    """Works across old and new transformers versions (argument names changed)."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw = dict(output_dir=output_dir, num_train_epochs=epochs,
              per_device_train_batch_size=batch_size,
              per_device_eval_batch_size=batch_size * 2,
              learning_rate=lr, weight_decay=0.01, seed=seed,
              save_strategy="epoch", logging_steps=100)
    if "eval_strategy" in params:
        kw["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kw["evaluation_strategy"] = "epoch"
    common = protocol == "common"
    for name, val in [("load_best_model_at_end", True),
                      # standard: val_loss (as the paper); common protocol: validation macro-F1
                      ("metric_for_best_model", "eval_f1_macro" if common else "eval_loss"),
                      ("greater_is_better", common),
                      ("max_grad_norm", 1.0),                   # the library default, stated explicitly
                      ("save_total_limit", 1),
                      ("report_to", "none"),
                      ("warmup_ratio", 0.06),
                      ("dataloader_num_workers", 0),
                      ("fp16", torch.cuda.is_available())]:
        if name in params:
            kw[name] = val
    return TrainingArguments(**kw)


def build_trainer(model, args, train_ds, val_ds, tokenizer, patience, pos_weight=None):
    params = inspect.signature(Trainer.__init__).parameters
    kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
              data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
              compute_metrics=compute_metrics,
              callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    if "processing_class" in params:
        kw["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kw["tokenizer"] = tokenizer
    if pos_weight is not None:
        return WeightedTrainer(pos_weight=pos_weight, **kw)
    return Trainer(**kw)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def maybe_save_model(trainer, tokenizer, key, seed, val_f1_macro):
    """Keep on disk only the best run (by validation macro-F1) of each model, for the web app."""
    d = model_dir(key)
    meta_path = os.path.join(d, "best_meta.json")
    best = -1.0
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            best = json.load(f).get("val_f1_macro", -1.0)
    if val_f1_macro > best:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        trainer.save_model(d)
        tokenizer.save_pretrained(d)
        with open(meta_path, "w") as f:
            json.dump({"seed": seed, "val_f1_macro": val_f1_macro}, f)
        print(f"  saved model to {d} (best validation macro-F1 so far)")


def run_one(key, seed, data_dir="data", out_dir="results", epochs=3, batch_size=32, lr=2e-5,
            max_len=MAX_LEN, val_size=0.1, patience=2, max_train_samples=None,
            max_test_samples=None, save_model=False, overwrite=False, protocol="standard"):
    cfg = MODELS[key]
    common = protocol == "common"
    name = f"{key}_p3" if common else key
    run_dir = os.path.join(out_dir, "runs", name, f"seed_{seed}")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(metrics_path) and not overwrite:
        print(f"[skip] {cfg['label']} seed {seed}: already done ({metrics_path})")
        if common:
            from src.rnn_proposed import write_p4
            write_p4(key, seed, out_dir)
        with open(metrics_path) as f:
            return json.load(f)
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"{cfg['label']} ({cfg['checkpoint']})  |  seed {seed}")
    print("=" * 72)
    t0 = time.time()
    set_seed(seed)

    train_df, val_df, test_df, test_source = get_splits(
        data_dir, seed, val_size, max_train_samples, max_test_samples)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  [{test_source}]"
          + ("  [common protocol]" if common else ""))
    pos_weight = None
    if common:
        from src.rnn_proposed import pos_weights
        for d in (train_df, val_df, test_df):          # same light cleaning as the recurrent models
            d["text"] = d["comment_text"].map(clean_text_light)
        pos_weight = pos_weights(train_df[LABELS].values.astype(np.float32))

    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["checkpoint"], num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

    train_ds = ToxicDataset(train_df["text"], train_df[LABELS].values, tokenizer, max_len)
    val_ds = ToxicDataset(val_df["text"], val_df[LABELS].values, tokenizer, max_len)
    test_ds = ToxicDataset(test_df["text"], test_df[LABELS].values, tokenizer, max_len)

    ckpt_dir = os.path.join(run_dir, "_checkpoints")
    args = build_training_args(ckpt_dir, epochs, batch_size, lr, seed, protocol)
    trainer = build_trainer(model, args, train_ds, val_ds, tokenizer, patience, pos_weight)

    trainer.train()
    val_eval = trainer.evaluate()
    pred = trainer.predict(test_ds)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probs = sigmoid(logits)
    y_true = test_df[LABELS].values.astype(int)
    test_metrics = multilabel_metrics(y_true, probs)

    if common:
        vpred = trainer.predict(val_ds)
        vlogits = vpred.predictions[0] if isinstance(vpred.predictions, tuple) else vpred.predictions
        val_probs = sigmoid(vlogits)

    result = {
        "model": name, "model_label": TRANSFORMER_COMMON[name]["label"] if common else cfg["label"],
        "checkpoint": cfg["checkpoint"], "seed": seed,
        "test_source": test_source,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "max_len": max_len,
                   "val_size": val_size, "patience": patience},
        "val": {k.replace("eval_", ""): float(v) for k, v in val_eval.items()
                if isinstance(v, (int, float))},
        "test": test_metrics,
        "train_minutes": round((time.time() - t0) / 60, 2),
    }
    if common:
        result.update({"stage": "p3", "base_model": key})
        result["config"].update({
            "protocol": "common", "cleaning": "light", "truncation": "keep first max_len tokens",
            "padding": "right + attention mask", "weighted_loss": True,
            "pos_weight": dict(zip(LABELS, pos_weight.tolist())), "early_stopping": "val macro-F1 @0.5",
            "grad_clip": 1.0, "threshold": 0.5, "warmup_ratio": 0.06, "weight_decay": 0.01,
            "device": "cuda" if torch.cuda.is_available() else
                      ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
            "torch": torch.__version__, "transformers": transformers.__version__})
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "test_outputs.npz"),
                        probs=probs.astype(np.float32), y_true=y_true.astype(np.int8))
    if common:
        np.savez_compressed(os.path.join(run_dir, "val_outputs.npz"), probs=val_probs.astype(np.float32),
                            y_true=val_df[LABELS].values.astype(np.int8))
        from src.rnn_proposed import write_p4
        write_p4(key, seed, out_dir)

    print(f"\n{cfg['label']} seed {seed}: TEST  micro-F1={test_metrics['f1_micro']:.4f}  "
          f"macro-F1={test_metrics['f1_macro']:.4f}  AUC={test_metrics['roc_auc_macro']:.4f}  "
          f"exact-match={test_metrics['exact_match_accuracy']:.4f}  "
          f"({result['train_minutes']} min)")

    if save_model and not common:                      # common protocol never touches models/
        maybe_save_model(trainer, tokenizer, key, seed, result["val"].get("f1_macro", 0.0))

    shutil.rmtree(ckpt_dir, ignore_errors=True)
    del trainer, model, train_ds, val_ds, test_ds
    free_memory()
    return result
