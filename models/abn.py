import json
import os
from typing import Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

from .resnet_abn import build_from_arch


class ABNForImageClassification(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = nn.CrossEntropyLoss()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        arch: Optional[str] = None,
        num_labels: Optional[int] = None,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
        revision: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        local_files_only: bool = False,
        filename: Optional[str] = None,
    ) -> "ABNForImageClassification":
        """事前学習済みチェックポイントからモデルを構築して重みをロードします。

        - `pretrained_model_name_or_path`: 以下のいずれか
          - ローカルのディレクトリ（`model.safetensors` を含む）
          - ローカルのファイルパス（`.safetensors` もしくは torch.save フォーマット）
          - Hugging Face Hub のリポジトリ ID（例: "org/repo"）
        - `arch`: 未指定なら "resnet152"。
        - `num_labels`: 未指定時は `fc.weight` の 0 次元から推定（失敗時は 10）。
        - `map_location`: "cpu"/"cuda" など。
        - `strict`: `load_state_dict` の `strict`。
        - `revision`, `token`, `local_files_only`, `filename`: Hugging Face Hub からの取得時に使用。
        """

        # 1) 入力の解決
        #    - ローカル: ディレクトリ/ファイル
        #    - それ以外は Hugging Face Hub とみなして取得を試みる
        target_path = pretrained_model_name_or_path
        repo_id: Optional[str] = None
        if os.path.isdir(target_path):
            cand = os.path.join(target_path, "model.safetensors")
            if os.path.exists(cand):
                target_path = cand
        elif os.path.exists(target_path):
            pass
        else:
            repo_id = pretrained_model_name_or_path
            try:
                from huggingface_hub import hf_hub_download, snapshot_download
            except Exception as e:
                raise RuntimeError(
                    "Hugging Face Hub からのダウンロードには huggingface_hub が必要です。\n"
                    "pip install huggingface_hub を実行してください。"
                ) from e

            # config.json があれば arch と num_labels を補完
            try:
                cfg_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                    revision=revision,
                    token=token,
                    local_files_only=local_files_only,
                )
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if arch is None:
                    arch = (
                        cfg.get("arch")
                        or cfg.get("model_arch")
                        or cfg.get("architecture")
                    )
                if num_labels is None:
                    num_labels = cfg.get("num_labels")
            except Exception:
                pass

            # モデルファイル候補を優先順位で解決
            candidates = [
                filename,
                "model.safetensors",
                "pytorch_model.bin",
                "model.bin",
                "model.pt",
                "model.ckpt",
            ]
            found_path = None
            for fname in candidates:
                if not fname:
                    continue
                try:
                    found_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=fname,
                        revision=revision,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    break
                except Exception:
                    continue

            if not found_path:
                # スナップショットで全体を落として既知名を探索
                repo_dir = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=token,
                    local_files_only=local_files_only,
                )
                for fname in candidates:
                    if not fname:
                        continue
                    cand = os.path.join(repo_dir, fname)
                    if os.path.exists(cand):
                        found_path = cand
                        break

            if not found_path:
                raise FileNotFoundError(
                    "Hugging Face Hub 上で既知のモデルファイルが見つかりませんでした。"
                )
            target_path = found_path

        # 2) state dict を読み込み
        if target_path.endswith(".safetensors") and os.path.exists(target_path):
            state = safe_load_file(target_path, device=str(map_location))
        else:
            ckpt = torch.load(target_path, map_location=map_location)
            if isinstance(ckpt, dict) and "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt

        # 3) 既知の接頭辞を簡易除去
        prefixes = (
            "model.base_model.",
            "module.base_model.",
            "base_model.",
            "model.",
            "module.",
        )
        normalized = {}
        for k, v in state.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    break
            normalized[nk] = v

        # 4) クラス数の決定（fc.weight の行数）
        inferred_num = None
        w = normalized.get("fc.weight")
        if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
            inferred_num = int(w.shape[0])
        final_num_labels = (
            num_labels if num_labels is not None else (inferred_num or 10)
        )

        # 5) モデル構築と重みロード
        final_arch = arch or "resnet152"
        base_model = build_from_arch(final_arch, num_classes=final_num_labels)
        base_model.load_state_dict(normalized, strict=strict)

        # 6) ラッパー生成と簡易 config
        model = cls(base_model)

        # 7) 読み込みデバイスへ移動
        model.to(map_location)

        return model

    def forward(self, pixel_values=None, labels=None, **kwargs):
        att_logits, per_logits, _ = self.base_model(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(att_logits, labels) + self.loss_fn(per_logits, labels)
        return {"loss": loss, "logits": per_logits}
