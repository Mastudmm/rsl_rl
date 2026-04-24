from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .motion_loader import AMPLoader


def validate_amp_pipeline(
    env: Any,
    motion_files: list[str],
    amp_data: Any,
    *,
    strict: bool = True,
    max_files_to_scan: int = 8,
    expected_obs_dim: int | None = None,
) -> dict[str, Any]:
    """Run pre-training consistency checks for AMP data and env interface.

    Checks include:
    - env provides ``get_amp_obs_for_expert_trans``
    - AMP env observation tensor shape/finite values
    - motion file existence and basic JSON/frame validity
    - env AMP obs dim matches expert AMP obs dim
    """

    if not hasattr(env, "get_amp_obs_for_expert_trans"):
        raise ValueError(
            "AMP preflight failed: env is missing get_amp_obs_for_expert_trans(). "
            "Please use RslRlVecEnvWrapper with AMP support."
        )

    env_amp_obs = env.get_amp_obs_for_expert_trans()
    if not isinstance(env_amp_obs, torch.Tensor):
        raise ValueError("AMP preflight failed: env AMP obs must be a torch.Tensor.")
    if env_amp_obs.ndim != 2:
        raise ValueError(
            f"AMP preflight failed: env AMP obs must be rank-2 [num_envs, obs_dim], got {tuple(env_amp_obs.shape)}."
        )
    if not torch.isfinite(env_amp_obs).all():
        raise ValueError("AMP preflight failed: env AMP obs contains NaN/Inf values.")

    env_obs_dim = int(env_amp_obs.shape[-1])
    expert_obs_dim = int(getattr(amp_data, "observation_dim"))
    if expected_obs_dim is not None and env_obs_dim != int(expected_obs_dim):
        raise ValueError(
            "AMP preflight failed: env AMP obs dim does not match configured "
            f"amp_expected_obs_dim={expected_obs_dim}. got env={env_obs_dim}."
        )
    if env_obs_dim != expert_obs_dim:
        raise ValueError(
            "AMP preflight failed: env/expert observation dimension mismatch: "
            f"env={env_obs_dim}, expert={expert_obs_dim}."
        )

    if not motion_files:
        raise ValueError("AMP preflight failed: no motion files configured.")

    missing_files = [f for f in motion_files if not Path(f).is_file()]
    if missing_files:
        raise ValueError(
            "AMP preflight failed: some motion files do not exist:\n"
            + "\n".join(missing_files[:10])
        )

    warnings: list[str] = []
    scan_files = motion_files[: max(1, int(max_files_to_scan))]
    for path in scan_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("Frames", None)
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError(f"AMP preflight failed: '{path}' has invalid Frames (need >=2 frames).")

        frame0 = frames[0]
        if not isinstance(frame0, list):
            raise ValueError(f"AMP preflight failed: '{path}' frame must be a list.")

        frame_dim = len(frame0)
        if expert_obs_dim == AMPLoader.ROOT_Z_ONLY_OBS_END_IDX:
            # 37D training mode can consume either 37D frames directly or 43D
            # frames (with runtime filtering of root velocity terms).
            if frame_dim not in (AMPLoader.ROOT_Z_ONLY_OBS_END_IDX, AMPLoader.AMP_OBS_END_IDX):
                raise ValueError(
                    f"AMP preflight failed: '{path}' frame dim={frame_dim}, "
                    f"expected {AMPLoader.ROOT_Z_ONLY_OBS_END_IDX} or {AMPLoader.AMP_OBS_END_IDX} "
                    "for amp_obs_dim=37."
                )
        elif frame_dim != expert_obs_dim:
            raise ValueError(
                f"AMP preflight failed: '{path}' frame dim={frame_dim}, "
                f"expected {expert_obs_dim}."
            )

        sample = torch.tensor(frame0, dtype=torch.float32)
        if not torch.isfinite(sample).all():
            raise ValueError(f"AMP preflight failed: '{path}' first frame contains NaN/Inf.")

        if float(data.get("FrameDuration", 0.0)) <= 0.0:
            warnings.append(f"'{path}' FrameDuration <= 0 (will break temporal sampling).")
        if float(data.get("MotionWeight", 0.0)) < 0.0:
            warnings.append(f"'{path}' MotionWeight < 0.")

    if warnings and strict:
        raise ValueError("AMP preflight failed due to warnings in strict mode:\n" + "\n".join(warnings))

    return {
        "ok": True,
        "num_motion_files": len(motion_files),
        "scanned_motion_files": len(scan_files),
        "env_obs_dim": env_obs_dim,
        "expert_obs_dim": expert_obs_dim,
        "warnings": warnings,
    }
