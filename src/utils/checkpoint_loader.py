# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
from typing import Any

import torch
from torch.serialization import MAP_LOCATION

from src.utils.logging import get_logger

logger = get_logger(os.path.basename(__file__))

_HF_PREFIX = "hf://"
_HTTPS_PREFIX = "https://"
_HTTP_PREFIX = "http://"


def _resolve_hf_path(r_path: str) -> str:
    """Resolve an hf://<repo_id>/<filename> path to a local cache path.

    Format: hf://<repo_id>/<filename_in_repo>
    Example: hf://facebook/vjepa2-vitl/vitl.pt
    """
    from huggingface_hub import hf_hub_download

    # Strip prefix and split into repo_id / filename
    remainder = r_path[len(_HF_PREFIX):]
    # repo_id is first two path components (org/model); rest is filename
    parts = remainder.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"HuggingFace path must be hf://<org>/<repo>/<filename>, got: {r_path}"
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    logger.info(f"Downloading from HuggingFace: repo={repo_id}  file={filename}")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info(f"Downloaded to: {local_path}")
    return local_path


def robust_checkpoint_loader(r_path: str, map_location: MAP_LOCATION = "cpu", max_retries: int = 3) -> Any:
    """
    Loads a checkpoint from a local path, a HuggingFace Hub path, or an HTTPS URL.

    HuggingFace paths:  hf://<org>/<repo>/<filename>
    Direct URLs:        https://... or http://...
    e.g.  https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
    """
    if r_path.startswith(_HF_PREFIX):
        r_path = _resolve_hf_path(r_path)
    elif r_path.startswith(_HTTPS_PREFIX) or r_path.startswith(_HTTP_PREFIX):
        logger.info(f"Downloading checkpoint from URL: {r_path}")
        return torch.hub.load_state_dict_from_url(r_path, map_location=map_location)

    retries = 0

    while retries < max_retries:
        try:
            return torch.load(r_path, map_location=map_location)
        except Exception as e:
            logger.warning(f"Encountered exception when loading checkpoint {e}")
            retries += 1
            if retries < max_retries:
                sleep_time_s = (2**retries) * random.uniform(1.0, 1.1)
                logger.warning(f"Sleeping {sleep_time_s}s and trying again, count {retries}/{max_retries}")
                time.sleep(sleep_time_s)
                continue
            else:
                raise e
