# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):

    logger.info(f"Running pre-training of app: {app}")
    logger.info(f"Importing app.{app}.train")
    module = importlib.import_module(f"app.{app}.train")
    logger.info(f"Imported app.{app}.train")
    return module.main(args=args, resume_preempt=resume_preempt)
