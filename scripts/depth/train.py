# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import time
import logging

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":
    # build logger
    # logging and checkpoint saving
    log_path = os.path.join(opts.log_dir, opts.model_zoo)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, "train.log"))
    stream_handler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(opts)

    trainer = Trainer(opts, logger)

    tic = time.time()
    trainer.train()
    logger.info("Training Finished! Total training time is %dh %dm" %
                (int((time.time() - tic) / 3600), int((time.time() - tic) % 3600 / 60)))
