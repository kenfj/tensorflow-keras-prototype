# -*- coding: utf-8 -*-
"""make TensorBoard"""
from __future__ import absolute_import
from __future__ import unicode_literals
from datetime import datetime, timedelta, timezone
from keras.callbacks import TensorBoard, EarlyStopping
import os


def tensorboard(dir_base_name="/tmp/tf_log", run_name=None):
    if run_name is None:
        JST = timezone(timedelta(hours=+9), 'JST')
        run_name = datetime.now(JST).strftime('%Y-%m-%d_%H%M_%S')

    log_dir = dir_base_name + "/" + run_name
    os.makedirs(log_dir)

    print("tensorboard log_dir: " + log_dir)

    return TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )


def early_stopping(patience=1):
    return EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=patience,
        verbose=1,
        mode='auto'
    )
