#
# Copyright 2018 FastWave LLC
#
# Developed by Federico Raimondo <federaimondo@fastwavellc.com>
#
# NOTICE:  All information contained herein is, and remains the property of
# FastWave LLC. The intellectual and technical concepts contained
# herein are proprietary to FastWave LLC and its suppliers and may be covered
# by U.S. and Foreign Patents, patents in process, and are protected by
# trade secret or copyright law. Dissemination of this information or
# reproduction of this material is strictly forbidden unless prior written
# permission is obtained from FastWave LLC.

import numpy as np

from mne.utils import logger

import trcio

fname = '/Users/fraimondo/data/intra/EEG_12.TRC'

raw = trcio.read_raw_trc(fname, preload=True)
data1, times1 = raw[:, 1:10]
data2, times2 = trcio.read_raw_trc(fname, preload=False)[:, 1:10]
#
