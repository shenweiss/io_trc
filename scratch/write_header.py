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

import trcio
import time
import numpy as np
from mne.utils import verbose, logger
import time



fname_in = '/Users/fraimondo/data/intra/EEG_12.TRC'
fname_out = '/Users/fraimondo/data/intra/EEG_12_b.TRC'

raw = trcio.read_raw_trc(fname_in, preload=True, include=None)
header = raw._raw_extras[0]

fid = open(fname_out, 'wb')
trcio.io._write_raw_trc_header(raw, fid)
trcio.io._write_raw_trc_data(raw, fid)
fid.close()

raw2 = trcio.read_raw_trc(fname_out, preload=True, include=None)
from numpy.testing import assert_array_almost_equal
assert_array_almost_equal(raw._data, raw2._data)