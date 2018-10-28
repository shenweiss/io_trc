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
fname_in = '/media/nas/eeg-data/test/EEG_12.TRC'
fname_out = '/Users/fraimondo/data/intra/EEG_12_b.TRC'
fname_out = '/media/nas/eeg-data/test/EEG_12_b.TRC'

raw = trcio.read_raw_trc(fname_in, preload=True, include=None)
header = raw._raw_extras[0]

fid = open(fname_out, 'wb')
trcio.io._write_raw_trc_header(raw, fid)
trcio.io._write_raw_trc_data(raw, fid)
fid.close()

raw2 = trcio.read_raw_trc(fname_out, preload=True, include=None)
from numpy.testing import assert_array_almost_equal
assert_array_almost_equal(raw._data, raw2._data)
header2 = raw2._raw_extras[0]

# for k in header.keys():
#     print('========= {} ========='.format(k))
#     if 'electrodes' not in k and k != 'montages':
        
#         print('Orig')
#         print(header[k])
#         print('---------------------------------')
#         print('New')
#         print(header2[k])
#     elif k == 'electrodes':
#         el1 = header[k]
#         el2 = header2[k]
#         for t_1, t_2 in zip(el1, el2):
#             t_1_name = t_1['label+']
#             t_2_name = t_2['label+']
#             for t_k in t_1.keys():
#                 if t_1[t_k] != t_2[t_k]:
#                     print('----------------DIFF---------------')
#                     print('Orig {} ({})'.format(t_1_name, t_k))
#                     print(t_1[t_k])
#                     print('---------------------------------')
#                     print('New {} ({})'.format(t_2_name, t_k))
#                     print(t_2[t_k])
#     elif k == 'montages':
#         for k_1, v_1 in header[k].items():
#             v_2 = header2[k][k_1]
#             for t_k in v_1.keys():
#                 e1 = v_1[t_k]
#                 e2 = v_2[t_k]
#                 if isinstance(e1, np.ndarray):
#                     equal = np.array_equal(e1, e2)
#                 else:
#                     equal = e1 == e2
#                 if  equal is False:
#                     print('----------------DIFF---------------')
#                     print('Orig {} ({})'.format(k_1, t_k))
#                     print(v_1[t_k])
#                     print('---------------------------------')
#                     print('New {} ({})'.format(k_1, t_k))
#                     print(v_2[t_k])
