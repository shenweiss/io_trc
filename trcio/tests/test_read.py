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
from mne.io.tests.test_raw import _test_raw_reader

from trcio import read_raw_trc


def test_read_trc():
    """Test importing Micromed TRC files"""
    trc_fname = '/Users/fraimondo/data/intra/EEG_12.TRC'
    raw = read_raw_trc(trc_fname, include=None)
    assert ('RawTRC' in repr(raw))
    raw = _test_raw_reader(read_raw_trc, input_fname=trc_fname)
