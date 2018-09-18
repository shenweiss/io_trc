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

from mne.io.base import BaseRaw

from mne.utils import verbose, logger, warn


@verbose
def _read_raw_egi_mff(input_fname, preload=False,
                      verbose=None):
    """Read TRC file as raw object."""
    pass


class RawTRC(BaseRaw):
    """RawTRC class."""

    @verbose
    def __init__(self, input_fname, preload=False, verbose=None):
        pass

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        pass
