import mne
import trcio
import numpy as np

edf_fname = ('/Users/fraimondo/data/intra/449_correct.edf')

trc_fname = './test/test.trc'


raw_edf = mne.io.read_raw_edf(edf_fname, preload=True)
raw_edf.pick_types(eeg=True, stim=False)

to_rename = {x: x.split(' ')[1].split('-')[0] for x in raw_edf.ch_names}
raw_edf.rename_channels(to_rename)

trcio.write_raw_trc(raw_edf, trc_fname)

raw_trc = trcio.read_raw_trc(trc_fname, preload=True)

# Same channels
np.testing.assert_equal(raw_edf.ch_names, raw_trc.ch_names)

# Same sample frequency
np.testing.assert_equal(raw_edf.info['sfreq'], raw_trc.info['sfreq'])

# Same data
np.testing.assert_array_almost_equal(raw_edf._data, raw_trc._data)
