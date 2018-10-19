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
import time
import datetime

from mne.io.base import BaseRaw
from mne.io.meas_info import _empty_info
from mne.io.utils import _create_chs, _mult_cal_one
from mne.io.constants import FIFF

from mne.utils import verbose, logger

_micromed_units = {
    -1: 1e-9,   # nVolt
    0: 1e-6,    # uVolt
    1: 1e-3,    # mVolt
    2: 1,       # Volt
    100: 1,     # %
    101: 1,     # bpm
    102: 1      # Adim.

}


@verbose
def _read_raw_trc_header(input_fname, verbose=None):
    header = dict()
    with open(input_fname) as fid:
        fid.seek(175, 0)
        header_type = np.fromfile(fid, 'B', 1)[0]
        if header_type in [0, 1]:
            logger.info('This is a System 1 header')
        elif header_type == 2:
            logger.info('This is a System 2 header')
        elif header_type in [3, 4]:
            logger.info('This is a System98 header')
        else:
            logger.info('Unknown header type')

        if header_type != 4:
            logger.warning(
                'This reader is intended for micromed System98 files (4)')

        fid.seek(0, 0)
        header['header_type'] = header_type

        # Version
        version = np.fromfile(fid, 'S1', 32).astype('U1')
        version = ''.join(version[:-2])  # Discard last 2 chars 0x00 0x1A
        header['version'] = version

        logger.info('Reading {}'.format(version))
        logger.info('Reading Recording data')

        # Laboratory
        laboratory = np.fromfile(fid, 'S1', 32).astype('U1')
        laboratory = ''.join(laboratory[:-1])  # Discard last char 0x00
        logger.info('\tLaboratory: {}'.format(laboratory))
        header['laboratory'] = laboratory

        # Patient Data
        logger.info('\tPatient data')

        surname = np.fromfile(fid, 'S1', 22).astype('U1')
        surname = ''.join(surname)
        logger.info('\t\tSurname: {}'.format(surname))
        header['surname'] = surname

        name = np.fromfile(fid, 'S1', 20).astype('U1')
        name = ''.join(name)
        logger.info('\t\tName: {}'.format(name))
        header['name'] = name

        birth_month = np.fromfile(fid, 'B', 1)[0]
        birth_day = np.fromfile(fid, 'B', 1)[0]
        birth_year = np.fromfile(fid, 'B', 1)[0] + 1900
        header['birth_month'] = birth_month
        header['birth_day'] = birth_day
        header['birth_year'] = birth_year

        logger.info('\t\tBirth Date: {}-{}-{}'.format(
            birth_year, birth_month, birth_day))

        reserved = np.fromfile(fid, 'B', 19)

        # Recording date and time
        rec_day = np.fromfile(fid, 'B', 1)[0]
        rec_month = np.fromfile(fid, 'B', 1)[0]
        rec_year = np.fromfile(fid, 'B', 1)[0] + 1900

        rec_hour = np.fromfile(fid, 'B', 1)[0]
        rec_min = np.fromfile(fid, 'B', 1)[0]
        rec_sec = np.fromfile(fid, 'B', 1)[0]
        header['rec_day'] = rec_day
        header['rec_month'] = rec_month
        header['rec_year'] = rec_year

        header['rec_hour'] = rec_hour
        header['rec_min'] = rec_min
        header['rec_sec'] = rec_sec

        logger.info(
            '\tRecording Date: {}-{}-{}'.format(rec_year, rec_month, rec_day))
        logger.info(
            '\tRecording Time: {}:{}:{}'.format(rec_hour, rec_min, rec_sec))

        aq_unit = np.fromfile(fid, 'u2', 1)[0]
        logger.info('\tAcqusition Unit: {}'.format(aq_unit))
        header['aq_unit'] = aq_unit

        file_type = np.fromfile(fid, 'u2', 1)[0]
        logger.info('\tFile Type: {}'.format(file_type))
        header['file_type'] = file_type

        data_start = np.fromfile(fid, 'u4', 1)[0]
        n_channels = np.fromfile(fid, 'u2', 1)[0]
        row_size = np.fromfile(fid, 'u2', 1)[0]
        logger.info(
            '\tData stored at {} in {} channels (multiplexer {})'.format(
                data_start, n_channels, row_size))
        header['data_start'] = data_start
        header['n_channels'] = n_channels
        header['row_size'] = row_size

        min_sample_freq = np.fromfile(fid, 'u2', 1)[0]
        logger.info('\tMin Sample Freq: {}'.format(min_sample_freq))
        header['min_sample_freq'] = min_sample_freq

        n_bytes_sample = np.fromfile(fid, 'u2', 1)[0]
        logger.info('\tBytes per Sample: {}'.format(n_bytes_sample))
        header['n_bytes_sample'] = n_bytes_sample

        compressed = np.fromfile(fid, 'u2', 1)[0]
        logger.info('\tCompressed: {}'.format(compressed))
        header['compressed'] = compressed
        if compressed == 1:
            logger.error('Cannot read compressed data')

        n_montages = np.fromfile(fid, 'u2', 1)[0]
        logger.info('\tNumber of Montages: {}'.format(n_montages))
        header['n_montages'] = n_montages

        video_start = np.fromfile(fid, 'u4', 1)[0]
        logger.info('\tVideo data starts at {}'.format(video_start))
        header['video_start'] = video_start

        video_sync = np.fromfile(fid, 'u2', 1)[0]
        logger.info('\tVideo sync: {}'.format(video_sync))
        header['video_sync'] = video_sync

        reserved = np.fromfile(fid, 'B', 16)

        logger.info('Reading descriptors')
        descriptors = {}

        keys = [
            'ORDER', 'LABCOD', 'NOTE', 'FLAGS', 'TRONCA', 'IMPED_B', 'IMPED_E',
            'MONTAGE', 'COMPRESS', 'AVERAGE', 'HISTORY', 'DVIDEO',
            'EVENT A', 'EVENT B', 'TRIGGER', 'BRAINIMG'
        ]

        for t_k in keys:
            descriptor = np.fromfile(fid, 'S1', 8).astype('U1')
            descriptor = ''.join(descriptor).strip()
            if descriptor != t_k:
                logger.warning(
                    '\tWrong descriptor for {} (found {})'.format(
                        t_k, descriptor))
            t_start = np.fromfile(fid, 'u4', 1)[0]
            t_len = np.fromfile(fid, 'u4', 1)[0]
            descriptors[t_k] = dict(start=t_start, len=t_len)
            logger.info('\t{}: @{} ({} bytes)'.format(t_k, t_start, t_len))

        reserved = np.fromfile(fid, 'B', 208)
        header['descriptors'] = descriptors

        # Read order of electodes
        fid.seek(descriptors['ORDER']['start'], 0)
        order = np.fromfile(fid, 'u2', descriptors['ORDER']['len'])
        order = order[:n_channels]

        orig_electrodes = []
        el_st = descriptors['LABCOD']['start']
        el_len = descriptors['LABCOD']['len']
        fid.seek(el_st, 0)
        el_end = el_st + el_len
        while (fid.tell() < el_end):
            t_el = {}
            t_el['status'] = np.fromfile(fid, 'B', 1)[0]
            t_el['type'] = np.fromfile(fid, 'B', 1)[0]
            t_el['label+'] = ''.join(
                np.fromfile(fid, 'S1', 6).astype('U1')).strip()
            t_el['label-'] = ''.join(
                np.fromfile(fid, 'S1', 6).astype('U1')).strip()
            t_el['log_min'] = np.fromfile(fid, 'i4', 1)[0]
            t_el['log_max'] = np.fromfile(fid, 'i4', 1)[0]
            t_el['log_gnd'] = np.fromfile(fid, 'i4', 1)[0]
            t_el['phys_min'] = np.fromfile(fid, 'i4', 1)[0]
            t_el['phys_max'] = np.fromfile(fid, 'i4', 1)[0]
            t_el['meas_unit'] = np.fromfile(fid, 'u2', 1)[0]
            t_el['pref_hpass_limit'] = np.fromfile(fid, 'u2', 1)[0]
            t_el['pref_hpass_type'] = np.fromfile(fid, 'u2', 1)[0]
            t_el['pref_lpass_limit'] = np.fromfile(fid, 'u2', 1)[0]
            t_el['pref_lpass_type'] = np.fromfile(fid, 'u2', 1)[0]
            t_el['srate_coef'] = np.fromfile(fid, 'u2', 1)[0]
            t_el['position'] = np.fromfile(fid, 'u2', 1)[0]
            t_el['latitude'] = np.fromfile(fid, 'f4', 1)[0]
            t_el['longitude'] = np.fromfile(fid, 'f4', 1)[0]
            t_el['present_map'] = np.fromfile(fid, 'B', 1)[0]
            t_el['present_avg'] = np.fromfile(fid, 'B', 1)[0]
            t_el['description'] = ''.join(
                np.fromfile(fid, 'S1', 32).astype('U1')).strip()
            t_el['pos_x'] = np.fromfile(fid, 'f4', 1)[0]
            t_el['pos_y'] = np.fromfile(fid, 'f4', 1)[0]
            t_el['pos_z'] = np.fromfile(fid, 'f4', 1)[0]
            t_el['pos_coord'] = np.fromfile(fid, 'u2', 1)[0]
            reserved = np.fromfile(fid, 'B', 24)  # noqa F841
            # print(t_el)
            orig_electrodes.append(t_el)

        srates = np.array(
            [x['srate_coef'] for x in [orig_electrodes[y] for y in order]])

        if np.unique(srates).shape[0] != 1:
            raise ValueError('Cannot read data with mixed sample rates')

        logger.info('Reading notes')
        notes = {}
        fid.seek(descriptors['NOTE']['start'], 0)
        keep_reading = True
        while keep_reading is True:
            sample = np.fromfile(fid, 'u4', 1)[0]
            if sample != 0:
                text = np.fromfile(fid, np.unicode_, 40)
                text = ''.join(text).strip()
                logger.info('\tNote at sample {}: {}'.format(sample, text))
                notes[sample] = text
            else:
                keep_reading = False
        header['notes'] = notes

        logger.info('Reading flags')
        flags = []
        fid.seek(descriptors['FLAGS']['start'], 0)
        keep_reading = True
        while keep_reading is True:
            sample_st = np.fromfile(fid, 'i4', 1)[0]
            sample_end = np.fromfile(fid, 'i4', 1)[0]
            if 0 in [sample_st or sample_end]:
                keep_reading = False
            if sample_st != 0:
                flags.append((sample_st, sample_end))
                logger.info(
                    '\Flag found [{} - {}]'.format(sample_st, sample_end))
        header['flags'] = flags

        logger.info('Reading segments description')
        segments = {}
        desc_st = descriptors['TRONCA']['start']
        desc_end = desc_st + descriptors['TRONCA']['len']
        fid.seek(desc_st, 0)
        keep_reading = True
        while keep_reading is True:
            time = np.fromfile(fid, 'u4', 1)[0]
            if time == 0 or fid.tell() >= desc_end:
                keep_reading = False
            else:
                sample = np.fromfile(fid, 'u4', 1)[0]
                segments[sample] = time

        # TODO: Fix to read shennan's bad TRC files
        # if len(segments) != 0:
        #     raise ValueError('Cannot read reduced file')
        header['segments'] = segments

        logger.info('Reading starting impedances')
        fid.seek(descriptors['IMPED_B']['start'], 0)
        for t_el in orig_electrodes:
            t_el['imped_b+'] = np.fromfile(fid, 'B', 1)[0]
            t_el['imped_b-'] = np.fromfile(fid, 'B', 1)[0]

        logger.info('Reading ending impedances')
        fid.seek(descriptors['IMPED_E']['start'], 0)
        for t_el in orig_electrodes:
            t_el['imped_e+'] = np.fromfile(fid, 'B', 1)[0]
            t_el['imped_e-'] = np.fromfile(fid, 'B', 1)[0]

        logger.info('Reading montages')
        fid.seek(descriptors['MONTAGE']['start'], 0)
        montages = {}
        for i_mtg in range(n_montages):
            t_montage = {}
            t_montage['lines'] = np.fromfile(fid, 'u2', 1)[0]
            t_montage['sectors'] = np.fromfile(fid, 'u2', 1)[0]
            t_montage['base_time'] = np.fromfile(fid, 'u2', 1)[0]
            t_montage['notch'] = np.fromfile(fid, 'u2', 1)[0]
            t_montage['colors'] = np.fromfile(fid, 'B', 128)
            t_montage['selection'] = np.fromfile(fid, 'B', 128)
            description = ''.join(
                    np.fromfile(fid, 'S1', 64).astype('U1')).strip()
            t_montage['description'] = description
            inputs = np.fromfile(fid, 'u2', 256)
            noninv = [orig_electrodes[x]['label+'] for x in inputs[::2]]
            inv = [orig_electrodes[x]['label+'] for x in inputs[1::2]]
            t_montage['inputs'] = np.c_[noninv, inv]
            t_montage['hipass'] = np.fromfile(fid, 'u4', 128)
            t_montage['lowpass'] = np.fromfile(fid, 'u4', 128)
            t_montage['reference'] = np.fromfile(fid, 'u4', 128)
            reserverd = np.fromfile(fid, 'B', 1720)
            montages[description] = t_montage
            # print(t_montage)
        header['montages'] = montages

        logger.info('Reading triggers')
        triggers = []
        desc_st = descriptors['TRIGGER']['start']
        desc_end = desc_st + descriptors['TRIGGER']['len']
        fid.seek(desc_st, 0)
        keep_reading = True
        while fid.tell() < desc_end and keep_reading is True:
            sample = np.fromfile(fid, 'u4', 1)[0]
            if sample == 0xFFFFFFFF:
                keep_reading = False
            else:
                value = np.fromfile(fid, 'u2', 1)[0]
                triggers.append((sample, value))
        header['triggers'] = triggers

        electrodes = [orig_electrodes[x] for x in order]
        header['electrodes'] = electrodes

        # Do computations to keep it simpler

        # number of samples
        fid.seek(0, 2)
        fsize = fid.tell()
        n_data_bytes = fsize - data_start
        header['n_data_bytes'] = n_data_bytes
        header['n_samples'] = int(n_data_bytes / row_size)

        # sample frequency
        sfreq = srates[0] * min_sample_freq
        header['sfreq'] = sfreq
        return header


class RawTRC(BaseRaw):
    """RawTRC class."""

    @verbose
    def __init__(self, input_fname, preload=False, verbose=None):
        logger.info('Reading TRC file header from {}'.format(input_fname))
        self.input_fname = input_fname
        header = self._read_header()

        # TODO: Compute cals

        info = _empty_info(header['sfreq'])

        electrodes = header['electrodes']
        ch_names = [t_el['label+'] for t_el in electrodes]

        # TODO: Add misc channels
        eog = []
        ecg = []
        emg = []
        misc = []

        log_gnd = np.array([x['log_gnd'] for x in electrodes])
        log_max = np.array([x['log_max'] for x in electrodes])
        log_min = np.array([x['log_min'] for x in electrodes])
        phys_max = np.array([x['phys_max'] for x in electrodes])
        phys_min = np.array([x['phys_min'] for x in electrodes])
        units = [x['meas_unit'] for x in electrodes]
        unit_scalar = [_micromed_units[x] for x in units]

        cals = (phys_max - phys_min) / (log_max - log_min + 1) * unit_scalar

        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, ecg, emg, misc)

        rec_time = datetime.datetime(
            header['rec_year'], header['rec_month'], header['rec_day'],
            header['rec_hour'], header['rec_min'], header['rec_sec'])
        rec_timestamp = time.mktime(rec_time.timetuple())
        info['meas_date'] = (rec_timestamp, 0)
        info['chs'] = chs
        info._update_redundant()
        header['log_gnd'] = log_gnd

        super(RawTRC, self).__init__(
            info, preload=preload, orig_format='float', filenames=[input_fname],
            last_samps=[header['n_samples'] - 1], raw_extras=[header],
            verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):

        header = self._raw_extras[fi]
        data_start = header['data_start']
        n_channels = header['n_channels']
        n_bytes_sample = header['n_bytes_sample']
        row_size = int(header['row_size'] / n_bytes_sample)
        sample_size_code = 'u{}'.format(n_bytes_sample)
        log_gnd = header['log_gnd']

        samples_to_read = int(stop - start)

        chunk_start = start * row_size * n_bytes_sample + data_start  # in bytes
        chunk_len = samples_to_read * row_size  # in samples
        logger.info('Reading {} samples from {} (len {})'.format(
            samples_to_read, chunk_start, chunk_len))

        with open(self.input_fname) as fid:
            fid.seek(chunk_start, 0)  # Go to start of reading chunk
            raw_data = np.fromfile(fid, sample_size_code, chunk_len)
            raw_data = raw_data.reshape(samples_to_read, n_channels)
            raw_data = (raw_data - log_gnd).T
            _mult_cal_one(data, raw_data, idx, cals, mult)

    def _read_header(self):
        return _read_raw_trc_header(self.input_fname)


@verbose
def read_raw_trc(input_fname, preload=False, include=None, verbose=None):
    return RawTRC(input_fname=input_fname, preload=preload, verbose=verbose)
