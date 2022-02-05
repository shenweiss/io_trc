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
import struct
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

MAX_CAN = 256
MAX_LAB = 640
MAX_NOTE = 200
MAX_NOTE_SECTION = 1000
MAX_FLAG = 100
MAX_SAMPLE = 128 
MAX_HISTORY = 30
MAX_FILE = 1024 
MAX_TRIGGER = 8192
MAX_CAN_VIEW = 128
MAX_MONT = 30 
MAX_SEGM = 100
MAX_EVENT = 100
AVERAGE_FREE = 108


def _safe_unicode(text):
    return np.array([i if len(i) > 0 and ord(i) < 128 else '' 
                     for i in text]).astype('U1')


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
        order = np.fromfile(fid, 'u2', int(descriptors['ORDER']['len'] / 2))
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

            t_val = ''.join(_safe_unicode(np.fromfile(fid, 'S1', 6))).strip()
            t_el['label+'] = t_val
            
            t_val = ''.join(_safe_unicode(np.fromfile(fid, 'S1', 6))).strip()
            t_el['label-'] = t_val

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

            t_val = ''.join(_safe_unicode(np.fromfile(fid, 'S1', 32))).strip()
            t_el['description'] = t_val
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
        desc_st = descriptors['NOTE']['start']
        desc_end = desc_st + descriptors['NOTE']['len']
        fid.seek(desc_st, 0)
        keep_reading = True
        while keep_reading is True and fid.tell() < desc_end:
            sample = np.fromfile(fid, 'u4', 1)[0]
            if sample != 0:
                text = ''.join(
                    _safe_unicode(np.fromfile(fid, 'S1', 40))).strip()
                text = text
                logger.info('\tNote at sample {}: {}'.format(sample, text))
                notes[sample] = text
            else:
                keep_reading = False
        header['notes'] = notes

        logger.info('Reading flags')
        flags = []
        desc_st = descriptors['FLAGS']['start']
        desc_end = desc_st + descriptors['FLAGS']['len']
        fid.seek(desc_st, 0)
        keep_reading = True
        while keep_reading is True and fid.tell() < desc_end:
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
                    _safe_unicode(np.fromfile(fid, 'S1', 64))).strip()
            t_montage['description'] = description
            inputs = np.fromfile(fid, 'u2', 256)
            t_montage['_orig_inputs'] = inputs
            noninv = [orig_electrodes[x]['label+'] for x in inputs[::2]]
            inv = [orig_electrodes[x]['label+'] for x in inputs[1::2]]
            t_montage['inputs'] = np.c_[noninv, inv]
            t_montage['hipass'] = np.fromfile(fid, 'u4', 128)
            t_montage['lowpass'] = np.fromfile(fid, 'u4', 128)
            t_montage['reference'] = np.fromfile(fid, 'u4', 128)
            reserved = np.fromfile(fid, 'B', 1720)
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
        header['orig_electrodes'] = orig_electrodes

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


def _cvt_string(f_text, f_len, f_end, filling=0x20):
    b_text = f_text.encode('UTF-8')[:f_len]
    n_missing = f_len - len(b_text)
    b_text += bytes([filling] * n_missing)
    if f_end is not None:
        b_text += bytes(f_end)
    return b_text

_def_el_32 = {
    'Fp1': (90.0, 108.0), 
    'Fp2': (90.0, 72.0), 
    'F3': (61.79999923706055, 130.6999969482422), 
    'F4': (61.79999923706055, 49.29999923706055), 
    'F7': (90.0, 144.0), 
    'F8': (90.0, 36.0), 
    'Fz': (45.0, 90.0), 
    'C3': (45.0, 180.0), 
    'C4': (45.0, 0.0), 
    'Cz': (0.0, 0.0), 
    'P3': (61.79999923706055, 229.3000030517578), 
    'P4': (61.79999923706055, 310.70001220703125), 
    'Pz': (45.0, 270.0), 
    'O1': (90.0, 252.0), 
    'O2': (90.0, 288.0), 
    'T3': (90.0, 180.0), 
    'T4': (90.0, 0.0), 
    'T5': (90.0, 216.0), 
    'T6': (90.0, 324.0), 
    'Fpz': (90.0, 90.0), 
    'Oz': (90.0, 270.0), 
    'A2': (0.0, 0.0), 
    'A1': (0.0, 0.0), 
    '.....': (0.0, 0.0), 
    '.....': (0.0, 0.0), 
    '.....': (0.0, 0.0), 
    '.....': (0.0, 0.0), 
    '.....': (0.0, 0.0), 
    '.....': (0.0, 0.0), 
    '.....': (0.0, 0.0), 
    '.....': (0.0, 0.0)
}

def_el_172_176 = {
    'PULS+': (0.0, 0.0, 102), 
    'BEAT+': (0.0, 0.0, 101), 
    'MKR+': (0.0, 0.0, 0), 
    'SpO2+': (0.0, 0.0, 100)
}

def _default_electrodes():
    electrodes = []
    keys = ['status', 'type', 'label+', 'label-', 'log_min', 'log_max', 'log_gnd',
            'phys_min', 'phys_max', 'meas_unit', 'pref_hpass_limit', 
            'pref_hpass_type', 'pref_lpass_limit', 'pref_lpass_type', 'srate_coef',
            'position', 'latitude', 'longitude', 'present_map', 'present_avg',
            'description', 'pos_x', 'pos_y', 'pos_z', 'pos_coord']
    for i in range(0, MAX_LAB):
        t_el = {}
        for t_k in keys:
            t_el[t_k] = 0
        t_el['type'] = 0
        t_el['label+'] = '.....'
        t_el['label-'] = '.....'   
        t_el['position'] = 0
        t_el['srate_coef'] = 1 
        t_el['description'] = ''
        electrodes.append(t_el)
    
    electrodes[0]['status'] = 0
    electrodes[0]['type'] = 0
    electrodes[0]['label+'] = 'AVG'
    electrodes[0]['label-'] = 'G2'
    electrodes[0]['log_max'] = 65535
    electrodes[0]['log_gnd'] = 32768
    electrodes[0]['phys_min'] = -3200
    electrodes[0]['phys_max'] = 3200
    electrodes[0]['pref_hpass_limit'] = 0
    electrodes[0]['present_avg'] = 1

    # for i, (lbl, (lat, lon)) in enumerate(_def_el_32.items()):
    #     electrodes[i+1]['status'] = 0
    #     electrodes[i+1]['type'] = 0
    #     electrodes[i+1]['label+'] = lbl
    #     electrodes[i+1]['label-'] = 'G2'
    #     electrodes[i+1]['log_max'] = 65535
    #     electrodes[i+1]['log_gnd'] = 32768
    #     electrodes[i+1]['phys_min'] = -3200
    #     electrodes[i+1]['phys_max'] = 3200
    #     electrodes[i+1]['pref_hpass_limit'] = 0
    #     electrodes[i+1]['present_avg'] = 1
    #     electrodes[i+1]['latitude'] = lat
    #     electrodes[i+1]['longitude'] = lon

    # for i, (lbl, (lat, lon, unit)) in enumerate(def_el_172_176.items()):
    #     electrodes[i+172]['status'] = 0
    #     electrodes[i+172]['type'] = 0
    #     electrodes[i+172]['label+'] = lbl
    #     electrodes[i+172]['label-'] = 'G2'
    #     electrodes[i+172]['log_max'] = 65535
    #     electrodes[i+172]['log_gnd'] = 32768
    #     electrodes[i+172]['phys_min'] = -3200
    #     electrodes[i+172]['phys_max'] = 3200
    #     electrodes[i+172]['pref_hpass_limit'] = 0
    #     electrodes[i+172]['present_avg'] = 1
    #     electrodes[i+172]['latitude'] = lat
    #     electrodes[i+172]['longitude'] = lon
    #     electrodes[i+172]['meas_unit'] = unit


    return electrodes

@verbose
def _write_raw_trc_header(raw, fid, verbose=None):
    # MAX 256 Channels
    if raw.info['nchan'] > MAX_CAN:
        raise ValueError('Cannot save more than {} channels'.format(MAX_CAN))

    logger.info('Writing header')
    version = _cvt_string('* MICROMED  Brain-Quick file *', 30, [0x00, 0x1A])
    fid.write(version)

    #TODO: Get this info right
    header = {}
    header['laboratory'] = ''
    header['surname'] = 'LName'
    header['name'] = 'Name'
    header['birth_month'] = 4
    header['birth_day'] = 25
    header['birth_year'] = 1985
    header['aq_unit'] = 35
    header['file_type'] = 74

    laboratory = _cvt_string(header['laboratory'], 31, [0x00])
    fid.write(laboratory)

    surname = _cvt_string(header['surname'], 22, None)
    fid.write(surname)

    name = _cvt_string(header['name'], 20, None)
    fid.write(name)

    birth_date = bytes([
        header['birth_month'],
        header['birth_day'],
        header['birth_year'] - 1900])

    fid.write(birth_date)

    reserved = bytes([0x00] * 19)
    fid.write(reserved)

#   rec_time = time.gmtime(raw.info['meas_date'][0])
    
    rec_time_b = bytes([
        1,
        1,
        10,
        9,
        0,
        0])
    fid.write(rec_time_b)

    fid.write(struct.pack('H', header['aq_unit']))
    fid.write(struct.pack('H', header['file_type']))

    # Keep this for later
    to_write_later_offsets = {}
    to_write_later_values = {}
    to_write_later_offsets['DATA_START'] = fid.tell()
    fid.write(struct.pack('I', 0xFFFFFFFF))

    fid.write(struct.pack('H', raw.info['nchan']))

    # 16 bits elements
    fid.write(struct.pack('H', raw.info['nchan'] * 2))

    fid.write(struct.pack('H', int(raw.info['sfreq'])))

    fid.write(struct.pack('H', 2))

    # Non compressed
    fid.write(struct.pack('H', 0))

    # No montages for now
    to_write_later_offsets['N_MONTAGES'] = fid.tell()
    fid.write(struct.pack('H', 0))

    # No video
    fid.write(struct.pack('I', 0xFFFFFFFF))
    fid.write(struct.pack('H', 0))

    reserved = bytes([0x00] * 15)
    fid.write(reserved)

    # Write File Type
    fid.write(bytes([0x4]))

    # Write descriptors
    descriptor_keys = [
        'ORDER', 'LABCOD', 'NOTE', 'FLAGS', 'TRONCA', 'IMPED_B', 'IMPED_E',
        'MONTAGE', 'COMPRESS', 'AVERAGE', 'HISTORY', 'DVIDEO',
        'EVENT A', 'EVENT B', 'TRIGGER', 'BRAINIMG'
    ]

    for t_k in descriptor_keys:
        name = _cvt_string(t_k, 8, None)
        fid.write(name)
        to_write_later_offsets['{}_START'.format(t_k)] = fid.tell()
        fid.write(struct.pack('I', 0xFFFFFFFF))
        to_write_later_offsets['{}_LEN'.format(t_k)] = fid.tell()
        fid.write(struct.pack('I', 0xFFFFFFFF))


    reserved = bytes([0x00] * 208)
    fid.write(reserved)

    # Create electrodes
    hpass = raw.info['highpass']
    lpass = raw.info['lowpass']
    electrodes = _default_electrodes()

    for i, ch_name in enumerate(raw.ch_names):
        idx = i + 1
        # idx = i + 33
        # if idx >= 172:
        #     idx += 4

        if len(ch_name) > 5:
            logger.warning('Channel {} will be cut to {}'.format(
                ch_name, ch_name[:5]))
        electrodes[idx]['status'] = 1
        electrodes[idx]['label+'] = ch_name[:5]
        electrodes[idx]['label-'] = 'G2'
        electrodes[idx]['log_max'] = 65535
        electrodes[idx]['log_gnd'] = 32768
        electrodes[idx]['phys_min'] = -3200
        electrodes[idx]['phys_max'] = 3200
        electrodes[idx]['position'] = i
        electrodes[idx]['pref_hpass_limit'] = 150
        electrodes[idx]['present_avg'] = 0

    # Write order of electrodes up to 256
    electrodes_names = [x['label+'] for x in electrodes]
    order = np.zeros(MAX_CAN, dtype=np.uint16)
    for i, ch_name in enumerate(raw.ch_names):
        idx = electrodes_names.index(ch_name)
        order[i] = idx
    start = fid.tell()
    fid.write(struct.pack('{}H'.format(MAX_CAN), *order))
    end = fid.tell()
    to_write_later_values['ORDER_START'] = start
    to_write_later_values['ORDER_LEN'] = end - start

    # Write electrodes
    start = fid.tell()
    for i in range(0, MAX_LAB):
        t_el = electrodes[i]
        
        fid.write(struct.pack('B', t_el['status']))
        fid.write(struct.pack('B', t_el['type']))
        fid.write(_cvt_string(t_el['label+'], 5, [0x00], filling=0))
        fid.write(_cvt_string(t_el['label-'], 5, [0x00], filling=0))
        fid.write(struct.pack('i', t_el['log_min']))
        fid.write(struct.pack('i', t_el['log_max']))
        fid.write(struct.pack('i', t_el['log_gnd']))
        fid.write(struct.pack('i', t_el['phys_min']))
        fid.write(struct.pack('i', t_el['phys_max']))
        fid.write(struct.pack('H', t_el['meas_unit']))
        fid.write(struct.pack('H', t_el['pref_hpass_limit']))
        fid.write(struct.pack('H', t_el['pref_hpass_type']))
        fid.write(struct.pack('H', t_el['pref_lpass_limit']))
        fid.write(struct.pack('H', t_el['pref_lpass_type']))
        fid.write(struct.pack('H', t_el['srate_coef']))
        fid.write(struct.pack('H', t_el['position']))
        fid.write(struct.pack('f', t_el['latitude']))
        fid.write(struct.pack('f', t_el['longitude']))
        fid.write(struct.pack('B', t_el['present_map']))
        fid.write(struct.pack('B', t_el['present_avg']))
        fid.write(_cvt_string(t_el['description'], 31, [0x00], filling=0))
        fid.write(struct.pack('f', t_el['pos_x']))
        fid.write(struct.pack('f', t_el['pos_y']))
        fid.write(struct.pack('f', t_el['pos_z']))
        fid.write(struct.pack('H', t_el['pos_coord']))
        reserved = bytes([0x00] * 24)
        fid.write(reserved)

    end = fid.tell()
    to_write_later_values['LABCOD_START'] = start
    to_write_later_values['LABCOD_LEN'] = end - start

    # Write notes
    start = fid.tell()
    fid.write(struct.pack('I', 1))
    fid.write(_cvt_string('Created with FastWave TRC Writer', 40, None))
    for i in range(MAX_NOTE_SECTION-1):
        fid.write(bytes([0x00] * 44))
    end = fid.tell()
    to_write_later_values['NOTE_START'] = start
    to_write_later_values['NOTE_LEN'] = end - start

    # Write flags
    start = fid.tell()
    fid.write(bytes([0x00] * 8 * MAX_FLAG))
    end = fid.tell()
    to_write_later_values['FLAGS_START'] = start
    to_write_later_values['FLAGS_LEN'] = end - start

    # Write segments
    start = fid.tell()
    fid.write(bytes([0x00] * 8 * MAX_SEGM))
    end = fid.tell()
    to_write_later_values['TRONCA_START'] = start
    to_write_later_values['TRONCA_LEN'] = end - start

    # Write start impedances
    start = fid.tell()
    fid.write(bytes([0xFF] * 2 * MAX_CAN))
    end = fid.tell()
    to_write_later_values['IMPED_B_START'] = start
    to_write_later_values['IMPED_B_LEN'] = end - start

    # Write end impedances
    start = fid.tell()
    fid.write(bytes([0xFF] * 2 * MAX_CAN))
    end = fid.tell()
    to_write_later_values['IMPED_E_START'] = start
    to_write_later_values['IMPED_E_LEN'] = end - start

    # Write montages

    # Create subject specific Ref. montage
    montages = []
    n_lines = min(MAX_CAN_VIEW, raw.info['nchan'])
    t_montage = {}
    t_montage['lines'] = n_lines
    t_montage['sectors'] = 0
    t_montage['base_time'] = 15
    t_montage['notch'] = 1
    t_montage['colors'] = np.ones(MAX_CAN_VIEW, dtype=np.uint8) * 6
    t_montage['selection'] = np.zeros(MAX_CAN_VIEW, dtype=np.uint8)
    t_montage['description'] = 'Ref.'
    _ref_ch = np.zeros(MAX_CAN_VIEW, dtype=np.uint16)
    _idx_ch = np.zeros(MAX_CAN_VIEW, dtype=np.uint16)
    for i, x in enumerate(raw.ch_names[:n_lines]):
        _idx_ch[i] = electrodes_names.index(x)
    inputs = np.hstack(np.c_[_ref_ch, _idx_ch])
    t_montage['inputs'] = inputs
    hipass = np.zeros(MAX_CAN_VIEW, dtype=np.uint32)
    hipass[:n_lines] = 530
    t_montage['hipass'] = hipass
    lowpass = np.zeros(MAX_CAN_VIEW, dtype=np.uint32)
    lowpass[:n_lines] = 60000
    t_montage['lowpass'] = lowpass
    reference = np.zeros(MAX_CAN_VIEW, dtype=np.uint32)
    reference[:n_lines] = 400
    t_montage['reference'] = reference
    montages.append(t_montage)

    n_montages = len(montages)
    to_write_later_values['N_MONTAGES'] = n_montages

    start = fid.tell()
    for i, t_m in enumerate(montages):
        fid.write(struct.pack('H', t_m['lines']))
        fid.write(struct.pack('H', t_m['sectors']))
        fid.write(struct.pack('H', t_m['base_time']))
        fid.write(struct.pack('H', t_m['notch']))
        fid.write(t_m['colors'].astype('u1').tobytes())
        fid.write(t_m['selection'].astype('u1').tobytes())
        fid.write(_cvt_string(t_m['description'], 63, [0x00], filling=0))
        fid.write(t_m['inputs'].astype('u2').tobytes())
        fid.write(t_m['hipass'].astype('u4').tobytes())
        fid.write(t_m['lowpass'].astype('u4').tobytes())
        fid.write(t_m['reference'].astype('u4').tobytes())
        fid.write(bytes([0x00] * 1560))
        fid.write(bytes([0x05] * t_m['lines']))  # Color black
        fid.write(bytes([0x00] * (MAX_CAN_VIEW - t_m['lines'])))
        fid.write(bytes([0x00] * 32))

    n_field = 8 + MAX_CAN_VIEW * 2 + 64 + 4 * MAX_CAN_VIEW * 4 + 1720
    for i in range(n_montages, MAX_MONT):
        fid.write(bytes([0x00] * n_field))
    end = fid.tell()
    to_write_later_values['MONTAGE_START'] = start
    to_write_later_values['MONTAGE_LEN'] = end - start

    # Write compression
    start = fid.tell()
    fid.write(bytes([0x00] * 10))
    end = fid.tell()
    to_write_later_values['COMPRESS_START'] = start
    to_write_later_values['COMPRESS_LEN'] = end - start

    # Write average
    start = fid.tell()
    fid.write(bytes([0x00] * (AVERAGE_FREE + 5 * 4)))
    end = fid.tell()
    to_write_later_values['AVERAGE_START'] = start
    to_write_later_values['AVERAGE_LEN'] = end - start

    # Write history
    start = fid.tell()
    fid.write(bytes([0xFF] * MAX_SAMPLE * 4))
    for i in range(MAX_HISTORY):
        fid.write(bytes([0x00] * 4096))
    end = fid.tell()
    to_write_later_values['HISTORY_START'] = start
    to_write_later_values['HISTORY_LEN'] = end - start

    # Write dvideo
    start = fid.tell()
    for i in range(MAX_FILE):
        fid.write(bytes([0xFF] * 12))
        fid.write(bytes([0xFF] * 4))
    end = fid.tell()
    to_write_later_values['DVIDEO_START'] = start
    to_write_later_values['DVIDEO_LEN'] = end - start

    # Write event a
    start = fid.tell()
    fid.write(bytes([0x00] * (64 + MAX_EVENT * 8)))
    end = fid.tell()
    to_write_later_values['EVENT A_START'] = start
    to_write_later_values['EVENT A_LEN'] = end - start

    # Write event b
    start = fid.tell()
    fid.write(bytes([0x00] * (64 + MAX_EVENT * 8)))
    end = fid.tell()
    to_write_later_values['EVENT B_START'] = start
    to_write_later_values['EVENT B_LEN'] = end - start

    # Write triggers
    start = fid.tell()
    fid.write(bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF] * MAX_TRIGGER))
    end = fid.tell()
    to_write_later_values['TRIGGER_START'] = start
    to_write_later_values['TRIGGER_LEN'] = end - start

    # Write brain image
    to_write_later_values['BRAINIMG_START'] = 0
    to_write_later_values['BRAINIMG_LEN'] = 0


    to_write_later_values['DATA_START'] = fid.tell()

    for key in descriptor_keys:
        offset_start = to_write_later_offsets['{}_START'.format(key)]
        value_start = to_write_later_values['{}_START'.format(key)]
        fid.seek(offset_start, 0)
        fid.write(struct.pack('I', value_start))
        offset_len = to_write_later_offsets['{}_LEN'.format(key)]
        value_len = to_write_later_values['{}_LEN'.format(key)]
        logger.info('\t {} starts at {} (len = {})'.format(
                key, value_start, value_len))
        fid.seek(offset_len, 0)
        fid.write(struct.pack('I', value_len))

    offset = to_write_later_offsets['DATA_START']
    value = to_write_later_values['DATA_START']
    fid.seek(offset, 0)
    fid.write(struct.pack('I', value))

    offset = to_write_later_offsets['N_MONTAGES']
    value = to_write_later_values['N_MONTAGES']
    fid.seek(offset, 0)
    fid.write(struct.pack('H', value))

    fid.seek(to_write_later_values['DATA_START'], 0)


@verbose
def _write_raw_trc_data(raw, fid, verbose=None):
    logger.info('Writing data')
    start_time = time.time()
    log_min = 0
    log_max = 65535
    log_gnd = 32768
    phys_min = -3200
    phys_max = 3200
    unit_scalar = 1e-6

    cals = (phys_max - phys_min) / (log_max - log_min + 1) * unit_scalar

    if ((np.min(raw._data) * unit_scalar) < phys_min or 
            (np.max(raw._data) * unit_scalar) > phys_max):
        logger.warning(
                'Data is out of the range [{}, {}] uV'.format(
                    phys_min, phys_max))

    data = raw._data / cals
    data += log_gnd
    data = np.hstack(data.T)
    fid.write(data.astype('u2').tobytes())
    end_time = time.time()
    logger.info('File data written in {} seconds'.format(end_time - start_time))


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
        start_time = time.time()
        with open(self._filenames[0]) as fid:
            fid.seek(chunk_start, 0)  # Go to start of reading chunk
            raw_data = np.fromfile(fid, sample_size_code, chunk_len)
            raw_data = raw_data.reshape(samples_to_read, n_channels)
            raw_data = (raw_data - log_gnd).T
            _mult_cal_one(data, raw_data, idx, cals, mult)
        end_time = time.time()
        logger.info('Read data in {} seconds'.format(end_time - start_time))

    def _read_header(self):
        return _read_raw_trc_header(self.input_fname)


@verbose
def read_raw_trc(input_fname, preload=False, include=None, verbose=None):
    return RawTRC(input_fname=input_fname, preload=preload, verbose=verbose)


@verbose
def write_raw_trc(raw, output_fname, verbose=None):
    with open(output_fname, 'wb') as fid:
        _write_raw_trc_header(raw, fid)
        _write_raw_trc_data(raw, fid)
