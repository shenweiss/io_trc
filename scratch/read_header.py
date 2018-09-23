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

header = trcio.io._read_raw_trc_header(fname)

fid = open(fname, 'r')
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
    logger.warning('This reader is intended for micromed System98 files (4)')

fid.seek(0, 0)

# Version
version = np.fromfile(fid, 'S1', 32).astype('U1')
version = ''.join(version[:-2])  # Discard last 2 chars 0x00 0x1A

logger.info('Reading {}'.format(version))
logger.info('Reading Recording data')

# Laboratory
laboratory = np.fromfile(fid, 'S1', 32).astype('U1')
laboratory = ''.join(laboratory[:-1])  # Discard last char 0x00
logger.info('\tLaboratory: {}'.format(laboratory))

# Patient Data
logger.info('\tPatient data')

surname = np.fromfile(fid, 'S1', 22).astype('U1')
surname = ''.join(surname)
logger.info('\t\tSurname: {}'.format(surname))

name = np.fromfile(fid, 'S1', 20).astype('U1')
name = ''.join(name)
logger.info('\t\tName: {}'.format(name))

birth_month = np.fromfile(fid, 'B', 1)[0]
birth_day = np.fromfile(fid, 'B', 1)[0]
birth_year = np.fromfile(fid, 'B', 1)[0] + 1900

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

logger.info('\tRecording Date: {}-{}-{}'.format(rec_year, rec_month, rec_day))
logger.info('\tRecording Time: {}:{}:{}'.format(rec_hour, rec_min, rec_sec))

aq_unit = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tAcqusition Unit: {}'.format(aq_unit))

file_type = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tFile Type: {}'.format(file_type))

data_start = np.fromfile(fid, 'u4', 1)[0]
n_channels = np.fromfile(fid, 'u2', 1)[0]
row_size = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tData stored at {} in {} channels (multiplexer {})'.format(
    data_start, n_channels, row_size))

min_sample_freq = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tMin Sample Freq: {}'.format(min_sample_freq))
n_bytes_sample = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tBytes per Sample: {}'.format(n_bytes_sample))
compressed = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tCompressed: {}'.format(compressed))
if compressed == 1:
    logger.error('Cannot read compressed data')

n_montages = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tNumber of Montages: {}'.format(n_montages))

video_start = np.fromfile(fid, 'u4', 1)[0]
logger.info('\tVideo data starts at {}'.format(video_start))

video_sync = np.fromfile(fid, 'u2', 1)[0]
logger.info('\tVideo sync: {}'.format(video_sync))

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
            '\tWrong descriptor for {} (found {})'.format(t_k, descriptor))
    t_start = np.fromfile(fid, 'u4', 1)[0]
    t_len = np.fromfile(fid, 'u4', 1)[0]
    descriptors[t_k] = dict(start=t_start, len=t_len)
    logger.info('\t{}: @{} ({} bytes)'.format(t_k, t_start, t_len))

reserved = np.fromfile(fid, 'B', 208)


# Read order of electodes
fid.seek(descriptors['ORDER']['start'], 0)
order = np.fromfile(fid, 'u2', descriptors['ORDER']['len'])



order = order[:n_channels]

electrodes = []
el_st = descriptors['LABCOD']['start']
el_len = descriptors['LABCOD']['len']
fid.seek(el_st, 0)
el_end = el_st + el_len
while (fid.tell() < el_end):
    t_el = {}
    t_el['status'] = np.fromfile(fid, 'B', 1)[0]
    t_el['type'] = np.fromfile(fid, 'B', 1)[0]
    t_el['label+'] = ''.join(np.fromfile(fid, 'S1', 6).astype('U1')).strip()
    t_el['label-'] = ''.join(np.fromfile(fid, 'S1', 6).astype('U1')).strip()
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
    reserved = np.fromfile(fid, 'B', 24)
    # print(t_el)
    electrodes.append(t_el)

# electrodes = [electrodes[x] for x in order]
# srates = np.array([x['srate_coef'] for x in electrodes])
#
# if np.unique(srates).shape[0] != 1:
#     raise ValueError('Cannot read data with mixed sample rates')
#
# logger.info('Reading notes')
# notes = {}
# fid.seek(descriptors['NOTE']['start'], 0)
# keep_reading = True
# while keep_reading is True:
#     sample = np.fromfile(fid, 'u4', 1)[0]
#     if sample != 0:
#         text = ''.join(np.fromfile(fid, 'S1', 40).astype('U1')).strip()
#         logger.info('\tNote at sample {}: {}'.format(sample, text))
#         notes[sample] = text
#     else:
#         keep_reading = False
#
# logger.info('Reading flags')
# flags = []
# fid.seek(descriptors['FLAGS']['start'], 0)
# keep_reading = True
# while keep_reading is True:
#     sample_st = np.fromfile(fid, 'i4', 1)[0]
#     sample_end = np.fromfile(fid, 'i4', 1)[0]
#     if 0 in [sample_st or sample_end]:
#         keep_reading = False
#     if sample_st != 0:
#         flags.append((sample_st, sample_end))
#         logger.info('\Flag found [{} - {}]'.format(sample_st, sample_end))
#
# logger.info('Reading segments description')
# segments = {}
# desc_st = descriptors['TRONCA']['start']
# desc_end = desc_st + descriptors['TRONCA']['len']
# fid.seek(desc_st, 0)
# keep_reading = True
# while keep_reading is True:
#     time = np.fromfile(fid, 'u4', 1)[0]
#     if time == 0 or fid.tell() >= desc_end:
#         keep_reading = False
#     else:
#         sample = np.fromfile(fid, 'u4', 1)[0]
#         segments[sample] = time
#
# if len(segments) != 0:
#     raise ValueError('Cannot read reduced file')
#
# logger.info('Reading starting impedances')
# fid.seek(descriptors['IMPED_B']['start'], 0)
# for t_el in electrodes:
#     t_el['imped_b+'] = np.fromfile(fid, 'B', 1)[0]
#     t_el['imped_b-'] = np.fromfile(fid, 'B', 1)[0]
#
# logger.info('Reading ending impedances')
# fid.seek(descriptors['IMPED_E']['start'], 0)
# for t_el in electrodes:
#     t_el['imped_e+'] = np.fromfile(fid, 'B', 1)[0]
#     t_el['imped_e-'] = np.fromfile(fid, 'B', 1)[0]
#
# logger.info('Reading triggers')
# triggers = []
# desc_st = descriptors['TRIGGER']['start']
# desc_end = desc_st + descriptors['TRIGGER']['len']
# fid.seek(desc_st, 0)
# keep_reading = True
# while fid.tell() < desc_end and keep_reading is True:
#     sample = np.fromfile(fid, 'u4', 1)[0]
#     if sample == 0xFFFFFFFF:
#         keep_reading = False
#     else:
#         value = np.fromfile(fid, 'u2', 1)[0]
#         triggers.append((sample, value))
