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
import struct

fname_in = '/Users/fraimondo/data/intra/EEG_12.TRC'
fname_out = '/Users/fraimondo/data/intra/EEG_12_b.TRC'

raw = trcio.read_raw_trc(fname_in, preload=False, include=None)
header = raw._raw_extras[0]

def _cvt_string(f_text, f_len, f_end):
    b_text = f_text.encode('UTF-8')[:f_len]
    n_missing = f_len - len(b_text)
    b_text += bytes([0x00] * n_missing)
    if f_end is not None:
        b_text += bytes(f_end)
    return b_text

fid = open(fname_out, 'wb')
version = _cvt_string(header['version'], 30, [0x00, 0x1A])
fid.write(version)

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

rec_time = time.gmtime(raw.info['meas_date'][0])

rec_time_b = bytes([
    rec_time.tm_mday,
    rec_time.tm_mon,
    rec_time.tm_year - 1900,
    rec_time.tm_hour,
    rec_time.tm_min,
    rec_time.tm_sec])
fid.write(rec_time_b)

fid.write(struct.pack('H', header['aq_unit']))
fid.write(struct.pack('H', header['file_type']))

# Keep this for later
data_start_pos = fid.tell()
fid.write(struct.pack('I', 0xFFFFFFFF))

fid.write(struct.pack('H', raw.info['nchan']))

# 16 bits elements
fid.write(struct.pack('H', raw.info['nchan'] * 2))

fid.write(struct.pack('H', int(raw.info['sfreq'])))

fid.write(struct.pack('H', 2))

# Non compressed
fid.write(struct.pack('H', 0))

# No montages
fid.write(struct.pack('H', 0))

# No video
fid.write(struct.pack('I', 0xFFFFFFFF))
fid.write(struct.pack('H', 0))

reserved = bytes([0x00] * 15)
fid.write(reserved)

fid.write(bytes([0x4]))