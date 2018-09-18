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

from os import path as op

from numpy.distutils.core import setup

# get the version
version = None
with open(op.join('trcio', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DIST = 'trcio'
DESC = 'TRC File IO Module.'
URL = 'https://gitlab.liaa.dc.uba.ar/tju-uba/io_trc'
VERSION = version


if __name__ == "__main__":
    setup(name=DIST,
          maintainer='Federico Raimondo',
          maintainer_email='federaimondo@fastwavellc.com',
          description=DESC,
          license='Propietary',
          url=URL,
          version=VERSION,
          download_url=URL,
          long_description=open('README.md').read(),
          classifiers=['Programming Language :: Python'],
          platforms='any',
          packages=['trcio'],
          package_data={'trcio': []})
