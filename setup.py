"""
Setup script.
"""

from os import path
from setuptools import setup, find_packages

def read(fname):
    text = open(path.join(path.dirname(__file__), fname)).read()
    text = text.split('\n\n')
    name = text[0].split()[0]
    description = text[1].strip('.')
    long_description = text[2]
    return name, description, long_description


if __name__ == '__main__':
    NAME, DESCRIPTION, LONG_DESCRIPTION = read('README.md')

    setup(name=NAME,
          version='0.0.1',
          author='Matthew W. Hoffman',
          author_email='mwh30@cam.ac.uk',
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license='Simplified BSD',
          packages=find_packages(),
          package_data={'': ['*.txt', '*.npz']},
          install_requires=['numpy', 'scipy', 'matplotlib'])

