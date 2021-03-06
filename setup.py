#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The setup script.
"""

import sys
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.10',
    'scipy',
    'urllib3',
    'matplotlib'
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
    'wheel',
    'flake8',
    'bump2version',
]


setup(
    name='spharpy',
    version='0.3.2',
    description="Python package for spherical array processing.",
    long_description=readme + '\n\n' + history,
    author="Marco Berzborn",
    author_email='marco.berzborn@akustik.rwth-aachen.de',
    url='https://git.rwth-aachen.de/mbe/spharpy/',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='spharpy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements
)
