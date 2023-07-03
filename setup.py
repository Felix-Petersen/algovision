"""
algovision - a library for differentiable algorithms and integrating algorithms into neural networks.

Copyright (c) Felix Petersen.

This source code is licensed under the MIT license found in the LICENSE file.
"""

__author__ = "Felix Petersen"
__email__ = "ads0399@felix-petersen.de"

import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='algovision',
    version='v0.1.2',
    description='AlgoVision - A Framework for Differentiable Algorithms and Algorithmic Supervision',
    author='Felix Petersen',
    author_email=__email__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Felix-Petersen/algovision',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT License',
    package_dir={'algovision': 'algovision'},
    packages=['algovision'],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch>=1.9.0'
    ]
)
