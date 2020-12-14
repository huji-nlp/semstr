#!/bin/bash
set -xe

pip install collective.checkdocs twine
python setup.py checkdocs || exit 1
python setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
