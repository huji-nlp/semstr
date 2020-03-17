Scheme Evaluation and Mapping for Structural Text Representation
================================================================

Collection of utilities for conversion and evaluation of semantic and syntactic text representation schemes.

### Requirements
* Python 3.6

### Install

Create a Python virtual environment:
    
    virtualenv --python=/usr/bin/python3 venv
    . venv/bin/activate              # on bash
    source venv/bin/activate.csh     # on csh

Install the latest release:

    pip install semstr

Alternatively, install the latest code from GitHub (may be unstable):

    git clone https://github.com/danielhers/semstr
    cd semstr
    pip install .

### Convert
To convert an SDP file to CoNLL-U, for example, run:
```
$ python semstr/convert.py test_files/20001001.sdp -f conllu
Converting: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 76.49file/s, file=20001001.sdp]
```
In this example, multiple heads are preserved in the `deps` column:
```
$ cat 20001001.conllu
# format = sdp
# sent_id = 20001001
# text = Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 .
1	Pierre	Pierre	NNP	NNP	_	0	root	0:root	_
2	Vinken	_generic_proper_ne_	NNP	NNP	_	1	compound	1:compound|6:ARG1|9:ARG1	_
3	,	_	,	,	_	1	orphan	1:orphan	_
4	61	_generic_card_ne_	CD	CD	_	1	orphan	1:orphan	_
5	years	year	NNS	NNS	_	4	ARG1	4:ARG1	_
6	old	old	JJ	JJ	_	5	measure	5:measure	_
7	,	_	,	,	_	1	orphan	1:orphan	_
8	will	will	MD	MD	_	1	orphan	1:orphan	_
9	join	join	VB	VB	_	1	orphan	1:orphan	_
10	the	the	DT	DT	_	1	orphan	1:orphan	_
11	board	board	NN	NN	_	9	ARG2	9:ARG2|10:BV	_
12	as	as	IN	IN	_	1	orphan	1:orphan	_
13	a	a	DT	DT	_	1	orphan	1:orphan	_
14	nonexecutive	_generic_jj_	JJ	JJ	_	1	orphan	1:orphan	_
15	director	director	NN	NN	_	12	ARG2	12:ARG2|13:BV|14:ARG1	_
16	Nov.	Nov.	NNP	NNP	_	1	orphan	1:orphan	_
17	29	_generic_dom_card_ne_	CD	CD	_	16	of	16:of	_
18	.	_	.	.	_	1	orphan	1:orphan	_
```
For any other source and target formats, just replace `test_files/20001001.sdp` and `conllu`.
Supported formats are: `json,conll,conllu,sdp,export,amr,txt`.

Author
------
* Daniel Hershcovich: daniel.hershcovich@gmail.com


License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](LICENSE.txt)).


[![Build Status (Travis CI)](https://travis-ci.org/danielhers/semstr.svg?branch=master)](https://travis-ci.org/danielhers/semstr)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/github/danielhers/semstr?svg=true)](https://ci.appveyor.com/project/danielh/semstr)
[![Build Status (Docs)](https://readthedocs.org/projects/semstr/badge/?version=latest)](http://semstr.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/SEMSTR.svg)](https://badge.fury.io/py/SEMSTR)
