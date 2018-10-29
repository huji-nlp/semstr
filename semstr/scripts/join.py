#!/usr/bin/env python3
import sys
from itertools import filterfalse

import configargparse
import os
import re
import shutil
from glob import glob
from tqdm import tqdm
from ucca.ioutil import gen_files

desc = """Concatenate files according to order in reference"""

AMR_ID_PATTERN = re.compile("#\s*::id\s+(\S+)")
CONLLU_ID_PATTERN = re.compile("#\s*sent_id\s*=\s*(\S+)")
SDP_ID_PATTERN = re.compile("#\s*(\d+).*")


def find_ids(lines):
    for line in lines:
        m = AMR_ID_PATTERN.match(line) or SDP_ID_PATTERN.match(line) or CONLLU_ID_PATTERN.match(line)
        if m:
            yield m.group(1)


def main(args):
    with open(args.reference, encoding="utf-8") as f:
        # noinspection PyTypeChecker
        order = dict(map(reversed, enumerate(find_ids(f), start=1)))

    def _index(key_filename):
        basename = os.path.splitext(os.path.basename(key_filename))[0]
        index = order.get(basename) or order.get(basename.rpartition("_0")[0])
        if index is None:
            raise ValueError("Not found: " + basename)
        return index

    files = [f for pattern in args.filenames for f in gen_files(sorted(glob(pattern)) or [pattern])]
    if len(files) > len(order):
        raise ValueError("Files missing in reference: " + ", ".join(filterfalse(_index, files)))
    if len(order) > len(files):
        print("Warning: reference contains unmatched IDs", file=sys.stderr)
    t = tqdm(sorted(files, key=_index), desc="Writing " + args.out, unit=" files")
    with open(args.out, "wb") as out_f:
        for filename in t:
            t.set_postfix(f=filename)
            with open(filename, "rb") as f:
                shutil.copyfileobj(f, out_f)
                if args.add_newlines:
                    out_f.write(os.linesep.encode("utf-8"))


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("out", help="output file")
    argparser.add_argument("reference", help="file with headers determining the reference order")
    argparser.add_argument("filenames", nargs="+", help="directory or files to join, identified by filename")
    argparser.add_argument("-n", "--add-newlines", action="store_true", help="add extra newlines between files")
    main(argparser.parse_args())
