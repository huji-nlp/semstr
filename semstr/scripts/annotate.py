#!/usr/bin/env python3

import argparse

from tqdm import tqdm
from ucca.ioutil import write_passage
from ucca.textutil import annotate_all

from semstr.cfgutil import read_specs, add_specs_args
from semstr.convert import FROM_FORMAT
from semstr.scripts.udpipe import annotate_udpipe

desc = """Read passages in any format, and write back with spaCy/UDPipe annotations."""


def main(args):
    for passages, out_dir, lang, udpipe in read_specs(args, converters=FROM_FORMAT):
        if udpipe:
            passages = annotate_udpipe(passages, udpipe, args.verbose)
        for passage in annotate_all(passages if args.verbose else
                                    tqdm(passages, unit=" passages", desc="Annotating " + out_dir),
                                    as_array=args.as_array, replace=not udpipe, lang=lang, verbose=args.verbose):
            write_passage(passage, outdir=out_dir, verbose=args.verbose, binary=args.binary)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("-a", "--as-array", action="store_true", help="save annotations as array in passage level")
    argparser.add_argument("-v", "--verbose", action="store_true", help="print tagged text for each passage")
    main(argparser.parse_args())
