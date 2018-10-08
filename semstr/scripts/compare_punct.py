#!/usr/bin/env python3

import os
import sys

import configargparse
from tqdm import tqdm
from ucca import ioutil, layer0

from semstr.cfgutil import add_verbose_arg
from semstr.evaluate import read_files

desc = """Read annotations with the same tokens, checking if a terminal is annotated as punctuation in one iff
it is annotated as punctuation in the other."""


def compare_punct(files, name=None, verbose=0, basename=False, matching_ids=False, **kwargs):
    guessed, ref = [iter(read_files(f, verbose=verbose, force_basename=basename, **kwargs)) for f in files]
    for (g, r) in tqdm(zip(guessed, ref), unit=" passages", desc=name, total=len(files[-1])):
        if matching_ids:
            while g.ID < r.ID:
                g = next(guessed)
            while g.ID > r.ID:
                r = next(ref)
        terminals = [(f.converted if f.format else f.passage).layer(layer0.LAYER_ID).all for f in (g, r)]
        for t1, t2 in zip(*terminals):
            assert t1.text == t2.text, "Terminal text: %s != %s (passage %s, terminal %s)" % (t1, t2, r.ID, t1.ID)
            if t1.punct != t2.punct:
                if verbose:
                    with ioutil.external_write_mode():
                        print("Passage %s: terminal '%s' (%s) is %s in left passage but %s in right passage" % (
                            r.ID, t1, t1.ID, t1.tag, t2.tag))
                yield r.ID, t1, t2


def main(args):
    files = [[os.path.join(d, f) for f in os.listdir(d) if not os.path.isdir(os.path.join(d, f))]
             if os.path.isdir(d) else [d] for d in (args.guessed, args.ref)]
    errors = list(compare_punct(files, **vars(args)))
    if errors:
        if not args.quiet:
            sys.stderr.flush()
            sys.stdout.flush()
            print("Found %d mismatches:" % len(errors))
            for i, t1, t2 in errors:
                print(i, t1.ID, t1, t1.tag, t2.tag)
        sys.exit(1)


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("guessed", help="filename/directory for the guessed annotation(s)")
    argparser.add_argument("ref", help="filename/directory for the reference annotation(s)")
    argparser.add_argument("-f", "--format", default="amr", help="default format (if cannot determine by suffix)")
    argparser.add_argument("-i", "--matching-ids", action="store_true", help="skip passages without a match (by ID)")
    argparser.add_argument("-b", "--basename", action="store_true", help="force passage ID to be file basename")
    group = argparser.add_mutually_exclusive_group()
    add_verbose_arg(group, help="detailed evaluation output")
    group.add_argument("-q", "--quiet", action="store_true", help="do not print anything")
    main(argparser.parse_args())
