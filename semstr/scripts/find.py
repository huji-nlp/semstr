#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm
from ucca import layer0
from ucca.textutil import annotate_all, Attr

from semstr.cfgutil import read_specs, add_specs_args
from semstr.convert import FROM_FORMAT
from semstr.scripts.udpipe import annotate_udpipe

desc = """Find all instances of a certain word or dependency relation."""


def main(args):
    words = args.word or []
    categories = list(args.category or ())
    dependencies = list(args.dependency or ())
    if args.case_insensitive:
        words = list(map(str.lower, words))
    for spec in read_specs(args, converters=FROM_FORMAT):
        if args.dependency:
            spec.passages = annotate_udpipe(spec.passages, spec.udpipe) if spec.udpipe else \
                annotate_all(spec.passages, as_array=True, replace=not spec.udpipe, lang=spec.lang)
        t = tqdm(spec.passages, unit=" passages", desc="Finding")
        if words:
            t.set_postfix(words=",".join(words))
        if categories:
            t.set_postfix(categories=",".join(categories))
        if dependencies:
            t.set_postfix(dependencies=",".join(dependencies))
        found = 0
        filename = os.path.join(spec.out_dir, "_".join(words + categories + dependencies) + ".txt")
        with open(filename, "w", encoding="utf-8") as f:
            for passage in t:
                for terminal in passage.layer(layer0.LAYER_ID).all:
                    parent = terminal.parents[0]
                    word = terminal.text
                    if args.case_insensitive:
                        word = word.lower()
                    if (not words or word in words) and (
                            not categories or parent.ftag in categories) and (
                            not dependencies or get_annotation(terminal, spec.udpipe) in dependencies):
                        print(passage.ID, parent.fparent, file=f)
                        found += 1
                        t.set_postfix(found=found)
        print("Wrote '%s'" % filename)


def get_annotation(terminal, udpipe=False):
    return terminal.tok[Attr.DEP.value] if udpipe else terminal.get_annotation(Attr.DEP, as_array=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("-w", "--word", nargs="+", help="Word(s) to find")
    argparser.add_argument("-i", "--case-insensitive", action="store_true", help="Ignore case when looking up words")
    argparser.add_argument("-e", "--category", nargs="+", help="Incoming edge categories to find")
    argparser.add_argument("-d", "--dependency", nargs="+", help="Dependency relation(s) to find dependents of")
    main(argparser.parse_args())
