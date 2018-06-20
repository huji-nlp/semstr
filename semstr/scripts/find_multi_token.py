#!/usr/bin/env python3

import argparse
import os

from tqdm import tqdm
from ucca import layer1

from semstr.cfgutil import read_specs, add_specs_args
from semstr.convert import FROM_FORMAT

desc = """Find all instances of a certain category with more than one terminal."""


def main(args):
    categories = "".join(args.category)
    for spec in read_specs(args, converters=FROM_FORMAT):
        t = tqdm(spec.passages, unit=" passages", desc="Finding multi-token units", postfix={"categories": categories})
        found = 0
        filename = os.path.join(spec.out_dir, "multi_token_" + categories + ".txt")
        with open(filename, "w", encoding="utf-8") as f:
            for passage in t:
                for node in passage.layer(layer1.LAYER_ID).all:
                    try:
                        if node.ftag in categories and len(node.get_terminals(remotes=True)) > 1:
                            print(passage.ID, node.ID, node, file=f)
                            found += 1
                            t.set_postfix(found=found)
                    except (AttributeError, KeyError, ValueError, TypeError):
                        pass
        print("Wrote '%s'" % filename)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    argparser.add_argument("-e", "--category", nargs="+", default=["P", "S"], help="Incoming edge categories to find")
    main(argparser.parse_args())
