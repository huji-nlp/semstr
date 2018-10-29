#!/usr/bin/env python3

import sys

import configargparse
import os
from glob import glob
from tqdm import tqdm
from ucca import ioutil
from ucca.normalization import normalize

from semstr.cfgutil import add_verbose_arg, add_boolean_option
from semstr.convert import CONVERTERS
from semstr.evaluate import EVALUATORS, Scores

desc = """Convert files to UCCA standard format, convert back to the original format and evaluate.
"""


def main(args):
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    scores = []
    for pattern in args.filenames:
        for filename in sorted(glob(pattern)) or [pattern]:
            file_scores = []
            basename, ext = os.path.splitext(os.path.basename(filename))
            passage_format = ext.lstrip(".")
            if passage_format == "txt":
                passage_format = args.format
            in_converter, out_converter = CONVERTERS.get(passage_format, CONVERTERS[args.format])
            evaluate = EVALUATORS.get(passage_format, EVALUATORS[args.format])
            with open(filename, encoding="utf-8") as f:
                t = tqdm(in_converter(f, passage_id=basename, return_original=True), unit=" passages",
                         desc=("Converting '%s'" % filename) +
                              ((", writing to '%s'" % args.out_dir) if args.out_dir else ""))
                for passage, ref, passage_id in t:
                    if args.normalize:
                        normalize(passage, extra=args.extra_normalization)
                    if args.out_dir:
                        os.makedirs(args.out_dir, exist_ok=True)
                        outfile = os.path.join(args.out_dir, passage.ID + ".xml")
                        if args.verbose:
                            with ioutil.external_write_mode():
                                print("Writing '%s'..." % outfile, file=sys.stderr, flush=True)
                        ioutil.passage2file(passage, outfile)
                    try:
                        guessed = out_converter(passage, wikification=args.wikification, use_original=False)
                    except Exception as e:
                        raise ValueError("Error converting %s back from %s" % (filename, passage_format)) from e
                    if args.out_dir:
                        outfile = os.path.join(args.out_dir, passage.ID + ext)
                        if args.verbose:
                            with ioutil.external_write_mode():
                                print("Writing '%s'..." % outfile, file=sys.stderr, flush=True)
                        with open(outfile, "w", encoding="utf-8") as f_out:
                            print("\n".join(guessed), file=f_out)
                    try:
                        s = evaluate(guessed, ref, verbose=args.verbose > 1, units=args.units)
                    except Exception as e:
                        raise ValueError("Error evaluating conversion of %s" % filename) from e
                    file_scores.append(s)
                    if args.verbose:
                        with ioutil.external_write_mode():
                            print(passage_id)
                            s.print()
                    t.set_postfix(F1="%.2f" % (100.0 * Scores(file_scores).average_f1()))
            scores += file_scores
    print()
    if args.verbose and len(scores) > 1:
        print("Aggregated scores:")
    Scores(scores).print()


def check_args(parser, args):
    if args.extra_normalization and not args.normalize:
        parser.error("Cannot specify --extra-normalization without --normalize")
    return args


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="file names to convert and evaluate")
    argparser.add_argument("-f", "--format", choices=CONVERTERS, default="amr",
                           help="default format (if cannot determine by suffix)")
    add_verbose_arg(argparser, help="detailed evaluation output")
    argparser.add_argument("--units", action="store_true", help="print mutual and unique units")
    add_boolean_option(argparser, "wikification", "Spotlight to wikify any named node (for AMR)")
    argparser.add_argument("-o", "--out-dir", help="output directory (if unspecified, files are not written)")
    argparser.add_argument("-n", "--normalize", action="store_true", help="normalize passages before conversion")
    argparser.add_argument("-e", "--extra-normalization", action="store_true", help="more normalization rules")
    main(check_args(argparser, argparser.parse_args()))
