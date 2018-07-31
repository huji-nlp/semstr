#!/usr/bin/env python3

import csv
import os
import re
import sys
from itertools import groupby

import configargparse
from tqdm import tqdm
from ucca import evaluation, ioutil, constructions as ucca_constructions
from ucca.evaluation import LABELED, UNLABELED

from semstr.cfgutil import add_verbose_arg
from semstr.convert import CONVERTERS, UCCA_EXT
from semstr.evaluation import amr, sdp, conllu

desc = """Parses files in any format, and evaluates using the proper evaluator."""


EVALUATORS = {
    None: evaluation,
    "sdp": sdp,
    "conllu": conllu,
    "amr": amr,
}


class Scores:
    """
    Keeps score objects from multiple formats and/or languages
    """
    def __init__(self, scores):
        self.elements = [(t.aggregate(s), l) for (t, l), s in groupby(scores,
                                                                      lambda x: (type(x), getattr(x, "lang", None)))]
        element, _ = self.elements[0] if len(self.elements) == 1 else (None, None)
        self.name = element.name if element else "Multiple"
        self.format = element.format if element else None

    @staticmethod
    def aggregate(scores):
        return Scores([e for s in scores for e, _ in s.elements])

    def average_f1(self, *args, **kwargs):
        return sum(e.average_f1(*args, **kwargs) for e, _ in self.elements) / len(self.elements)

    def print(self, *args, **kwargs):
        for element, lang in self.elements:
            if len(self.elements):
                print(element.name + ((" (" + lang + ")") if lang else "") + ":", *args, **kwargs)
            element.print(*args, **kwargs)

    def fields(self, *args, **kwargs):
        return [f for e, _ in self.elements for f in e.fields(*args, **kwargs)]

    def titles(self, *args, **kwargs):
        return [(e.name + (("_" + l) if l else "") + "_" + f) for e, l in self.elements
                for f in e.titles(*args, **kwargs)]

    def details(self, average_f1):
        return "" if len(self.elements) < 2 else \
            " (" + ", ".join("%.3f" % average_f1(e) for e, _ in self.elements) + ")"

    def __str__(self):
        print(",".join(self.fields()))


class ConvertedPassage:
    def __init__(self, converted, original=None, passage_id=None,
                 converted_format=None, in_converter=None, out_converter=None):
        self.converted = converted
        self.passage = converted if original is None else original
        self.ID = converted.ID if passage_id is None else passage_id
        self.format = converted_format
        self.in_converter = in_converter
        self.out_converter = out_converter


def passage_format(filename):
    basename, ext = os.path.splitext(os.path.basename(filename))
    return basename, None if ext in UCCA_EXT else ext.lstrip(".")


def read_files(files, default_format=None, verbose=0, force_basename=False):
    for filename in sorted(files, key=lambda x: tuple(map(int, re.findall("\d+", x))) or x):
        basename, converted_format = passage_format(filename)
        if converted_format == "txt":
            converted_format = default_format
        in_converter, out_converter = CONVERTERS.get(converted_format, CONVERTERS[default_format])
        kwargs = dict(converted_format=converted_format, in_converter=in_converter, out_converter=out_converter)
        if in_converter:
            with open(filename, encoding="utf-8") as f:
                for converted, passage, passage_id in in_converter(f, passage_id=basename, return_original=True):
                    if verbose:
                        with tqdm.external_write_mode():
                            print("Converting %s from %s" % (filename, converted_format))
                    yield ConvertedPassage(converted, passage, basename if force_basename else passage_id, **kwargs)
        else:
            passage_id = basename if force_basename else None
            yield ConvertedPassage(ioutil.file2passage(filename), passage_id=passage_id, **kwargs)


def evaluate_all(evaluate, files, name=None, verbose=0, quiet=False, basename=False, matching_ids=False,
                 units=False, errors=False, unlabeled=False, normalize=True, constructions=None, **kwargs):
    guessed, ref = [iter(read_files(f, kwargs["format"], verbose=verbose, force_basename=basename)) for f in files]
    for (g, r) in tqdm(zip(guessed, ref), unit=" passages", desc=name, total=len(files[-1])):
        if matching_ids:
            while g.ID < r.ID:
                g = next(guessed)
            while g.ID > r.ID:
                r = next(ref)
        if not quiet:
            with tqdm.external_write_mode():
                print(r.ID, end=" ")
        if g.format != r.format:
            # noinspection PyCallingNonCallable
            g.passage = g.converted if r.out_converter is None else r.out_converter(g.converted)
        result = evaluate(g.passage, r.passage, verbose=verbose > 1 or units, units=units, errors=errors,
                          eval_type=UNLABELED if unlabeled else None, normalize=normalize,
                          constructions=constructions)
        if not quiet:
            with tqdm.external_write_mode():
                print("F1: %.3f" % result.average_f1(UNLABELED if unlabeled else LABELED))
        if verbose:
            with tqdm.external_write_mode():
                result.print()
        yield result


def write_csv(filename, rows):
    if filename:
        with sys.stdout if filename == "-" else open(filename, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)


def main(args):
    files = [[os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d) else [d] for d in (args.guessed, args.ref)]
    try:
        evaluate = EVALUATORS.get(passage_format(files[1][0])[1], EVALUATORS[args.format]).evaluate
    except IndexError as e:
        raise ValueError("No reference passages found: %s" % args.ref) from e
    results = list(evaluate_all(evaluate, files, **vars(args)))
    summary = Scores(results)
    eval_type = UNLABELED if args.unlabeled else LABELED
    if len(results) > 1:
        if args.verbose:
            print("Aggregated scores:")
        if not args.quiet:
            print("F1: %.3f" % summary.average_f1(eval_type))
            summarize(summary)
    elif not args.verbose:
        summarize(summary, errors=args.errors)
    write_csv(args.out_file,     [summary.titles(eval_type)] + [result.fields(eval_type) for result in results])
    write_csv(args.summary_file, [summary.titles(eval_type), summary.fields(eval_type)])


def summarize(scores, errors=False):
    scores.print()
    if errors:
        for element, _ in scores.elements:
            if hasattr(element, "print_confusion_matrix"):
                element.print_confusion_matrix()


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("guessed", help="filename/directory for the guessed annotation(s)")
    argparser.add_argument("ref", help="filename/directory for the reference annotation(s)")
    argparser.add_argument("-f", "--format", default="amr", help="default format (if cannot determine by suffix)")
    argparser.add_argument("-o", "--out-file", help="file to write results for each evaluated passage to in CSV format")
    argparser.add_argument("-s", "--summary-file", help="file to write aggregated results to, in CSV format")
    argparser.add_argument("-u", "--unlabeled", action="store_true", help="print unlabeled F1 for individual passages")
    argparser.add_argument("-N", "--no-normalize", dest="normalize", action="store_false",
                           help="do not normalize passages before evaluation")
    argparser.add_argument("-i", "--matching-ids", action="store_true", help="skip passages without a match (by ID)")
    argparser.add_argument("-b", "--basename", action="store_true", help="force passage ID to be file basename")
    argparser.add_argument("--units", action="store_true", help="print mutual and unique units")
    argparser.add_argument("--errors", action="store_true", help="print confusion matrix with error distribution")
    group = argparser.add_mutually_exclusive_group()
    add_verbose_arg(group, help="detailed evaluation output")
    group.add_argument("-q", "--quiet", action="store_true", help="do not print anything")
    ucca_constructions.add_argument(argparser)
    main(argparser.parse_args())
