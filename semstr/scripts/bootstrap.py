#!/usr/bin/env python3

import os
from glob import glob

import configargparse
import numpy as np
from tqdm import tqdm
from ucca import evaluation

from semstr.evaluate import EVALUATORS, passage_format, evaluate_all, Scores

desc = """Evaluates statistical significance of F1 scores between two systems."""


def main(args):
    files = [None if d is None else [os.path.join(d, f) for f in os.listdir(d) if not os.path.isdir(os.path.join(d, f))]
             if os.path.isdir(d) else [d] for p in args.guessed + [args.ref] for d in glob(p) or [p]]
    ref_files = files[-1]
    n = len(ref_files)
    evaluate = EVALUATORS.get(passage_format(ref_files[0])[1], EVALUATORS[args.format])
    results = [list(evaluate_all(evaluate, [f, ref_files, None], n, **vars(args))) for f, n in zip(files, args.guessed)]
    for evaluated, name in zip(results[1:], args.guessed[1:]):
        print(name)
        baseline = results[0]
        pair = (baseline, evaluated)
        d = diff(pair, verbose=True)
        sample = np.random.choice(n, (args.nboot, n))
        s = np.sum(np.sign(d) * diff(pair, indices) > 2 * np.abs(d) for indices in tqdm(sample, unit=" samples"))
        print("p-value:")
        print(s / args.nboot)
        print()


def diff(results, indices=None, verbose=False):
    scores = [Scores(r if indices is None else [r[i] for i in indices]) for r in results]
    fields = np.array([s.fields() for s in scores], dtype=float)
    if verbose:
        print(" ".join(evaluation.Scores.field_titles()))
        print("\n".join(map(str, fields)))
    return fields[1] - fields[0]


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("guessed", nargs="+", help="directories for the guessed annotations: baseline, evaluated")
    argparser.add_argument("ref", help="directory for the reference annotations")
    argparser.add_argument("-b", "--nboot", type=int, default=int(1e4), help="number of bootstrap samples")
    argparser.add_argument("-f", "--format", default="amr", help="default format (if cannot determine by suffix)")
    group = argparser.add_mutually_exclusive_group()
    main(argparser.parse_args())
