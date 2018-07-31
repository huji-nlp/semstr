#!/usr/bin/env python3
import os

import configargparse
from tqdm import tqdm
from ucca.textutil import read_word_vectors

desc = """Filter a word embedding file to contain only words found in a given corpus"""


def main(args):
    with open(args.corpus, encoding="utf-8") as f:
        words = set(word for line in tqdm(f, desc="Reading '%s'" % args.corpus, unit=" lines") for word in line.split())
    it = read_word_vectors(None, None, args.filename)
    nr_row, nr_dim = next(it)
    found = 0
    with open(args.filename + args.suffix + "~", "w", encoding="utf-8") as f:
        for w, v in tqdm(it, desc="Filtering '%s'" % args.filename, unit=" lines"):
            if w in words:
                found += 1
                print(" ".join(map(str, [w] + list(v))), file=f)
    with open(args.filename + args.suffix + "~", encoding="utf-8") as f, \
            open(args.filename + args.suffix, "w", encoding="utf-8") as g:
        print(found, nr_dim, file=g)
        for line in f:
            print(line, file=g)
    os.remove(args.filename + args.suffix + "~")


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("filename", help="word vectors file to filter")
    argparser.add_argument("corpus", help="tokenized corpus to look up words in")
    argparser.add_argument("-s", "--suffix", default=".filtered", help="suffix to append to given input file")
    main(argparser.parse_args())
