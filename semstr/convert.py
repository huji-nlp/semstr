#!/usr/bin/env python3

import csv
import os
import re
import sys
from glob import glob

import configargparse
from tqdm import tqdm
from ucca import ioutil, layer1
from ucca.convert import from_text, to_text, from_json, to_json, split2sentences
from ucca.normalization import normalize

from semstr.cfgutil import add_verbose_arg
from semstr.validation import validate, print_errors

description = """Parses files in the specified format, and writes as the specified format.
Each passage is written to the file: <outdir>/<prefix><passage_id>.<extension> """


def from_conll(lines, passage_id, split=True, *args, **kwargs):
    """Converts from parsed text in CoNLL format to a Passage object.

    :param lines: iterable of lines in CoNLL format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?

    :return generator of Passage objects
    """
    del args, kwargs
    from semstr.conversion.conll import ConllConverter
    return ConllConverter().from_format(lines, passage_id, split)


def to_conll(passage, test=False, tree=True, *args, **kwargs):
    """ Convert from a Passage object to a string in CoNLL-X format (conll)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to True

    :return list of lines representing the dependencies in the passage
    """
    del args, kwargs
    from semstr.conversion.conll import ConllConverter
    return ConllConverter().to_format(passage, test, tree)


def from_export(lines, passage_id=None, split=True, *args, **kwargs):
    """Converts from parsed text in NeGra export format to a Passage object.

    :param lines: iterable of lines in NeGra export format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param split: split each sentence to its own passage?

    :return generator of Passage objects
    """
    del args, kwargs
    from semstr.conversion.export import ExportConverter
    return ExportConverter().from_format(lines, passage_id, split)


def to_export(passage, test=False, tree=False, *args, **kwargs):
    """ Convert from a Passage object to a string in NeGra export format (export)

    :param passage: the Passage object to convert
    :param test: whether to omit the edge and parent columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False

    :return list of lines representing a (discontinuous) tree structure constructed from the passage
    """
    del args, kwargs
    from semstr.conversion.export import ExportConverter
    return ExportConverter().to_format(passage, test, tree)


def from_amr(lines, passage_id=None, return_original=False, save_original=True, *args, **kwargs):
    """Converts from parsed text in AMR PENMAN format to a Passage object.

    :param lines: iterable of lines in AMR PENMAN format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param save_original: whether to save original AMR text in passage.extra
    :param return_original: return triple of (UCCA passage, AMR string, AMR ID)

    :return generator of Passage objects
    """
    del args, kwargs
    from semstr.conversion.amr import AmrConverter
    return AmrConverter().from_format(lines, passage_id, return_original=return_original, save_original=save_original)


def to_amr(passage, metadata=True, wikification=True, use_original=True, verbose=False, default_label=None,
           *args, **kwargs):
    """ Convert from a Passage object to a string in AMR PENMAN format (export)

    :param passage: the Passage object to convert
    :param metadata: whether to print ::id and ::tok lines
    :param wikification: whether to wikify named concepts, adding a :wiki triple
    :param use_original: whether to use original AMR text from passage.extra
    :param verbose: whether to print extra information
    :param default_label: label to use in case node has no label attribute

    :return list of lines representing an AMR in PENMAN format, constructed from the passage
    """
    del args, kwargs
    from semstr.conversion.amr import AmrConverter
    return AmrConverter().to_format(passage, metadata, wikification, verbose, use_original=use_original,
                                    default_label=default_label)


def from_conllu(lines, passage_id=None, split=True, return_original=False, annotate=False, *args, **kwargs):
    """Converts from parsed text in Universal Dependencies format to a Passage object.

    :param lines: iterable of lines in Universal Dependencies format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?
    :param return_original: return triple of (UCCA passage, Universal Dependencies string, sentence ID)
    :param annotate: whether to save dependency annotations in "extra" dict of layer 0

    :return generator of Passage objects
    """
    del args, kwargs
    from semstr.conversion.conllu import ConlluConverter
    return ConlluConverter().from_format(lines, passage_id, split, return_original=return_original, annotate=annotate)


def to_conllu(passage, test=False, *args, **kwargs):
    """ Convert from a Passage object to a string in Universal Dependencies format (conllu)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    from semstr.conversion.conllu import ConlluConverter
    return ConlluConverter().to_format(passage, test, tree=True)


def from_sdp(lines, passage_id, split=True, mark_aux=False, return_original=False, *args, **kwargs):
    """Converts from parsed text in SemEval 2015 SDP format to a Passage object.

    :param lines: iterable of lines in SDP format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?
    :param mark_aux: add a preceding # for labels of auxiliary edges added
    :param return_original: return triple of (UCCA passage, SDP string, sentence ID)

    :return generator of Passage objects
    """
    del args, kwargs
    from semstr.conversion.sdp import SdpConverter
    return SdpConverter(mark_aux=mark_aux).from_format(lines, passage_id, split, return_original=return_original)


def to_sdp(passage, test=False, tree=False, mark_aux=False, *args, **kwargs):
    """ Convert from a Passage object to a string in SemEval 2015 SDP format (sdp)

    :param passage: the Passage object to convert
    :param test: whether to omit the top, head, frame, etc. columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False
    :param mark_aux: omit edges with labels with a preceding #

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    from semstr.conversion.sdp import SdpConverter
    return SdpConverter(mark_aux=mark_aux).to_format(passage, test, tree)


CONVERTERS = {
    None: (None, None),
    "json": (from_json, to_json),
    "conll": (from_conll, to_conll),
    "conllu": (from_conllu, to_conllu),
    "sdp": (from_sdp, to_sdp),
    "export": (from_export, to_export),
    "amr": (from_amr, to_amr),
    "txt": (from_text, to_text),
}
FROM_FORMAT = {f: c[0] for f, c in CONVERTERS.items() if c[0] is not None}
TO_FORMAT = {f: c[1] for f, c in CONVERTERS.items() if c[1] is not None}

UCCA_EXT = (".xml", ".pickle")


def iter_files(patterns):
    for pattern in patterns:
        filenames = glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        yield from filenames


def iter_passages(patterns, desc=None, input_format=None, prefix="", split=False, mark_aux=False, annotate=False):
    t = tqdm(list(iter_files(patterns)), unit="file", desc=desc)
    for filename in t:
        t.set_postfix(file=filename)
        if not os.path.isfile(filename):
            raise IOError("Not a file: %s" % filename)
        no_ext, ext = os.path.splitext(filename)
        if ext in UCCA_EXT:  # UCCA input
            yield ioutil.file2passage(filename)
        else:
            basename = os.path.basename(no_ext)
            try:
                passage_id = re.search(r"\d+(\.\d+)*", basename).group(0)
            except AttributeError:
                passage_id = basename
            converter, _ = CONVERTERS.get(input_format or ext.lstrip("."), (from_text,))
            with open(filename, encoding="utf-8") as f:
                yield from converter(f, prefix + passage_id, split=split, mark_aux=mark_aux, annotate=annotate)


def map_labels(passage, label_map_file):
    if label_map_file:
        with open(label_map_file, encoding="utf-8") as f:
            label_map = dict(csv.reader(f))
        for node in passage.layer(layer1.LAYER_ID).all:
            for edge in node:
                mapped = label_map.get(edge.tag) or label_map.get(edge.tag.partition(":")[0])
                if mapped is not None:
                    edge.tag = mapped
        try:
            del passage.extra["format"]  # Remove original format as it no longer applies, after labels were replaced
        except KeyError:
            pass


def write_passage(passage, args):
    ext = {None: UCCA_EXT[args.binary], "amr": ".txt"}.get(args.output_format) or "." + args.output_format
    outfile = args.out_dir + os.path.sep + passage.ID + ext
    if args.verbose:
        with tqdm.external_write_mode():
            print("Writing '%s'..." % outfile, file=sys.stderr)
    if args.output_format is None:  # UCCA output
        ioutil.passage2file(passage, outfile, binary=args.binary)
    else:
        converter = CONVERTERS[args.output_format][1]
        output = "\n".join(converter(passage, wikification=args.wikification, default_label=args.default_label)) \
            if args.output_format == "amr" else \
            "\n".join(line for p in (split2sentences(passage) if args.split else [passage]) for line in
                      converter(p, test=args.test, tree=args.tree, mark_aux=args.mark_aux))
        with open(outfile, "w", encoding="utf-8") as f:
            print(output, file=f)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for passage in iter_passages(args.filenames, desc="Converting", input_format=args.input_format, prefix=args.prefix,
                                 split=args.split, mark_aux=args.mark_aux, annotate=args.annotate):
        map_labels(passage, args.label_map)
        if args.normalize:
            normalize(passage, extra=args.extra_normalization)
        if args.lang:
            passage.attrib["lang"] = args.lang
        write_passage(passage, args)
        if args.validate:
            errors = list(validate(passage, ucca_validation=args.ucca_validation, output_format=args.output_format))
            if errors:
                print_errors(errors, passage.ID)
                sys.exit(1)


def add_convert_args(p):
    p.add_argument("-t", "--test", action="store_true",
                   help="omit prediction columns (head and deprel for conll; top, pred, frame, etc. for sdp)")
    p.add_argument("-T", "--tree", action="store_true", help="remove multiple parents to get a tree")
    p.add_argument("-s", "--split", action="store_true", help="split each sentence to its own passage")
    p.add_argument("-m", "--mark-aux", action="store_true", help="mark auxiliary edges introduced/omit edges")
    p.add_argument("--label-map", help="CSV file specifying mapping of input edge labels to output edge labels")


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=description)
    argparser.add_argument("filenames", nargs="+", help="file names to convert")
    argparser.add_argument("-i", "--input-format", choices=CONVERTERS, help="input file format (detected by extension)")
    argparser.add_argument("-f", "--output-format", choices=CONVERTERS, help="output file format (default: UCCA)")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output passage ID prefix")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in binary format (.%s)" % UCCA_EXT[1])
    argparser.add_argument("-a", "--annotate", action="store_true", help="store dependency annotations in 'extra' dict")
    argparser.add_argument("-V", "--validate", action="store_true", help="validate every passage after conversion")
    argparser.add_argument("-u", "--ucca-validation", action="store_true", help="apply UCCA-specific validations")
    argparser.add_argument("--no-wikification", action="store_false", dest="wikification", help="no AMR wikification")
    argparser.add_argument("--default-label", help="use this for missing AMR labels, otherwise raise exception")
    group = argparser.add_mutually_exclusive_group()
    group.add_argument("--no-normalize", action="store_false", dest="normalize", help="do not normalize passage")
    group.add_argument("-e", "--extra-normalization", action="store_true", help="more normalization rules")
    argparser.add_argument("-l", "--lang", help="small two-letter language code to set in output passage metadata")
    add_convert_args(argparser)
    add_verbose_arg(argparser, help="detailed output")
    main(argparser.parse_args())
    sys.exit(0)
