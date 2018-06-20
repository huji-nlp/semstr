import os
from glob import glob

import configargparse
from ucca.ioutil import read_files_and_dirs


class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance

    def reload(cls):
        cls.instance = None


class VAction(configargparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if values is None:
            values = "1"
        try:
            values = int(values)
        except ValueError:
            values = values.count("v") + 1
        setattr(args, self.dest, values)


def add_verbose_arg(argparser, **kwargs):
    return argparser.add_argument("-v", "--verbose", nargs="?", action=VAction, default=0, **kwargs)


def get_group_arg_names(group):
    return [a.dest for a in group._group_actions]


def add_boolean_option(argparser, name, description, default=False, short=None, short_no=None):
    group = argparser.add_mutually_exclusive_group()
    options = [] if short is None else ["-" + short]
    options.append("--" + name)
    group.add_argument(*options, action="store_true", default=default, help="include " + description)
    no_options = [] if short_no is None else ["-" + short_no]
    no_options.append("--no-" + name)
    group.add_argument(*no_options, action="store_false", dest=name.replace("-", "_"), default=default,
                       help="exclude " + description)
    return group


class AnnotationSpecification:
    def __init__(self, passages, out_dir, lang, udpipe=None, conllu=None):
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.passages = passages
        self.out_dir = out_dir
        self.lang = lang
        if udpipe and conllu:
            raise ValueError("Cannot specify both UDPipe model and CoNLL-U files to get annotation from")
        self.udpipe = udpipe
        self.conllu = conllu


def read_specs(args, converters=None):
    specs = [(pattern, args.out_dir, args.lang, args.udpipe, args.conllu) for pattern in args.filenames]
    if args.list_file:
        with open(args.list_file, encoding="utf-8") as f:
            specs += [l.strip().split() for l in f if not l.startswith("#")]
    for spec in specs:
        pattern = spec[0]
        filenames = glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        yield AnnotationSpecification(passages=read_files_and_dirs(filenames, converters=converters),
                                      out_dir=spec[1] if len(spec) > 1 else args.out_dir,
                                      lang=spec[2] if len(spec) > 2 else args.lang,
                                      udpipe=spec[3] if len(spec) > 3 else args.udpipe,
                                      conllu=spec[4] if len(spec) > 4 else args.conllu)


def add_specs_args(p):
    p.add_argument("filenames", nargs="*", help="passage file names to annotate")
    p.add_argument("-f", "--list-file", help="file whose rows are <PATTERN> <OUT-DIR> <LANGUAGE>")
    p.add_argument("-o", "--out-dir", default=".", help="directory to write annotated files to")
    p.add_argument("-l", "--lang", default="en", help="small two-letter language code to use for spaCy model")
    group = p.add_mutually_exclusive_group()
    group.add_argument("-u", "--udpipe", help="use specified UDPipe model, not spaCy, for syntactic annotation")
    group.add_argument("-c", "--conllu", help="copy syntactic annotation from specified CoNLL-U files instead of spaCy")
    p.add_argument("-b", "--binary", action="store_true", help="write in binary format (.pickle)")
