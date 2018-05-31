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


def read_specs(args, converters=None):
    specs = [(pattern, args.out_dir, args.lang, args.udpipe) for pattern in args.filenames]
    if args.list_file:
        with open(args.list_file, encoding="utf-8") as f:
            specs += [l.strip().split() for l in f if not l.startswith("#")]
    for spec in specs:
        pattern = spec[0]
        out_dir = spec[1] if len(spec) > 1 else args.out_dir
        lang = spec[2] if len(spec) > 2 else args.lang
        udpipe = spec[3] if len(spec) > 3 else args.udpipe
        os.makedirs(out_dir, exist_ok=True)
        filenames = glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        yield read_files_and_dirs(filenames, converters=converters), out_dir, lang, udpipe


def add_specs_args(p):
    p.add_argument("filenames", nargs="*", help="passage file names to annotate")
    p.add_argument("-f", "--list-file", help="file whose rows are <PATTERN> <OUT-DIR> <LANGUAGE>")
    p.add_argument("-o", "--out-dir", default=".", help="directory to write annotated files to")
    p.add_argument("-l", "--lang", default="en", help="small two-letter language code to use for spaCy model")
    p.add_argument("-u", "--udpipe", help="use specified UDPipe model, not spaCy, for syntactic annotation")
    p.add_argument("-b", "--binary", action="store_true", help="write in binary format (.pickle)")
