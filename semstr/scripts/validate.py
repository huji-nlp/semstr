import argparse
import sys
from itertools import islice

from semstr.convert import iter_passages
from semstr.validation import validate, print_errors


def main(args):
    errors = ((p.ID, list(validate(p, normalization=args.normalize, extra_normalization=args.extra_normalization,
                                   ucca_validation=args.ucca_validation, output_format=args.format)))
              for p in iter_passages(args.filenames, desc="Validating"))
    errors = dict(islice(((k, v) for k, v in errors if v), 1 if args.strict else None))
    if errors:
        id_len = max(map(len, errors))
        for passage_id, es in sorted(errors.items()):
            print_errors(es, passage_id, id_len)
        sys.exit(1)
    else:
        print("No errors found.")


def check_args(parser, args):
    if args.extra_normalization and not args.normalize:
        parser.error("Cannot specify --extra-normalization without --normalize")
    return args


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Validate UCCA passages")
    argparser.add_argument("filenames", nargs="+", help="files or directories to validate")
    argparser.add_argument("-f", "--format", help="default format (if cannot determine by suffix)")
    argparser.add_argument("-s", "--split", action="store_true", help="split each sentence to its own passage")
    argparser.add_argument("-S", "--strict", action="store_true", help="fail as soon as a violation is found")
    argparser.add_argument("-u", "--ucca-validation", action="store_true", help="apply UCCA-specific validations")
    argparser.add_argument("-n", "--normalize", action="store_true", help="normalize passages before validation")
    argparser.add_argument("-e", "--extra-normalization", action="store_true", help="more normalization rules")
    main(check_args(argparser, argparser.parse_args()))
