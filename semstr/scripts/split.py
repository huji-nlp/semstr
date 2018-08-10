#!/usr/bin/env python3
import os
import re
from glob import glob

import configargparse

desc = """Split sentences/passages to separate files (important for shuffling before training the parser)"""

ID_PATTERN = re.compile("#\s*::id\s+(\S+)")
COMMENT_PREFIX = "#"


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    lines = []
    passage_id = 0
    doc_id = None
    for pattern in args.filenames:
        for filename in glob(pattern) or [pattern]:
            _, ext = os.path.splitext(filename)
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    clean = line.lstrip()
                    m_id = ID_PATTERN.match(clean) or \
                        re.match("#\s*(\d+).*", line) or re.match("#\s*sent_id\s*=\s*(\S+)", line)
                    m_docid = re.match("#\s*(?:new)doc[ _]id\s*=\s*(\S+)", line)
                    if m_id or m_docid or not clean or clean[0] != COMMENT_PREFIX or re.match("#\s*::", clean):
                        lines.append(line)
                        if m_docid:
                            doc_id = m_docid.group(1)
                            passage_id = 1
                        if m_id:
                            passage_id = m_id.group(1)
                    if not clean and any(map(str.strip, lines)):
                        if not args.doc_ids or doc_id in args.doc_ids:
                            write_file(args.out_dir, doc_id, passage_id, ext, lines, quiet=args.quiet)
                        lines.clear()
                        if isinstance(passage_id, str):
                            passage_id = None
                        else:
                            passage_id += 1
                if lines and (not args.doc_ids or doc_id in args.doc_ids):
                    write_file(args.out_dir, doc_id, passage_id, ext, lines, quiet=args.quiet)
    if not args.quiet:
        print()


def write_file(out_dir, doc_id, passage_id, ext, lines, quiet=False):
    if passage_id is None:
        raise ValueError("Could not determine passage ID")
    filename = os.path.join(out_dir, ("" if doc_id is None else (doc_id + ".")) + str(passage_id) + ext)
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)
    if not quiet:
        print("\rWrote %-70s" % filename, end="", flush=True)


if __name__ == '__main__':
    argparser = configargparse.ArgParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="file name(s) to split")
    argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
    argparser.add_argument("-q", "--quiet", action="store_true", help="less output")
    argparser.add_argument("--doc-ids", nargs="+", help="document IDs to keep from the input file (by '# doc_id')")
    main(argparser.parse_args())
