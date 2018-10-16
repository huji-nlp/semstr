#!/usr/bin/python3
from itertools import groupby

import argparse
from operator import methodcaller


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("headers_file", help="file whose headers should be taken")
    argparser.add_argument("content_file", help="file whose content should be taken")
    argparser.add_argument("output_file", nargs="?", help="file to write to")
    args = argparser.parse_args()

    with open(args.headers_file, encoding="utf-8") as hf, open(args.content_file, encoding="utf-8") as cf:
        hf_lines = read_lines(hf)
        cf_lines = read_lines(cf)

    file = open(args.output_file, "w", encoding="utf-8") if args.output_file else None
    headers = iter([list(l) for k, l in groupby(hf_lines, key=methodcaller("startswith", "#")) if k] + [[""]])
    print_headers(file, headers)
    for prev_line, line in zip([""] + cf_lines, cf_lines):
        if line:
            print(line, file=file)
        elif prev_line and headers:
            print(file=file)
            print_headers(file, headers)

    if file is not None:
        file.close()


def read_lines(hf):
    lines = list(map(str.strip, hf))
    return lines[next(i for i, line in enumerate(lines) if line):]  # Skip empty lines


def print_headers(file, headers):
    for header in next(headers):
        print(header, file=file)


if __name__ == '__main__':
    main()
