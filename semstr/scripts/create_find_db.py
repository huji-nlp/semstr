#!/usr/bin/env python3

import argparse
import os
import sqlite3

from tqdm import tqdm
from ucca import layer0
from ucca.textutil import annotate_all, Attr

from semstr.cfgutil import read_specs, add_specs_args
from semstr.convert import FROM_FORMAT
from semstr.scripts.udpipe import annotate_udpipe

desc = """Create an index containing all terminals from a corpus, with their dependency relation and category."""


def main(args):
    for passages, out_dir, lang, udpipe in read_specs(args, converters=FROM_FORMAT):
        passages = annotate_udpipe(passages, udpipe) if udpipe else \
            annotate_all(passages, as_array=True, replace=not udpipe, lang=lang)
        filename = os.path.join(out_dir, "find.db")
        with sqlite3.connect(filename) as conn:
            c = conn.cursor()
            c.execute("DROP TABLE terminals")
            c.execute("CREATE TABLE terminals (pid, tid, text, ftag, fparent, dep)")
            c.execute("CREATE INDEX idx_terminals_pid ON terminals (pid)")
            c.execute("CREATE INDEX idx_terminals_text ON terminals (text)")
            c.execute("CREATE INDEX idx_terminals_ftag ON terminals (ftag)")
            c.execute("CREATE INDEX idx_terminals_dep ON terminals (dep)")
            for passage in tqdm(passages, unit=" passages", desc="Creating " + filename):
                rows = []
                for terminal in passage.layer(layer0.LAYER_ID).all:
                    parent = terminal.parents[0]
                    rows.append((passage.ID, terminal.ID, terminal.text, parent.ftag, str(parent.fparent),
                                 get_annotation(terminal, udpipe)))
                c.executemany("INSERT INTO terminals VALUES (?,?,?,?,?,?)", rows)
                conn.commit()


def get_annotation(terminal, udpipe=False):
    return terminal.tok[Attr.DEP.value] if udpipe else terminal.get_annotation(Attr.DEP, as_array=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=desc)
    add_specs_args(argparser)
    main(argparser.parse_args())
