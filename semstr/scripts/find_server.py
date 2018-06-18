#!/usr/bin/env python3

import os
import sqlite3

from flask import Flask, Response, request

desc = """Flask server to find terminals from a corpus by their dependency relation and category."""

DB_FILENAME = "find.db"

app = Flask(__name__)


@app.route("/", methods=["POST"])
def find():
    with sqlite3.connect(DB_FILENAME) as conn:
        c = conn.cursor()
        values = tuple(request.values.get(param) or None for param in ("pid", "text", "ftag", "dep"))
        sql = """SELECT * FROM terminals WHERE
                         (?1 IS NULL OR pid=?1) AND
                         (?2 IS NULL OR text=?2) AND
                         (?3 IS NULL OR ftag=?3) AND
                         (?4 IS NULL OR dep=?4)
                         %s""" % (" COLLATE NOCASE" if request.values.get("nocase") else "")
        print(sql, values)
        rows = c.execute(sql, values)
        rows = list(rows)
        print("%d rows" % len(rows))
        return Response("\n".join(map("\t".join, rows)), headers={"Content-Type": "text/plain"})


session_opts = {
    "session.type": "file",
    "session.cookie_expires": 60 * 24 * 60 * 2,  # two days in seconds
    "session.data_dir": "./data",
    "session.auto": True
}

if __name__ == "__main__":
    app.run(debug=True, host=os.getenv("IP", "0.0.0.0"), port=int(os.getenv("PORT", 5002)))
