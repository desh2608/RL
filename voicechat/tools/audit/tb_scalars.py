#!/usr/bin/env python3
"""Minimal TensorBoard event reader (no tensorboard/protobuf dependency): prints scalar tags matching a regex.
Usage: tb_scalars.py EVENTS_FILE [REGEX]"""
import re
import struct
import sys


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return r, i


def fields(b):
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = varint(b, i)
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        elif w == 2:
            n, i = varint(b, i)
            v = b[i:i + n]
            i += n
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError("wire type %d" % w)
        yield f, w, v


def records(path):
    with open(path, "rb") as fh:
        while True:
            h = fh.read(12)
            if len(h) < 12:
                return
            n = struct.unpack("<Q", h[:8])[0]
            data = fh.read(n)
            fh.read(4)
            yield data


def main():
    path = sys.argv[1]
    pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")
    out = {}
    for rec in records(path):
        step, summ = None, None
        for f, w, v in fields(rec):
            if f == 2 and w == 0:
                step = v
            elif f == 5 and w == 2:
                summ = v
        if summ is None:
            continue
        for f, w, v in fields(summ):
            if f != 1 or w != 2:
                continue
            tag, val = None, None
            for f2, w2, v2 in fields(v):
                if f2 == 1 and w2 == 2:
                    tag = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 5:
                    val = struct.unpack("<f", v2)[0]
            if tag and val is not None and step is not None and pat.search(tag):
                out.setdefault(tag, []).append((step, val))
    for tag in sorted(out):
        vals = sorted(out[tag])
        print("%-60s n=%-3d %s" % (tag, len(vals), " ".join("%g" % round(v, 4) for _, v in vals)))


if __name__ == "__main__":
    main()
