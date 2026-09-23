# -*- coding: utf-8 -*-
"""Dump the tensor table of a GGUF file (name / type / shape / offset / bytes).

usage: dump-gguf-tensors.py <model.gguf> [out.csv] [--txt]
Pure stdlib, no gguf package required.
"""
import struct
import sys

# ggml type ids -> names (only the ones this project actually meets, plus common ones)
TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    40: "Q4_0_4_4", 41: "Q4_0_4_8", 42: "Q4_0_8_8",
    43: "TURBO3_0", 44: "TURBO4_0", 45: "TQ3_1S", 46: "TQ4_1S",
    142: "PQ2_0", 143: "PTQ1_0",
}
# block size / bytes-per-block for the types whose byte span we report
BLOCK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 40), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 30: (1, 2),
}


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def read_value(f, t):
    if t == 8:
        return read_string(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            read_value(f, et)
        return None
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t]
    return f.read(size)


def main(path, out=None, as_txt=False):
    rows = []
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", "not a gguf file"
        ver, = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            k = read_string(f)
            t = struct.unpack("<I", f.read(4))[0]
            read_value(f, t)
        for _ in range(n_tensors):
            name = read_string(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack("<" + "Q" * nd, f.read(8 * nd)))
            ttype = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            n = 1
            for d in dims:
                n *= d
            bs = BLOCK.get(ttype)
            nbytes = (n // bs[0]) * bs[1] if bs and n % bs[0] == 0 else ""
            rows.append((name, TYPE_NAMES.get(ttype, f"type_{ttype}"), ttype, nd,
                         "x".join(str(d) for d in dims), off, nbytes))

    header = "name,type,type_id,n_dims,dims,offset,bytes\n"
    lines = [",".join(str(x) for x in r) for r in rows]
    if as_txt:
        w = max(len(r[0]) for r in rows) + 2
        body = f"{'name':<{w}}{'type':<12}{'shape':<28}{'offset':>12}\n"
        body += "-" * (w + 52) + "\n"
        for r in rows:
            body += f"{r[0]:<{w}}{r[1]:<12}{r[4]:<28}{r[5]:>12}\n"
        text = body
    else:
        text = header + "\n".join(lines) + "\n"

    if out:
        with open(out, "w", encoding="utf-8", newline="\n") as o:
            o.write(text)
        print(f"{out}: {len(rows)} tensors")
    else:
        print(text)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    txt = "--txt" in args
    args = [a for a in args if a != "--txt"]
    main(args[0], args[1] if len(args) > 1 else None, txt)
