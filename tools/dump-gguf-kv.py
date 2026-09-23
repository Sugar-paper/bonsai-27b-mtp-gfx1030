# -*- coding: utf-8 -*-
"""Minimal GGUF KV reader - dump tokenizer.chat_template (and friends) without gguf pkg."""
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
       7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def read_value(f, t):
    if t == 8:
        return read_string(f)
    if t == 9:  # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    return struct.unpack(FMT[t], f.read(SIZES[t]))[0]


def main(path, keys):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a gguf"
        ver, = struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        print(f"# gguf v{ver}, tensors={n_tensors}, kv={n_kv}", file=sys.stderr)
        for _ in range(n_kv):
            k = read_string(f)
            t = struct.unpack("<I", f.read(4))[0]
            v = read_value(f, t)
            if not keys or any(s in k for s in keys):
                if isinstance(v, str) and len(v) > 8000:
                    print(f"### {k} (len={len(v)}) -> {path}.{k.split('.')[-1]}.txt")
                    with open(f"{path}.{k.split('.')[-1]}.txt", "w", encoding="utf-8") as o:
                        o.write(v)
                else:
                    print(f"### {k} = {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
