import os

# bežné kódovania, ktoré môžu byť zdrojom problémov
encodings = [
    "utf-8",
    "cp1250",
    "cp1251",
    "cp1252",
    "iso-8859-2",
    "latin1",
    "cp852"
]

# tu nastav cestu, kde chceš hľadať
root_dir = "../Knowledge"  # alebo napr. "."

for root, dirs, files in os.walk(root_dir):
    for name in dirs + files:
        # vezmeme raw bajty názvu
        name_bytes = name.encode("utf-8", errors="surrogateescape")
        print(f"\n=== Originál na disku ===")
        print(name)

        for enc in encodings:
            try:
                decoded = name_bytes.decode(enc)
                print(f"{enc:10} → {decoded}")
            except UnicodeDecodeError:
                print(f"{enc:10} → [neplatné bajty pre toto kódovanie]")
