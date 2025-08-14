#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from datetime import datetime

def scan_directory(root_dir: Path):
    """
    Prejde všetky súbory pod adresárom root_dir a yieldne dict s metadátami.
    """
    for file in root_dir.rglob("*.*"):
        try:
            stat = file.stat()
            yield {
                "path": str(file),
                "suffix": file.suffix.lower(),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(sep=' ', timespec='seconds'),
            }
        except Exception as e:
            # v prípade chyby pri čítaní metadát, zalogujeme cestu a pokračujeme
            yield {
                "path": str(file),
                "suffix": file.suffix.lower(),
                "size_bytes": None,
                "modified": None,
                "error": str(e),
            }

def write_csv(rows, output_path: Path):
    """
    Zapíše zoznam slovníkov do CSV s hlavičkou.
    """
    # vybereme všetky kľúče z prvého riadka
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Inventory all files under a directory and export metadata to CSV."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Cesta k vrchnému adresáru, ktorý sa má prehľadať."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("inventory.csv"),
        help="Výstupný CSV súbor (predvolene: inventory.csv)."
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"Chyba: {args.root} nie je adresár.")
        return

    print(f"Scanning {args.root} ...")
    rows = list(scan_directory(args.root))
    print(f"Nájdených súborov: {len(rows)}")
    print(f"Ukladám do {args.output} ...")
    write_csv(rows, args.output)
    print("Hotovo.")

if __name__ == "__main__":
    main()
