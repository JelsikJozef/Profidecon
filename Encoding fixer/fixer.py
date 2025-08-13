#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixer mojibake názvov súborov/priečinkov (UTF-8, stredná Európa, SK/CZ + TR)
- predvolene DRY-RUN (len vypíše, čo by premenoval)
- --apply vykoná premenovanie
- --show-examples zobrazí príklady, kde sa problematické sekvencie nachádzajú

Príklady:
  python fix_names.py --root ./Knowledge --show-examples
  python fix_names.py --root ./Knowledge                # dry-run
  python fix_names.py --root ./Knowledge --apply        # naozaj premenuj

Pozn.: Nemení obsah súborov, iba názvy.
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# --- Pomocné: stdout do UTF-8 (ak Python 3.7+ a na Linuxe) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# =========================
# 1) Rozšírený opravovač
# =========================
class WeirdSequenceFixerExt:
    """
    Dodatočné opravy mojibake sekvencií (bez dotyku obsahu súborov).

    Postup:
    1) Kontextové diakritiky: písmeno + (╠Б/╠М/╠И) -> písmeno s diakritikou
    2) Priame 1:1 náhrady (napr. ┬┤ -> ')
    3) Unicode NFC normalizácia a kozmetika (viacnásobné medzery, okraje)
    """

    # 1) Kontextové "kombinátory"
    COMBINERS: Dict[str, Dict[str, str]] = {
        "╠Б": {  # akút
            "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "y": "ý",
            "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú", "Y": "Ý",
            "l": "ĺ", "L": "Ĺ"
        },
        "╠М": {  # mäkčeň / háček
            "c": "č", "s": "š", "z": "ž", "C": "Č", "S": "Š", "Z": "Ž",
            "l": "ľ", "L": "Ľ", "t": "ť", "T": "Ť", "d": "ď", "D": "Ď",
            "n": "ň", "N": "Ň", "r": "ř", "R": "Ř"
        },
        "╠И": {  # prehláska
            "a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"
        },
    }

    # 2) Priame náhrady (validované na tvojich ukážkach)
    REPLACEMENTS: Dict[str, str] = {
        # EN/typografia
        "┬┤": "'",          # Landlord's
        "´": "",           # blúdiaci spacing-acute
        "┬╖": "•",         # bullet
        "꞉": ":",

        # SK/CZ (špecifické mojibake)
        "├┤": "ô",         # Dôvera
        "─║": "ĺ",         # predĺženie
        "─П": "ď",         # výpoveď
        "─М": "Č",         # Čestné
        "─З": "ć",         # Kovačević
        "├Н": "Í",         # MARKÍZA
        "┼а": "Š",         # PRERUŠENÝ
        "├Й": "Ý",         # PRERUŠENÝ
        "┼д": "ť",         # ŽIADOSŤ
        "tАУ": "–",        # en dash (UTF-8 → cp1251 mojibake)

        # Turečtina/poľština/rumunčina (podľa ukážok)
        "┼Ю": "Ş",  # Ş
        "┼Я": "ş",  # ş
        "├З": "Ç",  # Ç
        "├з": "ç",  # ç
        "├╢": "ö",  # ö
        "┼Д": "ń",  # ń
        "├Ц": "Ö",  # Ö
        "─░": "İ",  # İ (tur. veľké I s bodkou)
        "─Г": "ă",  # rum. ă

        # medzery
        "\u00A0": " ",     # NBSP
        "\u202F": " ",     # NNBSP / thin NBSP
    }

    # re pre “písmeno + (kombinátor)”
    _combiner_pat = re.compile(r"(.)(" + "|".join(map(re.escape, COMBINERS.keys())) + r")")

    def fix_name(self, name: str) -> str:
        if not name:
            return name

        s = name

        # 1) Kontextové kombinátory – aplikuj opakovane, kým sa niečo zmení
        while True:
            changed = False

            def _sub(m):
                nonlocal changed
                base, comb = m.group(1), m.group(2)
                repl = self.COMBINERS.get(comb, {}).get(base)
                if repl:
                    changed = True
                    return repl
                return m.group(0)

            s2 = self._combiner_pat.sub(_sub, s)
            if not changed:
                break
            s = s2

        # 2) Priame náhrady
        for bad, good in self.REPLACEMENTS.items():
            if bad in s:
                s = s.replace(bad, good)

        # 3) Normalizácia + kozmetika
        s = unicodedata.normalize("NFC", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s


# ==================================
# 2) Renamer + ukážky výskytov
# ==================================
COMMON_WEIRD_TOKENS: List[str] = [
    # kľúče z REPLACEMENTS + často sa vyskytujúce markery
    "┬┤", "´", "┬╖", "꞉",
    "├┤", "─║", "─П", "─М", "─З", "├Н", "┼а", "├Й", "┼д",
    "tАУ",
    "┼Ю", "┼Я", "├З", "├з", "├╢", "┼Д", "├Ц", "─░", "─Г",
    "\u00A0", "\u202F",
    # kombinátory – hľadáme ich samostatne
    "╠Б", "╠М", "╠И",
    # ďalšie z tvojich výpisov, ktoré nechávame na manuálne posúdenie
    "┬╖", "╨Ф", "╨", "цП", "Рф", "║д", "уВ│", "уГ╝"
]


class NameFixer:
    def __init__(self,
                 include: Optional[re.Pattern] = None,
                 exclude: Optional[re.Pattern] = None):
        self.ext = WeirdSequenceFixerExt()
        self.include = include
        self.exclude = exclude

    def _allowed(self, path: str) -> bool:
        rel = path
        if self.include and not self.include.search(rel):
            return False
        if self.exclude and self.exclude.search(rel):
            return False
        return True

    def propose(self, path: str) -> str:
        """Navrhni opravený NÁZOV (iba posledná komponenta)."""
        base = os.path.basename(path)
        fixed = self.ext.fix_name(base)
        return fixed

    def walk_paths(self, root: str) -> Iterable[Tuple[str, bool]]:
        """Yielduje (cesta, is_dir). Použijeme topdown=False, aby sme menili dirs až po files."""
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            # súbory
            for fn in filenames:
                old = os.path.join(dirpath, fn)
                if self._allowed(old):
                    yield old, False
            # priečinky
            for dn in dirnames:
                old = os.path.join(dirpath, dn)
                if self._allowed(old):
                    yield old, True

    def plan(self, root: str) -> List[Tuple[str, str, bool]]:
        """Vytvor plán premenovaní: (old_abs, new_abs, is_dir) – len ak by sa zmenil názov."""
        actions: List[Tuple[str, str, bool]] = []
        for old, is_dir in self.walk_paths(root):
            parent = os.path.dirname(old)
            new_name = self.propose(old)
            if not new_name or new_name == os.path.basename(old):
                continue
            new_abs = os.path.join(parent, new_name)

            # Ochrana: ak cieľ už existuje a nie je to presne rovnaká cesta -> preskoč
            if os.path.exists(new_abs) and os.path.abspath(new_abs) != os.path.abspath(old):
                print(f"⚠️  Kolízia, preskakujem:\n{old}\n -> {new_abs}\n", file=sys.stderr)
                continue

            actions.append((old, new_abs, is_dir))
        return actions

    def apply(self, actions: List[Tuple[str, str, bool]]) -> None:
        """Vykonaj premenovania (dirs aj files už v správnom poradí, keďže plan používa topdown=False)."""
        for old, new, is_dir in actions:
            try:
                os.rename(old, new)
                print(f"{old}  ->  {new}")
            except Exception as e:
                print(f"❌  Zlyhalo premenovanie:\n{old}\n -> {new}\n   {e}\n", file=sys.stderr)

    # ---------- Ukážky výskytov ----------
    def show_examples(self, root: str, tokens: List[str], max_examples: int = 10) -> None:
        """
        Vyhľadá a vypíše príklady, kde sa nachádzajú dané 'tokens' (podreťazce).
        Zobrazí počty + max N ukážok na token.
        """
        counts: Dict[str, int] = defaultdict(int)
        samples: Dict[str, List[str]] = defaultdict(list)

        def maybe_add(token: str, fullpath: str, is_dir: bool):
            base = os.path.basename(fullpath)
            # počítame výskyty v base name (nie v celej ceste)
            c = base.count(token)
            if c > 0:
                counts[token] += c
                if len(samples[token]) < max_examples:
                    tag = "[DIR]" if is_dir else "[FILE]"
                    samples[token].append(f"{tag} {fullpath}")

        # Prechádzame rekurzívne
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            for dn in dirnames:
                p = os.path.join(dirpath, dn)
                if not self._allowed(p):
                    continue
                for t in tokens:
                    if t:
                        maybe_add(t, p, True)
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if not self._allowed(p):
                    continue
                for t in tokens:
                    if t:
                        maybe_add(t, p, False)

        # Výstup
        # Najprv usporiadať tokeny podľa počtu desc
        ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

        for token, total in ordered:
            print(f"--- {repr(token)} (spolu {total}), ukážky max {max_examples} ---")
            for line in samples[token]:
                print(line)
            print()

        # Tokeny, ktoré sa nenašli, ale boli žiadané
        missing = [t for t in tokens if t not in counts]
        if missing:
            print("--- NENAŠLO SA ---")
            for t in missing:
                print(repr(t))
            print()


# =========================
# 3) CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Fixer názvov (mojibake) – dry-run/premenovanie/ukážky výskytov.")
    ap.add_argument("--root", default=".", help="Koreňový priečinok (default: .)")
    ap.add_argument("--apply", action="store_true", help="NAOZAJ premenuj (inak len dry-run).")
    ap.add_argument("--show-examples", action="store_true", help="Vypíš príklady výskytu problematických sekvencií.")
    ap.add_argument("--tokens", nargs="*", default=None,
                    help="Vlastný zoznam tokenov pre --show-examples. Ak neuvedieš, použije sa vstavaný.")
    ap.add_argument("--max-examples", type=int, default=10, help="Koľko ukážok max na token (default 10).")
    ap.add_argument("--include", default=None, help="Regex – spracuj len cesty, ktoré mu vyhovujú.")
    ap.add_argument("--exclude", default=None, help="Regex – vynechaj cesty, ktoré mu vyhovujú.")

    args = ap.parse_args()
    root = os.path.abspath(args.root)

    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    fixer = NameFixer(include=include_re, exclude=exclude_re)

    if args.show_examples:
        tokens = args.tokens if args.tokens else list(COMMON_WEIRD_TOKENS)
        fixer.show_examples(root, tokens=tokens, max_examples=args.max_examples)
        return

    # bežný režim: dry-run vs. apply
    actions = fixer.plan(root)

    if not actions:
        print("No changes to your files done. Would have converted 0 files.")
        return

    # Mimikuje tvoj formát výstupu:
    if not args.apply:
        # DRY-RUN – len vypíšeme plán
        for old, new, _ in actions:
            print(f"{old}  ->  {new}")
        print(f"\nNo changes to your files done. Would have converted {len(actions)} item(s).")
    else:
        fixer.apply(actions)
        print(f"\nDone. Renamed {len(actions)} item(s).")


if __name__ == "__main__":
    main()
