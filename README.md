# Profidecon

Profidecon je komplexný nástroj na spracovanie dokumentov, generovanie taxonómie a vektorové vyhľadávanie pre systémy Retrieval Augmented Generation (RAG). Projekt je určený pre právne, administratívne a znalostné dokumenty.

## Funkcionalita

- **Preprocessing**: Automatizované spracovanie dokumentov (OCR, normalizácia, obohatenie metadát, deduplikácia, export do JSONL)
- **Taxonomy Extract**: Extrakcia metadát a tém pre tvorbu taxonómie pomocou LLM
- **Taxonomy Analyze**: Návrh hierarchickej taxonómie vrátane kategórie krajín
- **Vector Load**: Vytváranie embeddingov a ich nahrávanie do Qdrant pre vektorové vyhľadávanie
- **Full Pipeline**: Spustenie celého workflow jedným príkazom

## Inštalácia

1. Klonujte repozitár a prejdite do adresára projektu:
   ```bash
   git clone ...
   cd Profidecon
   ```
2. Nainštalujte závislosti a CLI:
   ```bash
   pip install -e .
   ```

## Použitie

Všetky príkazy sú dostupné cez jednotný CLI nástroj `profidecon`:

```bash
profidecon --help
```

### Príkazy

- **Preprocessing**
  ```bash
  profidecon preprocess --input ./Knowledge --output ./Preprocessed
  ```
- **Taxonomy Extract**
  ```bash
  profidecon taxonomy-extract ./Knowledge --out metadata_raw.jsonl
  ```
- **Taxonomy Analyze**
  ```bash
  profidecon taxonomy-analyze ./Knowledge --preprocessed ./Preprocessed --out taxonomy.json
  ```
  > **Poznámka:** Ak zadáte cestu k adresáru vo voľbe `--out`, výsledná taxonómia bude uložená do súboru `taxonomy.json` v tomto adresári. Pre explicitné uloženie použite cestu k súboru (napr. `--out ./taxonomy.json`).

- **Vector Load**
  ```bash
  profidecon vector-load ./Preprocessed --glob "*.jsonl"
  ```
- **Full Pipeline**
  ```bash
  profidecon full-pipeline --input ./Knowledge --preprocessed ./Preprocessed --taxonomy-out taxonomy.json
  ```

### Globálne voľby
- `--verbose` : Podrobné logovanie
- `--config`  : Cesta ku konfiguračnému súboru

## Architektúra

- `main.py` – hlavný CLI vstup
- `preprocessor/` – moduly na spracovanie dokumentov a taxonómiu
- `vectorizer/` – embedding a Qdrant integrácia
- `Knowledge/` – vstupné dokumenty
- `Preprocessed/` – výstupné JSONL súbory

## Príklad workflow
1. Spracujte dokumenty:
   ```bash
   profidecon preprocess --input ./Knowledge --output ./Preprocessed
   ```
2. Vytvorte taxonómiu:
   ```bash
   profidecon taxonomy-analyze ./Knowledge --preprocessed ./Preprocessed --out taxonomy.json
   ```
3. Nahrajte embeddingy do Qdrant:
   ```bash
   profidecon vector-load ./Preprocessed
   ```

## Požiadavky
- Python 3.11+
- Qdrant server (pre vektorové vyhľadávanie)

## Licencia
MIT

## Kontakt
V prípade otázok kontaktujte autora projektu.
