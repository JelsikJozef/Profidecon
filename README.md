# Profidecon - Hybridný RAG systém pre právne dokumenty

Profidecon je pokročilý systém pre spracovanie a vyhľadávanie právnych dokumentov používajúci hybridné vyhľadávanie s dual vectors a tag boosting pre optimálne výsledky.

## 🚀 Kľúčové funkcie

- **Hybridné vyhľadávanie**: Dual vector embeddings (text + summary) s tag-based score boosting
- **Pokročilé spracovanie**: OCR, normalizácia textu, deduplication a kvalitné kontroly
- **Taxonómia dokumentov**: Automatická kategorizácia a extraktion tagov pomocou LLM
- **RAG optimalizácia**: Optimalizované pre slovenčinu a právne dokumenty
- **Vysoká presnosť**: 67% Recall@10 na testovacích dátach

## 📦 Inštalácia

```bash
# Klonovanie repozitára
git clone <repository-url>
cd Profidecon

# Inštalácia závislostí
pip install -e .

# Spustenie Qdrant servera
docker run -p 6333:6333 qdrant/qdrant
```

## 🛠️ Použitie

### CLI nástroj
Po inštalácii máte k dispozícii `profidecon` CLI:

```bash
profidecon --help
```

## 📋 Príklad workflow

### 1. Spracovanie dokumentov
```bash
# Spracovanie dokumentov s OCR a normalizáciou
profidecon preprocess --input ./Knowledge --output ./Preprocessed
```

### 2. Vytvorenie taxonómie
```bash
# Analýza a vytvorenie taxonómie zo všetkých JSONL súborov
profidecon taxonomy-analyze --input ./Preprocessed --output ./Taxonomy
```

### 3. Načítanie dual vectors do Qdrant
```bash
# Vytvorenie embeddings pre text aj summary s dual vector support
profidecon vector-load ./Preprocessed --glob "*.jsonl"
```

### 4. Hybridné vyhľadávanie

```python
from Main_programme.sdk.retrieval import RetrievalEngine

# Inicializácia retrieval engine
engine = RetrievalEngine()

# Hybridné vyhľadávanie s tag boostingom
results = engine.search(
    query="poplatok na ambasáde",
    user_tags=["poplatok", "prechodný pobyt"],
    tag_boost=0.25,  # 25% boost pre matching tagy
    limit=10
)

# Zobrazenie výsledkov
for i, result in enumerate(results, 1):
    boost_indicator = "📈 BOOSTED" if result.was_boosted else ""
    print(f"{i}. {boost_indicator} Score: {result.score:.3f}")
    print(f"   Category: {result.category}")
    print(f"   Text: {result.text[:100]}...")
    if result.matched_tags:
        print(f"   🎯 Matched tags: {result.matched_tags}")
```

### 5. Summary Vector Search
```python
# Vyhľadávanie v summary embeddings
summary_results = engine.search_summary_vector(
    query="dokumenty potrebné pre pobyt",
    user_tags=["dokumenty", "pobyt"],
    limit=5
)
```

## 🎯 Hybridné vyhľadávanie funkcie

### Dual Vector Embeddings
- **Body vectors**: Embeddings plného textu dokumentu
- **Summary vectors**: Embeddings súhrnu dokumentu
- Možnosť vyhľadávania v oboch typoch vektorov

### Tag Boosting
- Automatické zvýšenie skóre dokumentov s matching tagmi
- Konfigurovateľný boost factor (default 20%)
- Inteligentná re-ranking výsledkov

### Konfigurácia vyhľadávania

```python
from Main_programme.vectorizer.settings import settings

# Úprava globálnych nastavení
settings.tag_boost = 0.30  # 30% boost
settings.search_limit = 15  # 15 výsledkov
settings.use_summary_vector = True  # Použiť summary vektor
```

## 📊 Výkonnosť systému

Na testovacích dátach (33 ground truth queries):
- **Recall@10: 66.7%** (22/33 správnych výsledkov)
- **Tag boost efektívnosť: 36.4%** dotazov profituje z boostingu
- **Dual vector podpora**: Seamless switching medzi text a summary embeddings

## 🗂️ Štruktúra projektu

```
Profidecon/
├── preprocessor/        # Document processing pipeline
│   ├── taxonomy/       # LLM-based taxonomy extraction
│   └── processors/     # OCR, normalization, quality checks
├── vectorizer/         # Dual vector embedding system  
│   ├── embedder.py    # SentenceTransformer wrapper
│   ├── loader.py      # Qdrant integration with dual vectors
│   └── settings.py    # Configuration management
├── sdk/               # Retrieval API
│   └── retrieval.py   # Hybrid search engine
├── tests/             # Test data and evaluation
└── profidecon/        # Main CLI application
```

## 📚 Dokumentácia API

### RetrievalEngine

```python
from Main_programme.sdk.retrieval import RetrievalEngine

engine = RetrievalEngine()

# Základné vyhľadávanie
results = engine.search(query="text", user_tags=["tag1", "tag2"])

# Pokročilé možnosti
results = engine.search(
    query="search text",
    user_tags=["tag1", "tag2"],
    limit=10,
    tag_boost=0.25,
    use_summary_vector=False
)

# Summary vector search
results = engine.search_summary_vector(query="text", user_tags=["tag"])

# Získanie dokumentu podľa ID  
doc = engine.get_document_by_id("document-uuid")

# Štatistiky kolekcie
stats = engine.get_collection_stats()
```

### SearchResult Object

```python
@dataclass
class SearchResult:
    id: str                    # Document UUID
    score: float              # Final score (potentially boosted)
    original_score: float     # Original vector similarity score
    text: str                 # Full document text
    summary: str              # Document summary
    tags: List[str]           # Document tags
    category: str             # Document category
    source: str               # Original file path
    was_boosted: bool         # Whether tag boosting was applied
    boost_factor: float       # Applied boost factor
    matched_tags: List[str]   # Tags that matched user query
```

## ⚙️ Konfigurácia

Systém používa `vectorizer/settings.py` pre konfiguráciu:

```python
# Embedding model
embed_model = "intfloat/multilingual-e5-base"

# Qdrant nastavenia  
qdrant_url = "http://localhost:6333"
qdrant_collection = "profidecon_docs"

# Vyhľadávanie
tag_boost = 0.20              # 20% boost pre matching tagy
search_limit = 10             # Počet výsledkov
use_summary_vector = False    # Typ vektora pre vyhľadávanie

# Processing
batch_size = 32
chunk_size = 450
chunk_overlap = 100
```

## 🧪 Testovanie

```bash
# Test hybridného vyhľadávania
python test_hybrid_search.py

# Evaluácia na ground truth dátach
python test_human_csv.py

# RAG pipeline evaluácia
python evaluate_rag.py
```

## 📈 Metriky kvality

Systém poskytuje kompletnú evaluáciu:
- **Recall@K**: Úspešnosť nájdenia správnych dokumentov
- **Tag boost impact**: Efektívnosť tag boostingu
- **Vector type comparison**: Body vs summary embeddings
- **Response time**: Rýchlosť vyhľadávania

## 🔧 Pokročilé použitie

### Custom embedding models

```python
from Main_programme.vectorizer.settings import Settings

settings = Settings(embed_model="your-custom-model")
engine = RetrievalEngine(settings)
```

### Batch processing
```python
# Spracovanie viacerých dotazov
queries = ["query1", "query2", "query3"]
all_results = []

for query in queries:
    results = engine.search(query, tag_boost=0.30)
    all_results.extend(results)
```

## 🐛 Riešenie problémov

### Qdrant connection issues
```bash
# Skontrolujte či beží Qdrant server
curl http://localhost:6333/collections

# Reštart Qdrant
docker restart qdrant-container
```

### Embedding model loading
```bash
# Pre offline použitie, predownloadujte model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
```

## 🤝 Prispievanie

1. Fork repozitára
2. Vytvorte feature branch (`git checkout -b feature/amazing-feature`)
3. Commit zmeny (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Otvorte Pull Request

## 📄 Licencia

MIT

## 📞 Kontakt

V prípade otázok kontaktujte autora projektu.
