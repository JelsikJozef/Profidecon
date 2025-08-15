#!/usr/bin/env python3
"""
Taxonomy Analyzer for Profidecon

This module analyzes all JSONL files in the Preprocessed directory to create a comprehensive
taxonomy suitable for RAG applications. It extracts and organizes:
- Categories (from existing metadata)
- Tags (semantic keywords)
- Countries (mentioned in documents)
- Document types and sources

The analyzer processes all JSONL files and generates a structured taxonomy
that can be used for document organization and retrieval optimization.
"""

import json
import logging
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaxonomyStats:
    """Statistics about the taxonomy analysis."""
    total_files: int = 0
    processed_files: int = 0
    empty_files: int = 0
    categories: Dict[str, int] = None
    tags: Dict[str, int] = None
    countries: Dict[str, int] = None
    sources: Dict[str, int] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = {}
        if self.tags is None:
            self.tags = {}
        if self.countries is None:
            self.countries = {}
        if self.sources is None:
            self.sources = {}


class CountryExtractor:
    """Extracts country names from Slovak text."""
    
    # Common country names in Slovak and their variants
    COUNTRIES = {
        # EU Countries
        'slovensko': ['slovensko', 'slovenská republika', 'sr', 'slovakia'],
        'česko': ['česko', 'česká republika', 'čr', 'czech republic'],
        'poľsko': ['poľsko', 'poland', 'poľská republika'],
        'maďarsko': ['maďarsko', 'hungary', 'maďarská republika'],
        'rakúsko': ['rakúsko', 'austria', 'rakúska republika'],
        'nemecko': ['nemecko', 'germany', 'nemecká spolková republika', 'deutschland'],
        'francúzsko': ['francúzsko', 'france', 'francúzska republika'],
        'taliansko': ['taliansko', 'italy', 'talianska republika', 'italia'],
        'španielsko': ['španielsko', 'spain', 'španielske kráľovstvo'],
        'portugalsko': ['portugalsko', 'portugal', 'portugalská republika'],
        'holandsko': ['holandsko', 'netherlands', 'holandské kráľovstvo', 'nizozemsko'],
        'belgicko': ['belgicko', 'belgium', 'belgické kráľovstvo'],
        'luxembursko': ['luxembursko', 'luxembourg', 'luxemburské veľkovojvodstvo'],
        'dánsko': ['dánsko', 'denmark', 'dánske kráľovstvo'],
        'švédsko': ['švédsko', 'sweden', 'švédske kráľovstvo'],
        'fínsko': ['fínsko', 'finland', 'fínska republika'],
        'írsko': ['írsko', 'ireland', 'írska republika'],
        'grécko': ['grécko', 'greece', 'grécka republika'],
        'chorvátsko': ['chorvátsko', 'croatia', 'chorvátska republika'],
        'slovinsko': ['slovinsko', 'slovenia', 'slovinská republika'],
        'estónsko': ['estónsko', 'estonia', 'estónska republika'],
        'lotyšsko': ['lotyšsko', 'latvia', 'lotyšská republika'],
        'litva': ['litva', 'lithuania', 'litovská republika'],
        'malta': ['malta', 'maltská republika'],
        'cyprus': ['cyprus', 'cyperská republika'],
        'bulharsko': ['bulharsko', 'bulgaria', 'bulharská republika'],
        'rumunsko': ['rumunsko', 'romania', 'rumunská republika'],
        
        # Non-EU European countries
        'švajčiarsko': ['švajčiarsko', 'switzerland', 'švajčiarska konfederácia'],
        'nórsko': ['nórsko', 'norway', 'nórske kráľovstvo'],
        'island': ['island', 'iceland', 'islandská republika'],
        'spojené kráľovstvo': ['spojené kráľovstvo', 'uk', 'united kingdom', 'veľká británia', 'anglicko'],
        'srbsko': ['srbsko', 'serbia', 'srbská republika'],
        'čierna hora': ['čierna hora', 'montenegro'],
        'bosna a hercegovina': ['bosna a hercegovina', 'bosnia and herzegovina'],
        'macedónsko': ['macedónsko', 'north macedonia', 'severné macedónsko'],
        'albánsko': ['albánsko', 'albania', 'albánska republika'],
        'kosovo': ['kosovo', 'kosovská republika'],
        'moldavsko': ['moldavsko', 'moldova', 'moldavská republika'],
        'ukrajina': ['ukrajina', 'ukraine'],
        'bielorusko': ['bielorusko', 'belarus', 'bieloruská republika'],
        'rusko': ['rusko', 'russia', 'ruská federácia'],
        'turecko': ['turecko', 'turkey', 'turecká republika'],
        
        # Major non-European countries
        'usa': ['usa', 'united states', 'spojené štáty', 'americká'],
        'kanada': ['kanada', 'canada'],
        'čína': ['čína', 'china', 'čínska ľudová republika'],
        'japonsko': ['japonsko', 'japan'],
        'južná kórea': ['južná kórea', 'south korea', 'kórejská republika'],
        'india': ['india', 'indická republika'],
        'austrália': ['austrália', 'australia', 'austrálsky zväz'],
        'brazília': ['brazília', 'brazil', 'brazílska federatívna republika'],
        'mexiko': ['mexiko', 'mexico', 'mexické spojené štáty'],
        'argentína': ['argentína', 'argentina', 'argentínska republika'],
        'južná afrika': ['južná afrika', 'south africa', 'juhoafrická republika'],
        'egypt': ['egypt', 'egyptská arabská republika'],
        'maroko': ['maroko', 'morocco', 'marocké kráľovstvo'],
        'izrael': ['izrael', 'israel', 'štát izrael'],
        'saudská arábia': ['saudská arábia', 'saudi arabia', 'saudské kráľovstvo'],
        'emiráty': ['emiráty', 'uae', 'spojené arabské emiráty'],
    }
    
    def extract_countries(self, text: str) -> Set[str]:
        """Extract country names from text."""
        if not text:
            return set()
            
        text_lower = text.lower()
        found_countries = set()
        
        for country, variants in self.COUNTRIES.items():
            for variant in variants:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    found_countries.add(country)
                    break  # Found this country, move to next
                    
        return found_countries


class TaxonomyAnalyzer:
    """Analyzes JSONL files to create comprehensive taxonomy."""
    
    def __init__(self, preprocessed_dir: Path, output_dir: Path = None):
        """
        Initialize the taxonomy analyzer.
        
        Args:
            preprocessed_dir: Directory containing JSONL files
            output_dir: Directory for output files (default: Taxonomy/)
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("Taxonomy")
        self.output_dir.mkdir(exist_ok=True)
        
        self.country_extractor = CountryExtractor()
        self.stats = TaxonomyStats()
        
        # Taxonomy data structures
        self.categories = Counter()
        self.tags = Counter()
        self.countries = Counter()
        self.sources = Counter()
        self.document_types = Counter()
        self.languages = Counter()
        
        # Hierarchical structures
        self.category_tags = defaultdict(set)
        self.category_sources = defaultdict(set)
        self.source_hierarchy = defaultdict(set)
        
    def analyze_all_files(self) -> TaxonomyStats:
        """
        Analyze all JSONL files in the preprocessed directory.
        
        Returns:
            TaxonomyStats with analysis results
        """
        logger.info(f"Starting taxonomy analysis in: {self.preprocessed_dir}")
        
        # Find all JSONL files
        jsonl_files = list(self.preprocessed_dir.glob("*.jsonl"))
        self.stats.total_files = len(jsonl_files)
        
        if not jsonl_files:
            logger.warning(f"No JSONL files found in {self.preprocessed_dir}")
            return self.stats
            
        logger.info(f"Found {len(jsonl_files)} JSONL files to process")
        
        # Process each file
        for file_path in jsonl_files:
            try:
                self._process_file(file_path)
                self.stats.processed_files += 1
                
                if self.stats.processed_files % 100 == 0:
                    logger.info(f"Processed {self.stats.processed_files}/{self.stats.total_files} files")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
                
        # Update stats
        self.stats.categories = dict(self.categories)
        self.stats.tags = dict(self.tags)
        self.stats.countries = dict(self.countries)
        self.stats.sources = dict(self.sources)
        
        logger.info(f"Analysis complete: {self.stats.processed_files}/{self.stats.total_files} files processed")
        return self.stats
        
    def _process_file(self, file_path: Path) -> None:
        """Process a single JSONL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Skip empty or invalid documents
            if not data.get('text') and not data.get('summary'):
                self.stats.empty_files += 1
                return
                
            # Extract categories
            category = data.get('category', 'Unknown')
            self.categories[category] += 1
            
            # Extract and count tags
            tags = data.get('tags', [])
            for tag in tags:
                if tag:  # Skip empty tags
                    self.tags[tag] += 1
                    self.category_tags[category].add(tag)
                    
            # Extract countries from text and summary
            text_content = (data.get('text', '') + ' ' + data.get('summary', '')).strip()
            found_countries = self.country_extractor.extract_countries(text_content)
            for country in found_countries:
                self.countries[country] += 1
                
            # Extract source information
            source = data.get('source', 'Unknown')
            self.sources[source] += 1
            self.category_sources[category].add(source)
            
            # Build source hierarchy
            if source != 'Unknown':
                source_parts = source.split('/')
                for i in range(len(source_parts) - 1):
                    parent = '/'.join(source_parts[:i+1])
                    child = '/'.join(source_parts[:i+2])
                    self.source_hierarchy[parent].add(child)
                    
            # Extract language
            language = data.get('language', 'unknown')
            self.languages[language] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
    def generate_taxonomy(self) -> Dict[str, Any]:
        """
        Generate the complete taxonomy structure.
        
        Returns:
            Dictionary with organized taxonomy data
        """
        taxonomy = {
            'metadata': {
                'generated_at': self._get_timestamp(),
                'total_documents': self.stats.processed_files,
                'empty_documents': self.stats.empty_files,
                'analyzer_version': '1.0.0'
            },
            'categories': {
                'overview': dict(self.categories.most_common()),
                'detailed': {
                    category: {
                        'count': count,
                        'tags': list(self.category_tags[category]),
                        'sources': list(self.category_sources[category])
                    }
                    for category, count in self.categories.most_common()
                }
            },
            'tags': {
                'overview': dict(self.tags.most_common(100)),  # Top 100 tags
                'by_frequency': {
                    'high': dict(Counter({k: v for k, v in self.tags.items() if v >= 10}).most_common()),
                    'medium': dict(Counter({k: v for k, v in self.tags.items() if 3 <= v < 10}).most_common()),
                    'low': dict(Counter({k: v for k, v in self.tags.items() if v < 3}).most_common())
                }
            },
            'countries': {
                'overview': dict(self.countries.most_common()),
                'by_region': self._group_countries_by_region()
            },
            'sources': {
                'overview': dict(self.sources.most_common()),
                'hierarchy': {k: list(v) for k, v in self.source_hierarchy.items()}
            },
            'languages': dict(self.languages.most_common()),
            'statistics': {
                'total_categories': len(self.categories),
                'total_tags': len(self.tags),
                'total_countries': len(self.countries),
                'total_sources': len(self.sources),
                'unique_tags': sum(1 for count in self.tags.values() if count == 1),
                'common_tags': sum(1 for count in self.tags.values() if count >= 5)
            }
        }
        
        return taxonomy
        
    def _group_countries_by_region(self) -> Dict[str, Dict[str, int]]:
        """Group countries by geographical region."""
        eu_countries = {
            'slovensko', 'česko', 'poľsko', 'maďarsko', 'rakúsko', 'nemecko',
            'francúzsko', 'taliansko', 'španielsko', 'portugalsko', 'holandsko',
            'belgicko', 'luxembursko', 'dánsko', 'švédsko', 'fínsko', 'írsko',
            'grécko', 'chorvátsko', 'slovinsko', 'estónsko', 'lotyšsko',
            'litva', 'malta', 'cyprus', 'bulharsko', 'rumunsko'
        }
        
        europe_non_eu = {
            'švajčiarsko', 'nórsko', 'island', 'spojené kráľovstvo', 'srbsko',
            'čierna hora', 'bosna a hercegovina', 'macedónsko', 'albánsko',
            'kosovo', 'moldavsko', 'ukrajina', 'bielorusko', 'rusko', 'turecko'
        }
        
        regions = {
            'európska_únia': {},
            'európa_mimo_eú': {},
            'severná_amerika': {},
            'ázia': {},
            'ostatné': {}
        }
        
        for country, count in self.countries.items():
            if country in eu_countries:
                regions['európska_únia'][country] = count
            elif country in europe_non_eu:
                regions['európa_mimo_eú'][country] = count
            elif country in ['usa', 'kanada', 'mexiko']:
                regions['severná_amerika'][country] = count
            elif country in ['čína', 'japonsko', 'južná kórea', 'india']:
                regions['ázia'][country] = count
            else:
                regions['ostatné'][country] = count
                
        return regions
        
    def save_taxonomy(self, taxonomy: Dict[str, Any]) -> None:
        """Save taxonomy to multiple output formats."""
        
        # Save complete taxonomy as JSON
        output_file = self.output_dir / "taxonomy_complete.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(taxonomy, f, ensure_ascii=False, indent=2)
        logger.info(f"Complete taxonomy saved to: {output_file}")
        
        # Save individual components
        self._save_component(taxonomy['categories'], 'categories.json')
        self._save_component(taxonomy['tags'], 'tags.json')
        self._save_component(taxonomy['countries'], 'countries.json')
        self._save_component(taxonomy['sources'], 'sources.json')
        
        # Save RAG-optimized version (simplified for vector search)
        rag_taxonomy = self._create_rag_taxonomy(taxonomy)
        rag_file = self.output_dir / "taxonomy_rag.json"
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(rag_taxonomy, f, ensure_ascii=False, indent=2)
        logger.info(f"RAG-optimized taxonomy saved to: {rag_file}")
        
        # Save statistics summary
        stats_file = self.output_dir / "taxonomy_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(taxonomy['statistics'], f, ensure_ascii=False, indent=2)
        logger.info(f"Statistics saved to: {stats_file}")
        
    def _save_component(self, data: Dict, filename: str) -> None:
        """Save individual taxonomy component."""
        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def _create_rag_taxonomy(self, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """Create simplified taxonomy optimized for RAG applications."""
        return {
            'categories': list(taxonomy['categories']['overview'].keys()),
            'top_tags': list(dict(Counter(taxonomy['tags']['overview']).most_common(50)).keys()),
            'countries': list(taxonomy['countries']['overview'].keys()),
            'tag_hierarchy': {
                'legal_documents': [tag for tag in taxonomy['tags']['overview'] 
                                  if any(word in tag.lower() for word in ['zákon', 'vyhláška', 'nariadenie', 'smernica'])],
                'procedures': [tag for tag in taxonomy['tags']['overview'] 
                             if any(word in tag.lower() for word in ['žiadosť', 'pobyt', 'víza', 'proces'])],
                'institutions': [tag for tag in taxonomy['tags']['overview'] 
                               if any(word in tag.lower() for word in ['úrad', 'ministerstvo', 'polícia', 'ambasáda'])],
                'documents': [tag for tag in taxonomy['tags']['overview'] 
                            if any(word in tag.lower() for word in ['doklad', 'certifikát', 'potvrdenie', 'vyhlásenie'])]
            },
            'country_groups': taxonomy['countries']['by_region']
        }
        
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def print_summary(self) -> None:
        """Print analysis summary to console."""
        print("\n" + "="*60)
        print("📊 TAXONOMY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📁 Processed Files: {self.stats.processed_files}/{self.stats.total_files}")
        print(f"🗂️  Categories: {len(self.categories)}")
        print(f"🏷️  Unique Tags: {len(self.tags)}")
        print(f"🌍 Countries Found: {len(self.countries)}")
        print(f"📄 Sources: {len(self.sources)}")
        print(f"🗣️  Languages: {len(self.languages)}")
        
        print(f"\n📈 Top Categories:")
        for category, count in self.categories.most_common(10):
            print(f"  • {category}: {count}")
            
        print(f"\n🏷️  Top Tags:")
        for tag, count in self.tags.most_common(15):
            print(f"  • {tag}: {count}")
            
        print(f"\n🌍 Top Countries:")
        for country, count in self.countries.most_common(10):
            print(f"  • {country}: {count}")
            
        print("\n" + "="*60)


def load_raw_metadata(preprocessed_dir: Path) -> List[Dict[str, Any]]:
    """
    Load one JSON object per JSONL file (first line) from a directory.
    Returns a list of dict records.
    """
    records: List[Dict[str, Any]] = []
    preprocessed_dir = Path(preprocessed_dir)
    for p in sorted(preprocessed_dir.glob("*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as f:
                line = f.readline()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
        except Exception as e:
            logger.error("Failed to read %s: %s", p.name, e)
    return records


def build_tag_corpus(records: List[Dict[str, Any]]) -> Tuple["Any", List[str]]:
    """
    Build a TF-IDF matrix from a list of records with 'tags' arrays.
    Returns (X, feature_names)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs: List[str] = []
    for rec in records:
        tags = rec.get("tags") or []
        if isinstance(tags, list):
            docs.append(" ".join(str(t) for t in tags))
        else:
            docs.append("")
    vec = TfidfVectorizer(token_pattern=r"[^\s]+")
    X = vec.fit_transform(docs)
    features = list(vec.get_feature_names_out())
    return X, features


def cluster_tags(X: "Any", n_clusters: int = 5) -> List[int]:
    """
    Cluster documents by tag vectors using KMeans. Returns label list.
    """
    from sklearn.cluster import KMeans

    if getattr(X, "shape", None) is None or X.shape[0] == 0:
        return []
    k = max(1, min(n_clusters, X.shape[0]))
    # n_init='auto' for sklearn>=1.4 compatibility
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    return list(map(int, labels))


def propose_taxonomy(tags: List[str]) -> Dict[str, Any]:
    """
    Ask an LLM (OpenAI-style) to propose a JSON taxonomy given a list of tags.
    For tests, this uses openai.ChatCompletion.create which is monkeypatched there.
    """
    import openai  # type: ignore

    prompt = (
        "You are a helpful assistant. Given the following tags, propose a compact JSON taxonomy tree "
        "grouping related tags. Return ONLY valid JSON.\n\nTags: " + ", ".join(tags)
    )
    try:
        resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
        )
        content = resp.choices[0].message.content  # type: ignore[index]
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.error("propose_taxonomy failed: %s", e)
    # Fallback naive taxonomy
    return {"Root": {"Tags": list(tags)}}


def main(preprocessed_dir: str = "Preprocessed", output_dir: str = "Taxonomy"):
    """
    Main function to run taxonomy analysis.
    
    Args:
        preprocessed_dir: Directory containing JSONL files
        output_dir: Output directory for taxonomy files
    """
    try:
        # Initialize analyzer
        analyzer = TaxonomyAnalyzer(
            preprocessed_dir=Path(preprocessed_dir),
            output_dir=Path(output_dir)
        )
        
        # Run analysis
        stats = analyzer.analyze_all_files()
        
        # Generate taxonomy
        taxonomy = analyzer.generate_taxonomy()
        
        # Save results
        analyzer.save_taxonomy(taxonomy)
        
        # Print summary
        analyzer.print_summary()
        
        logger.info("✅ Taxonomy analysis completed successfully!")
        return taxonomy
        
    except Exception as e:
        logger.error(f"❌ Taxonomy analysis failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze JSONL files to create taxonomy")
    parser.add_argument("--input", "-i", default="Preprocessed", 
                       help="Input directory with JSONL files (default: Preprocessed)")
    parser.add_argument("--output", "-o", default="Taxonomy",
                       help="Output directory for taxonomy files (default: Taxonomy)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    main(args.input, args.output)
