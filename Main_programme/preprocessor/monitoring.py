from prometheus_client import Counter, Histogram, start_http_server
import time

# 1) Definícia metrik
DOCS_PROCESSED = Counter(
    "docs_processed_total", "Počet úspešne spracovaných dokumentov"
)
DOCS_FAILED = Counter(
    "docs_failed_total", "Počet dokumentov, pri ktorých spracovaní došlo k chybe"
)
PROCESS_TIME = Histogram(
    "doc_process_seconds", "Čas spracovania jedného dokumentu v sekundách"
)

def start_metrics_server(port: int = 8000):
    """
    Spustí jednoduchý HTTP endpoint /metrics pre Prometheus.
    """
    start_http_server(port)

def instrument(func):
    """
    Dekorátor na meranie a report metrik pri spracovaní jedného dokumentu.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            DOCS_PROCESSED.inc()
            return result
        except Exception:
            DOCS_FAILED.inc()
            raise
        finally:
            PROCESS_TIME.observe(time.time() - start)
    return wrapper
