import argparse
import json
import os
import sys
import logging

from crawler import Crawler
from indexer import InvertedIndex
from search import SearchEngine, BM25

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
PAGES_FILE = os.path.join(DATA_DIR, "pages.json")


def crawl_command(args):
    seeds = args.seeds
    if not seeds:
        seeds = [
            "https://en.wikipedia.org/wiki/Search_engine",
            "https://en.wikipedia.org/wiki/Information_retrieval",
            "https://en.wikipedia.org/wiki/Web_crawler",
            "https://en.wikipedia.org/wiki/Inverted_index",
            "https://en.wikipedia.org/wiki/Tf%E2%80%93idf",
        ]

    crawler = Crawler(
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        max_per_domain=args.max_per_domain,
        delay=args.delay,
        timeout=args.timeout,
    )

    logger.info("Starting crawl with %d seed URLs", len(seeds))
    results = crawler.crawl(seeds)

    os.makedirs(DATA_DIR, exist_ok=True)
    pages_data = [r.to_dict() for r in results]
    with open(PAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d pages to %s", len(pages_data), PAGES_FILE)

    return results


def index_command(args, pages=None):
    engine = SearchEngine()

    if pages is None:
        if not os.path.exists(PAGES_FILE):
            logger.error("No pages data found. Run 'crawl' first.")
            sys.exit(1)
        with open(PAGES_FILE, "r", encoding="utf-8") as f:
            pages_data = json.load(f)

        class _Page:
            pass

        pages = []
        for pd in pages_data:
            p = _Page()
            p.url = pd["url"]
            p.title = pd["title"]
            p.text = pd["text"]
            pages.append(p)

    doc_ids = engine.add_documents_batch(pages)
    logger.info("Indexed %d documents", len(doc_ids))

    os.makedirs(INDEX_DIR, exist_ok=True)
    engine.save_index(INDEX_DIR)
    logger.info("Index saved to %s", INDEX_DIR)

    return engine


def search_command(args):
    engine = SearchEngine()
    if not os.path.exists(INDEX_DIR):
        logger.error("No index found. Run 'index' first.")
        sys.exit(1)
    engine.load_index(INDEX_DIR)

    query = " ".join(args.query)
    results = engine.search(query, top_k=args.top_k)

    if not results:
        print(f"\nNo results found for: {query}")
        return

    print(f"\n{'='*70}")
    print(f"  Search results for: {query}")
    print(f"  Found {len(results)} result(s)")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Score: {r.score:.4f}")
        print(f"      Title: {r.title}")
        print(f"      URL:   {r.url}")
        if r.snippets:
            terms_str = ", ".join(r.snippets.keys())
            print(f"      Terms: {terms_str}")
    print()


def demo_command(args):
    logger.info("Running demo with built-in sample data...")
    print("\n" + "=" * 70)
    print("  Mini Search Engine Demo")
    print("=" * 70)

    sample_pages = [
        {
            "url": "https://example.com/python-tutorial",
            "title": "Python Programming Tutorial for Beginners",
            "text": (
                "Python is a high-level programming language known for its readability and simplicity. "
                "This tutorial covers Python basics including variables, data types, loops, and functions. "
                "Python supports multiple programming paradigms including procedural, object-oriented, "
                "and functional programming. Python is widely used in web development, data science, "
                "machine learning, and artificial intelligence. The Python ecosystem includes popular "
                "libraries such as NumPy, Pandas, TensorFlow, and Django."
            ),
        },
        {
            "url": "https://example.com/machine-learning",
            "title": "Introduction to Machine Learning with Python",
            "text": (
                "Machine learning is a subset of artificial intelligence that enables systems to learn "
                "from data. Python is the most popular language for machine learning due to its rich "
                "ecosystem of libraries. Scikit-learn provides simple tools for data mining and analysis. "
                "TensorFlow and PyTorch are deep learning frameworks that support neural networks. "
                "Machine learning algorithms include supervised learning, unsupervised learning, and "
                "reinforcement learning. Feature engineering and model evaluation are critical steps "
                "in the machine learning pipeline."
            ),
        },
        {
            "url": "https://example.com/web-development",
            "title": "Web Development with Python and JavaScript",
            "text": (
                "Web development involves building websites and web applications. The frontend uses "
                "HTML, CSS, and JavaScript to create user interfaces. The backend can use Python "
                "frameworks like Django and Flask. RESTful APIs connect frontend and backend systems. "
                "JavaScript frameworks like React and Vue.js are popular for building interactive UIs. "
                "Database management with SQL and NoSQL databases is essential for web applications. "
                "Web security practices include authentication, authorization, and data encryption."
            ),
        },
        {
            "url": "https://example.com/data-science",
            "title": "Data Science and Analytics with Python",
            "text": (
                "Data science combines statistics, programming, and domain knowledge to extract insights "
                "from data. Python is the primary language for data science with libraries like Pandas "
                "for data manipulation and Matplotlib for visualization. Data analysis involves data "
                "cleaning, exploratory analysis, and statistical modeling. Machine learning techniques "
                "are often used in data science for prediction and classification. Big data technologies "
                "like Apache Spark enable processing of large datasets."
            ),
        },
        {
            "url": "https://example.com/search-engine",
            "title": "How Search Engines Work: Crawling, Indexing, and Ranking",
            "text": (
                "Search engines use web crawlers to discover and download web pages from the internet. "
                "The crawled pages are then processed to build an inverted index that maps terms to "
                "documents. When a user submits a query, the search engine retrieves relevant documents "
                "from the index and ranks them by relevance. BM25 is a popular ranking function that "
                "considers term frequency, document length, and inverse document frequency. Modern "
                "search engines also use machine learning for query understanding and result ranking. "
                "TF-IDF is a simpler approach that BM25 improves upon by incorporating document length "
                "normalization and saturation of term frequency."
            ),
        },
        {
            "url": "https://example.com/information-retrieval",
            "title": "Information Retrieval and Text Mining",
            "text": (
                "Information retrieval is the science of searching for information in documents. "
                "The inverted index is the core data structure that enables fast keyword search. "
                "Boolean queries use AND, OR, and NOT operators to combine search terms. Relevance "
                "ranking uses algorithms like BM25 and TF-IDF to order results. Text mining extracts "
                "patterns and knowledge from unstructured text data. Natural language processing "
                "techniques help understand the semantics of queries and documents."
            ),
        },
        {
            "url": "https://example.com/crawler-design",
            "title": "Designing a Web Crawler: Architecture and Best Practices",
            "text": (
                "A web crawler systematically browses the internet to collect web pages. Key challenges "
                "include avoiding duplicate pages, respecting robots.txt, and managing crawl politeness. "
                "The URL frontier manages the queue of pages to visit. URL deduplication prevents "
                "revisiting the same page. Depth-limited crawling prevents the crawler from going too "
                "deep. Domain-level rate limiting ensures the crawler does not overload any single "
                "server. The crawler extracts links from each page and adds them to the frontier."
            ),
        },
        {
            "url": "https://example.com/database-indexing",
            "title": "Database Indexing and Query Optimization",
            "text": (
                "Database indexing improves query performance by creating data structures that enable "
                "fast lookups. B-tree indexes are commonly used for range queries. Hash indexes provide "
                "O(1) lookup for exact matches. Inverted indexes are used in full-text search engines. "
                "Query optimization involves choosing the best execution plan for a database query. "
                "Index maintenance is important for write-heavy workloads. Composite indexes can "
                "speed up multi-column queries."
            ),
        },
    ]

    class _Page:
        pass

    pages = []
    for sp in sample_pages:
        p = _Page()
        p.url = sp["url"]
        p.title = sp["title"]
        p.text = sp["text"]
        pages.append(p)

    engine = index_command(args, pages=pages)

    demo_queries = [
        "Python machine learning",
        "+Python +machine learning",
        "search engine indexing",
        "+crawler +index search",
        "web development JavaScript",
        "BM25 TF-IDF ranking",
        "database index query",
    ]

    for query in demo_queries:
        results = engine.search(query, top_k=5)
        print(f"\n  Query: '{query}'")
        print(f"  Results: {len(results)}")
        for i, r in enumerate(results[:3], 1):
            print(f"    {i}. [{r.score:.4f}] {r.title}")
        print()

    print("=" * 70)
    print("  Demo complete!")
    print("=" * 70)


def serve_command(args):
    from flask import Flask, render_template_string, request, jsonify

    app = Flask(__name__)

    engine = SearchEngine()
    if os.path.exists(INDEX_DIR):
        engine.load_index(INDEX_DIR)
    else:
        logger.warning("No index found at %s. Run 'index' or 'demo' first.", INDEX_DIR)

    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mini Search Engine</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f5f5; color: #333; min-height: 100vh; }
.container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
h1 { text-align: center; font-size: 2em; margin-bottom: 30px; color: #1a73e8; }
.search-box { display: flex; gap: 10px; margin-bottom: 30px; }
.search-box input { flex: 1; padding: 12px 16px; font-size: 16px; border: 2px solid #ddd;
                    border-radius: 8px; outline: none; transition: border-color 0.2s; }
.search-box input:focus { border-color: #1a73e8; }
.search-box button { padding: 12px 24px; font-size: 16px; background: #1a73e8; color: white;
                     border: none; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
.search-box button:hover { background: #1557b0; }
.help { background: #fff; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;
        font-size: 14px; color: #666; line-height: 1.6; }
.help code { background: #e8f0fe; padding: 2px 6px; border-radius: 4px; color: #1a73e8; }
.results { list-style: none; }
.result-item { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 12px;
               transition: box-shadow 0.2s; }
.result-item:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.result-title { font-size: 18px; color: #1a73e8; margin-bottom: 6px; }
.result-url { font-size: 13px; color: #006621; margin-bottom: 6px; }
.result-score { font-size: 12px; color: #999; }
.result-terms { font-size: 12px; color: #666; margin-top: 4px; }
.result-terms span { background: #e8f0fe; padding: 2px 8px; border-radius: 12px; margin-right: 4px; }
.stats { text-align: center; color: #999; margin-bottom: 20px; font-size: 14px; }
</style>
</head>
<body>
<div class="container">
<h1>🔍 Mini Search Engine</h1>
<form class="search-box" method="get" action="/">
<input type="text" name="q" value="{{ query }}" placeholder="Enter search query..." autofocus>
<button type="submit">Search</button>
</form>
<div class="help">
<strong>Query syntax:</strong>
<code>word</code> = optional (OR boost) &nbsp;|&nbsp;
<code>+word</code> = must include (AND) &nbsp;|&nbsp;
<code>-word</code> = exclude<br>
Example: <code>+Python +machine learning</code> → must contain "Python" AND "machine", boosted by "learning"
</div>
{% if query %}
<p class="stats">Found {{ results|length }} result(s) for "{{ query }}" (index: {{ doc_count }} docs)</p>
<ul class="results">
{% for r in results %}
<li class="result-item">
<div class="result-title">{{ r.title }}</div>
<div class="result-url">{{ r.url }}</div>
<div class="result-score">Relevance: {{ "%.4f"|format(r.score) }}</div>
{% if r.snippets %}
<div class="result-terms">
{% for term in r.snippets %}<span>{{ term }}</span>{% endfor %}
</div>
{% endif %}
</li>
{% endfor %}
</ul>
{% endif %}
</div>
</body>
</html>"""

    @app.route("/")
    def home():
        query = request.args.get("q", "")
        results = []
        if query:
            results = engine.search(query, top_k=20)
        return render_template_string(
            HTML_TEMPLATE,
            query=query,
            results=results,
            doc_count=engine.index.doc_count,
        )

    @app.route("/api/search")
    def api_search():
        query = request.args.get("q", "")
        top_k = int(request.args.get("top_k", 10))
        results = engine.search(query, top_k=top_k)
        return jsonify(
            {
                "query": query,
                "total": len(results),
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "url": r.url,
                        "title": r.title,
                        "score": round(r.score, 4),
                        "snippets": r.snippets,
                    }
                    for r in results
                ],
            }
        )

    @app.route("/api/index/add", methods=["POST"])
    def api_add_document():
        data = request.get_json()
        doc_id = engine.add_document(
            data.get("url", ""), data.get("title", ""), data.get("text", "")
        )
        engine.save_index(INDEX_DIR)
        return jsonify({"status": "ok", "doc_id": doc_id})

    logger.info("Starting web server at http://localhost:%d", args.port)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


def main():
    parser = argparse.ArgumentParser(
        description="Mini Search Engine - Crawl, Index, Search"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_crawl = subparsers.add_parser("crawl", help="Crawl web pages")
    p_crawl.add_argument("seeds", nargs="*", help="Seed URLs")
    p_crawl.add_argument("--max-depth", type=int, default=2)
    p_crawl.add_argument("--max-pages", type=int, default=50)
    p_crawl.add_argument("--max-per-domain", type=int, default=15)
    p_crawl.add_argument("--delay", type=float, default=1.0)
    p_crawl.add_argument("--timeout", type=int, default=10)

    p_index = subparsers.add_parser("index", help="Build index from crawled pages")

    p_search = subparsers.add_parser("search", help="Search the index")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--top-k", type=int, default=10)

    p_demo = subparsers.add_parser("demo", help="Run demo with sample data")

    p_serve = subparsers.add_parser("serve", help="Start web UI server")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.command == "crawl":
        crawl_command(args)
    elif args.command == "index":
        index_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "demo":
        demo_command(args)
    elif args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
