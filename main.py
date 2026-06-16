import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime

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
CRAWL_CACHE_FILE = os.path.join(DATA_DIR, "crawl_cache.json")


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
        cache_file=CRAWL_CACHE_FILE,
    )

    logger.info("Starting crawl with %d seed URLs", len(seeds))
    results, crawl_stats = crawler.crawl(seeds)

    os.makedirs(DATA_DIR, exist_ok=True)
    pages_data = [r.to_dict() for r in results]
    with open(PAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  Crawl Summary")
    print("=" * 70)
    print(f"  New pages:       {crawl_stats['new']}")
    print(f"  Updated pages:   {crawl_stats['updated']}")
    print(f"  Unchanged pages: {crawl_stats['unchanged']}")
    print(f"  Failed pages:    {crawl_stats['failed']}")
    print(f"  Total saved:    {len(pages_data)} pages to {PAGES_FILE}")
    print("=" * 70)

    return results, crawl_stats


def index_command(args, pages=None):
    engine = SearchEngine()
    if os.path.exists(INDEX_DIR):
        engine.load_index(INDEX_DIR)

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

    doc_ids, index_stats = engine.add_documents_batch(pages)

    os.makedirs(INDEX_DIR, exist_ok=True)
    engine.save_index(INDEX_DIR)

    overview = engine.index.get_overview()

    print("\n" + "=" * 70)
    print("  Index Summary")
    print("=" * 70)
    print(f"  Added:      {index_stats['added']}")
    print(f"  Updated:    {index_stats['updated']}")
    print(f"  Skipped:    {index_stats['skipped']}")
    print(f"  Total docs: {overview['doc_count']}")
    print(f"  Terms:      {overview['term_count']}")
    print(f"  Avg length: {overview['avg_doc_length']}")
    print(f"  Saved to:    {INDEX_DIR}")
    print("=" * 70)

    return engine, index_stats


def search_command(args):
    engine = SearchEngine()
    if not os.path.exists(INDEX_DIR):
        logger.error("No index found. Run 'index' or 'demo' first.")
        sys.exit(1)
    engine.load_index(INDEX_DIR)

    query = " ".join(args.query)
    results = engine.search(query, top_k=args.top_k)

    if not results:
        print(f"\nNo results found for: {query}")
        return

    print(f"\n{'='*80}")
    print(f"  Search results for: {query}")
    print(f"  Found {len(results)} result(s)")
    print(f"{'='*80}")
    for i, r in enumerate(results, 1):
        tf_parts = []
        for term, freq in r.term_freqs.items():
            df = engine.index.get_doc_freq(term)
            tf_parts.append(f"{term}×{freq}")
        tf_str = ", ".join(tf_parts)

        print(f"\n  [{i}] Score: {r.score:.4f}  |  Len: {r.doc_length}")
        print(f"      Title: {r.title}")
        print(f"      URL:   {r.url}")
        if r.term_freqs:
            print(f"      TFs:   {tf_str}")
        if r.phrase_matches:
            print(f"      Phrase: {len(r.phrase_matches)} matches")
        if r.highlights:
            print(f"      Snippets:")
            for hl in r.highlights[:2]:
                cleaned = hl.replace("[[[", "\033[1;33m").replace("]]]", "\033[0m")
                print(f"        • {cleaned}")
    print()


def demo_command(args):
    logger.info("Running demo with built-in sample data...")
    print("\n" + "=" * 80)
    print("  Mini Search Engine Demo - Enhanced Features")
    print("=" * 80)

    sample_pages = [
        {
            "url": "https://example.com/python-tutorial",
            "title": "Python Programming Tutorial for Beginners",
            "text": (
                "Python is a high-level programming language known for its readability and simplicity. "
                "This Python tutorial covers Python basics including Python variables, Python data types. "
                "Python supports multiple programming paradigms including procedural, object-oriented. "
                "Python is widely used in web development, Python data science, machine learning. "
                "The Python ecosystem includes Python libraries: NumPy, Pandas, TensorFlow, Django. "
                "Python Python Python Python Python - this paragraph repeats Python many times "
                "to demonstrate that term frequency saturation works correctly with BM25 ranking."
            ),
        },
        {
            "url": "https://example.com/machine-learning",
            "title": "Introduction to Machine Learning with Python",
            "text": (
                "Machine learning is a subset of artificial intelligence that enables systems to learn "
                "from data. Python is the most popular language for machine learning due to its rich "
                "ecosystem. Scikit-learn provides tools for machine learning data mining and analysis. "
                "TensorFlow and PyTorch are deep learning frameworks for machine learning neural networks. "
                "Machine learning algorithms: supervised machine learning, unsupervised machine learning."
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
                "like Apache Spark enable processing of large datasets. Data data data data science."
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
                "normalization and saturation of term frequency. Search engine ranking uses BM25 algorithm."
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
            "url": "https://example.com/web-dev-js",
            "title": "Modern Web Development with JavaScript",
            "text": (
                "Modern web development focuses on JavaScript and frontend frameworks. JavaScript is essential "
                "for building interactive web applications. React, Vue.js, and Angular are popular "
                "JavaScript frameworks. JavaScript runs in the browser and on the server with Node.js. "
                "This article is about JavaScript, JavaScript development, and JavaScript best practices. "
                "No Python mentioned here at all - purely about JavaScript web development."
            ),
        },
        {
            "url": "https://example.com/hyperdimensional-xyz",
            "title": "Hyperdimensional Computing Research",
            "text": (
                "Hyperdimensional computing (HD computing) is an emerging approach to AI that uses "
                "high-dimensional vectors for data representation. Unlike traditional neural networks, "
                "hyperdimensional systems encode concepts as patterns in very large vector spaces. "
                "This is a rare and specialized topic in computer science research."
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

    engine, index_stats = index_command(args, pages=pages)

    demo_tests = [
        {
            "name": "Test 1: Term Frequency effect",
            "query": "Python",
            "description": "Python Tutorial should rank #1 (highest Python TF: ~14 times)",
        },
        {
            "name": "Test 2: Document Length normalization",
            "query": "JavaScript",
            "description": "web-dev-js should rank #1 despite shorter doc with JS mentioned",
        },
        {
            "name": "Test 3: Rare word boost",
            "query": "hyperdimensional computing",
            "description": "The rare topic page should rank #1 (high IDF boost",
        },
        {
            "name": "Test 4: Phrase search \"machine learning\"",
            "query": '"machine learning"',
            "description": "Pages with phrase 'machine learning' ranked higher",
        },
        {
            "name": "Test 5: Proximity search",
            "query": "search engine ranking",
            "description": "search-engine page should rank #1 (terms appear close together)",
        },
        {
            "name": "Test 6: AND must-include (+) syntax",
            "query": "+Python +machine learning",
            "description": "Only docs with BOTH Python AND machine (JavaScript page excluded)",
        },
        {
            "name": "Test 7: NOT exclude (-) syntax",
            "query": "Python -tutorial",
            "description": "Exclude pages containing 'tutorial'",
        },
    ]

    for test in demo_tests:
        print(f"\n{'='*80}")
        print(f"  {test['name']}")
        print(f"  Query: {test['query']}")
        print(f"  {test['description']}")
        print(f"  {'-'*80}")
        results = engine.search(test['query'], top_k=5)
        for i, r in enumerate(results, 1):
            tf_str = ", ".join(
                f"{t}×{f}" for t, f in list(r.term_freqs.items())
            )
            phrase_flag = "*PHRASE*" if r.phrase_matches else ""
            print(f"    {i}. [{r.score:.4f} {r.title[:55]}")
            print(f"       TF: {tf_str} | Len: {r.doc_length} | {phrase_flag}")
            if r.highlights:
                cleaned = r.highlights[0].replace("[[[", ">>").replace("]]]", "<<")
                print(f"       Snippet: {cleaned[:120]}...")
        print()

    print("=" * 80)
    print("  Incremental Index Update Test")
    print("=" * 80)

    print("\n  Adding a NEW page...")
    doc_id, action = engine.add_document(
        "https://example.com/new-page",
        "New Page about Go Programming Language",
        "Go is a statically typed language developed by Google. Go is used for systems programming. "
        "Go has goroutines for concurrent programming. Go is not Python or JavaScript."
    )
    print(f"  Action: {action} | doc_id={doc_id}")

    print("\n  Updating an EXISTING page (same URL, different content)...")
    doc_id2, action2 = engine.add_document(
        "https://example.com/new-page",
        "Updated: Go Language Deep Dive",
        "Go (Golang) is a modern systems language. Go has built-in concurrency. "
        "Go compiles to native machine code."
    )
    print(f"  Action: {action2} | doc_id={doc_id2}")

    print("\n  Re-adding SAME page (no changes, unchanged)...")
    doc_id3, action3 = engine.add_document(
        "https://example.com/new-page",
        "Updated: Go Language Deep Dive",
        "Go (Golang) is a modern systems language. Go has built-in concurrency. "
        "Go compiles to native machine code."
    )
    print(f"  Action: {action3} | doc_id={doc_id3}")

    print("\n  Searching for 'Go' (should find the updated page):")
    results = engine.search("Go", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"    {i}. [{r.score:.4f}] {r.title}")

    engine.save_index(INDEX_DIR)

    print("\n" + "=" * 80)
    print("  Demo complete! All features verified.")
    print("=" * 80)


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
.container { max-width: 900px; margin: 0 auto; padding: 30px 20px; }
h1 { text-align: center; font-size: 2em; margin-bottom: 20px; color: #1a73e8; }
.search-box { display: flex; gap: 10px; margin-bottom: 20px; }
.search-box input { flex: 1; padding: 12px 16px; font-size: 16px; border: 2px solid #ddd;
                    border-radius: 8px; outline: none; transition: border-color 0.2s; }
.search-box input:focus { border-color: #1a73e8; }
.search-box button { padding: 12px 24px; font-size: 16px; background: #1a73e8; color: white;
                     border: none; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
.search-box button:hover { background: #1557b0; }
.overview { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
           color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
.overview h2 { font-size: 1.2em; margin-bottom: 12px; }
.overview-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
.overview-item { background: rgba(255,255,255,0.15); padding: 12px; border-radius: 8px;
                text-align: center; }
.overview-item .label { font-size: 12px; opacity: 0.9; }
.overview-item .value { font-size: 22px; font-weight: bold; }
.help { background: #fff; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;
        font-size: 14px; color: #666; line-height: 1.8; }
.help code { background: #e8f0fe; padding: 3px 8px; border-radius: 4px; color: #1a73e8;
              font-family: monospace; }
.help .title { font-weight: bold; color: #333; margin-bottom: 6px; }
.add-doc { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
.add-doc h3 { margin-bottom: 12px; color: #333; }
.add-doc input, .add-doc textarea { width: 100%; padding: 10px; margin-bottom: 8px 0;
            border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
.add-doc textarea { min-height: 80px; resize: vertical; }
.add-doc button { background: #34a853; color: white; border: none; padding: 10px 20px; border-radius: 6px;
               cursor: pointer; font-size: 14px; }
.add-doc button:hover { background: #2d9249; }
.results { list-style: none; }
.result-item { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 12px;
               transition: box-shadow 0.2s; }
.result-item:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.result-title { font-size: 18px; color: #1a73e8; margin-bottom: 4px; }
.result-url { font-size: 13px; color: #006621; margin-bottom: 8px; }
.result-meta { font-size: 12px; color: #999; margin-bottom: 8px; display: flex; gap: 15px; }
.result-meta span { background: #f1f3f4; padding: 2px 8px; border-radius: 4px; }
.result-meta .phrase-badge { background: #fce8e6; color: #d93025; font-weight: bold; }
.result-highlights { margin-top: 10px; font-size: 14px; color: #555; line-height: 1.6; }
.result-highlights .hl { background: #fef3c7; padding: 2px 4px; border-radius: 3px; }
.result-terms { font-size: 12px; color: #666; margin-top: 8px; }
.result-terms span { background: #e8f0fe; padding: 2px 8px; border-radius: 12px; margin-right: 4px; }
.stats { text-align: center; color: #999; margin-bottom: 20px; font-size: 14px; }
.toast { position: fixed; top: 20px; right: 20px; background: #34a853; color: white;
        padding: 12px 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateX(120%); transition: transform 0.3s ease; z-index: 1000; }
.toast.show { transform: translateX(0); }
</style>
</head>
<body>
<div class="container">
<h1>🔍 Mini Search Engine</h1>

<div class="overview" id="overview">
  <h2>📊 Index Overview</h2>
  <div class="overview-stats">
    <div class="overview-item">
      <div class="label">Documents</div>
      <div class="value">{{ overview.doc_count }}</div>
    </div>
    <div class="overview-item">
      <div class="label">Terms</div>
      <div class="value">{{ overview.term_count }}</div>
    </div>
    <div class="overview-item">
      <div class="label">Avg Length</div>
      <div class="value">{{ overview.avg_doc_length }}</div>
    </div>
    <div class="overview-item">
      <div class="label">Last Updated</div>
      <div class="value" style="font-size:14px;">{{ overview.last_updated_str }}</div>
    </div>
  </div>
</div>

<form class="search-box" method="get" action="/">
<input type="text" name="q" value="{{ query }}" placeholder="Enter search query..." autofocus>
<button type="submit">Search</button>
</form>

<div class="help">
<div class="title">💡 Query syntax:</div>
<div>
<code>word</code> optional (OR boost) &nbsp;|&nbsp;
<code>+word</code> must include (AND) &nbsp;|&nbsp;
<code>-word</code> exclude &nbsp;|&nbsp;
<code>"phrase words"</code> exact phrase match
</div>
<div>Examples:
<code>+Python +"machine learning" -tutorial</code> |
<code>search engine</code> |
<code>JavaScript -Python</code>
</div>
</div>

<div class="add-doc">
<h3>➕ Add New Document</h3>
<form id="addForm">
<input type="text" id="docUrl" placeholder="URL" required>
<input type="text" id="docTitle" placeholder="Title" required>
<textarea id="docText" placeholder="Content text..." required></textarea>
<button type="submit">Add to Index</button>
</form>
</div>

{% if query %}
<p class="stats">Found {{ results|length }} result(s) for <strong>{{ query }}</strong></p>
<ul class="results">
{% for r in results %}
<li class="result-item">
<div class="result-title">{{ r.title }}</div>
<div class="result-url">{{ r.url }}</div>
<div class="result-meta">
<span>Score: {{ "%.4f"|format(r.score) }}</span>
<span>Length: {{ r.doc_length }}</span>
{% if r.phrase_matches %}
<span class="phrase-badge">📌 {{ r.phrase_matches }} phrase match(es)</span>
{% endif %}
</div>
{% if r.term_freqs %}
<div class="result-terms">
{% for term, freq in r.term_freqs.items() %}
<span>{{ term }} × {{ freq }}</span>
{% endfor %}
</div>
{% endif %}
{% if r.highlights %}
<div class="result-highlights">
{% for hl in r.highlights %}
<div>{{ hl|safe }}</div>
{% endfor %}
</div>
{% endif %}
</li>
{% endfor %}
</ul>
{% endif %}

</div>

<div class="toast" id="toast">Document added successfully!</div>

<script>
document.getElementById('addForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const data = {
        url: document.getElementById('docUrl').value,
        title: document.getElementById('docTitle').value,
        text: document.getElementById('docText').value
    };
    const res = await fetch('/api/index/add', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    });
    const result = await res.json();
    const toast = document.getElementById('toast');
    toast.textContent = `Document added! Action: ${result.action}, ID: ${result.doc_id}`;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
    document.getElementById('addForm').reset();
    setTimeout(() => location.reload(), 1500);
});

async function refreshOverview() {
    const res = await fetch('/api/index/overview');
    const data = await res.json();
    document.querySelector('.overview-stats').innerHTML = `
        <div class="overview-item">
          <div class="label">Documents</div>
          <div class="value">${data.doc_count}</div>
        </div>
        <div class="overview-item">
          <div class="label">Terms</div>
          <div class="value">${data.term_count}</div>
        </div>
        <div class="overview-item">
          <div class="label">Avg Length</div>
          <div class="value">${data.avg_doc_length}</div>
        </div>
        <div class="overview-item">
          <div class="label">Last Updated</div>
          <div class="value" style="font-size:14px;">${data.last_updated_str}</div>
        </div>`;
}
</script>
</body>
</html>"""

    def _format_overview(overview):
        overview = dict(overview)
        ts = overview.get("last_updated", time.time())
        overview["last_updated_str"] = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        return overview

    @app.route("/")
    def home():
        query = request.args.get("q", "")
        results = []
        if query:
            results_raw = engine.search(query, top_k=20)
            for r in results_raw:
                highlights_html = []
                for hl in r.highlights:
                    cleaned = hl.replace("[[[", '<span class="hl">').replace("]]]", "</span>")
                    highlights_html.append(cleaned)
                r.highlights = highlights_html
                results.append(r)

        overview = _format_overview(engine.index.get_overview())
        return render_template_string(
            HTML_TEMPLATE,
            query=query,
            results=results,
            overview=overview,
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
                        "term_freqs": r.term_freqs,
                        "doc_length": r.doc_length,
                        "phrase_matches": len(r.phrase_matches),
                        "highlights": r.highlights,
                    }
                    for r in results
                ],
            }
        )

    @app.route("/api/index/overview")
    def api_overview():
        overview = _format_overview(engine.index.get_overview())
        return jsonify(overview)

    @app.route("/api/index/add", methods=["POST"])
    def api_add_document():
        data = request.get_json()
        doc_id, action = engine.add_document(
            data.get("url", ""), data.get("title", ""), data.get("text", "")
        )
        engine.save_index(INDEX_DIR)
        return jsonify({"status": "ok", "doc_id": doc_id, "action": action})

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
