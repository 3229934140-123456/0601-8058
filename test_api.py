import urllib.request, json, urllib.parse

def get(url):
    r = urllib.request.urlopen(url)
    return json.loads(r.read())

# Test 1: basic search
data = get('http://localhost:5000/api/search?q=Python')
print(f'Test 1 - Python search: {data["total"]} results')
print(f'  Top: {data["results"][0]["title"][:50]}')
print(f'  field_hits: {list(data["results"][0]["field_hits"].keys())}')
bd = data["results"][0]["score_breakdown"]
print(f'  bm25_body={bd["bm25_body"]:.3f} bm25_title={bd["bm25_title"]:.3f} bm25_url={bd["bm25_url"]:.3f}')
print()

# Test 2: title field search
data = get('http://localhost:5000/api/search?q=title%3APython')
print(f'Test 2 - title:Python: {data["total"]} results')
for res in data['results'][:3]:
    print(f'  - {res["title"][:50]} (score={res["score"]:.3f})')
print()

# Test 3: exclude term
data = get('http://localhost:5000/api/search?q=Python%20-tutorial')
print(f'Test 3 - Python -tutorial: {data["total"]} results')
has_tutorial = any('tutorial' in res['title'].lower() for res in data['results'])
print(f'  has tutorial page: {has_tutorial}')
print()

# Test 4: index overview
data = get('http://localhost:5000/api/index/overview')
print(f'Test 4 - Overview:')
print(f'  doc_count={data["doc_count"]}')
print(f'  term_count={data["term_count"]}')
print(f'  avg_doc_length={data["avg_doc_length"]:.1f}')
print(f'  snapshot_count={data.get("snapshot_count", 0)}')
print()

# Test 5: phrase search with breakdown
data = get('http://localhost:5000/api/search?q=%22machine%20learning%22')
print(f'Test 5 - Phrase "machine learning":')
if data['results']:
    bd = data['results'][0]['score_breakdown']
    print(f'  Top: {data["results"][0]["title"][:50]}')
    print(f'  bm25_body={bd["bm25_body"]:.3f} bm25_title={bd["bm25_title"]:.3f}')
    print(f'  phrase_boost={bd.get("phrase_boost", 0):.3f}')
    print(f'  proximity_boost={bd.get("proximity_boost", 0):.3f}')
    print(f'  must_boost={bd.get("must_boost", 0):.3f}')
print()

# Test 6: snapshot list
data = get('http://localhost:5000/api/index/snapshots')
print(f'Test 6 - Snapshots: {len(data.get("snapshots", []))} snapshots')
for s in data.get("snapshots", [])[:3]:
    print(f'  - #{s["id"]} {s.get("label", "auto")} ({s["doc_count"]} docs)')
print()

# Test 7: site filter (domain boundary)
data = get('http://localhost:5000/api/search?q=site%3Aexample.com%20Python')
print(f'Test 7 - site:example.com Python: {data["total"]} results')
all_example = True
for res in data['results']:
    url = res['url']
    domain = url.split('/')[2] if '://' in url else url.split('/')[0]
    is_example = domain == 'example.com' or domain.endswith('.example.com')
    if not is_example:
        all_example = False
        print(f'  WRONG DOMAIN: {domain} - {res["title"][:40]}')
print(f'  All on example.com domain: {all_example}')
print()

# Test 8: sort modes
print('Test 8 - Sort modes:')
for mode in ['combined', 'title_only', 'body_only']:
    data = get(f'http://localhost:5000/api/search?q=Python&sort={mode}')
    if data['results']:
        top_title = data['results'][0]['title'][:40]
        top_score = data['results'][0]['score']
        print(f'  {mode:12s} -> #1: {top_title}')
print()

# Test 9: query history
data = get('http://localhost:5000/api/query/history')
print(f'Test 9 - Query history: {len(data.get("history", []))} entries')
for h in data.get('history', [])[:5]:
    status = '✓' if h.get('has_results') else '✗'
    print(f'  {status} {h["query"]} ({h["result_count"]} results)')
print()

print('All 9 tests completed!')
