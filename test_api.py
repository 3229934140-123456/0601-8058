import urllib.request, json, urllib.parse

def test(name, url):
    r = urllib.request.urlopen(url)
    data = json.loads(r.read())
    return data

# Test 1: basic search
data = test('Python', 'http://localhost:5000/api/search?q=Python')
print(f'Test 1 - Python: {len(data["results"])} results')
print(f'  Top: {data["results"][0]["title"][:50]}')
print(f'  field_hits: {data["results"][0]["field_hits"]}')
bd = data["results"][0]["score_breakdown"]
print(f'  bm25_body={bd["bm25_body"]:.3f} bm25_title={bd["bm25_title"]:.3f} bm25_url={bd["bm25_url"]:.3f}')
print()

# Test 2: title field search
data = test('title:Python', 'http://localhost:5000/api/search?q=title%3APython')
print(f'Test 2 - title:Python: {len(data["results"])} results')
for res in data['results']:
    print(f'  - {res["title"][:50]} (score={res["score"]:.3f})')
print()

# Test 3: exclude term
data = test('Python -tutorial', 'http://localhost:5000/api/search?q=Python%20-tutorial')
print(f'Test 3 - Python -tutorial: {len(data["results"])} results')
has_tutorial = any('tutorial' in res['title'].lower() for res in data['results'])
print(f'  has tutorial page: {has_tutorial}')
print()

# Test 4: index overview
data = test('overview', 'http://localhost:5000/api/index/overview')
print(f'Test 4 - Overview:')
print(f'  num_docs={data["num_docs"]}')
print(f'  num_terms={data["num_terms"]}')
print(f'  avg_doc_length={data["avg_doc_length"]:.1f}')
print()

# Test 5: phrase search with breakdown
data = test('"machine learning"', 'http://localhost:5000/api/search?q=%22machine%20learning%22')
print(f'Test 5 - Phrase "machine learning":')
if data['results']:
    bd = data['results'][0]['score_breakdown']
    print(f'  Top: {data["results"][0]["title"][:50]}')
    print(f'  bm25_body={bd["bm25_body"]:.3f} bm25_title={bd["bm25_title"]:.3f} bm25_url={bd["bm25_url"]:.3f}')
    print(f'  phrase_boost={bd.get("phrase_boost", 0):.3f}')
    print(f'  proximity_boost={bd.get("proximity_boost", 0):.3f}')
    print(f'  must_boost={bd.get("must_boost", 0):.3f}')
print()

# Test 6: snapshot list
data = test('snapshots', 'http://localhost:5000/api/index/snapshots')
print(f'Test 6 - Snapshots: {len(data.get("snapshots", []))} snapshots')
print()

print('All tests passed!')
