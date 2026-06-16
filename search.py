import math
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict

from indexer import InvertedIndex, DEFAULT_FIELD_WEIGHTS

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    bm25_body: float = 0.0
    bm25_title: float = 0.0
    bm25_url: float = 0.0
    bm25_total: float = 0.0
    phrase_boost: float = 0.0
    proximity_boost: float = 0.0
    must_boost: float = 0.0
    total: float = 0.0
    per_term: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "bm25_body": round(self.bm25_body, 4),
            "bm25_title": round(self.bm25_title, 4),
            "bm25_url": round(self.bm25_url, 4),
            "bm25_total": round(self.bm25_total, 4),
            "phrase_boost": round(self.phrase_boost, 4),
            "proximity_boost": round(self.proximity_boost, 4),
            "must_boost": round(self.must_boost, 4),
            "total": round(self.total, 4),
            "per_term": {
                t: {k: round(v, 4) for k, v in d.items()}
                for t, d in self.per_term.items()
            },
        }


@dataclass
class SearchResult:
    doc_id: int
    url: str
    title: str
    score: float
    snippets: Dict = field(default_factory=dict)
    term_freqs: Dict = field(default_factory=dict)
    doc_length: int = 0
    phrase_matches: List = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    field_hits: Dict = field(default_factory=dict)
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    def __repr__(self):
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, title={self.title!r})"


class BM25FieldScorer:
    def __init__(self, index, field_weights=None, k1=1.8, b=0.75, delta=1.0):
        self.index = index
        self.field_weights = field_weights or DEFAULT_FIELD_WEIGHTS
        self.k1 = k1
        self.b = b
        self.delta = delta

    def _idf(self, term, field="body"):
        N = self.index.doc_count
        n = self.index.get_doc_freq(term, field)
        if n == 0:
            return 0.0
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    def score_term_field(self, term, doc_id, field):
        tf = self.index.get_term_freq(term, doc_id, field)
        if tf == 0:
            return 0.0
        doc_len = self.index.get_field_doc_length(doc_id, field)
        avgdl = self.index.get_field_avg_length(field)
        if avgdl == 0:
            return 0.0
        idf = self._idf(term, field)
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        return idf * (numerator / denominator + self.delta)

    def score_term(self, term, doc_id):
        total = 0.0
        per_field = {}
        for field, weight in self.field_weights.items():
            score = self.score_term_field(term, doc_id, field)
            per_field[field] = score
            total += weight * score
        return total, per_field

    def score_document(self, terms, doc_id):
        total = 0.0
        breakdown = ScoreBreakdown()
        for term in terms:
            term_total, per_field = self.score_term(term, doc_id)
            total += term_total
            breakdown.bm25_body += self.field_weights["body"] * per_field["body"]
            breakdown.bm25_title += self.field_weights["title"] * per_field["title"]
            breakdown.bm25_url += self.field_weights["url"] * per_field["url"]
            breakdown.per_term[term] = {
                "body": self.field_weights["body"] * per_field["body"],
                "title": self.field_weights["title"] * per_field["title"],
                "url": self.field_weights["url"] * per_field["url"],
                "total": term_total,
            }
        breakdown.bm25_total = total
        return total, breakdown


@dataclass
class ParsedQuery:
    must_terms: List[str] = field(default_factory=list)
    should_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
    phrases: List[List[str]] = field(default_factory=list)
    field_terms: Dict[str, List[str]] = field(default_factory=dict)
    site_terms: List[str] = field(default_factory=list)


class QueryParser:
    def __init__(self, tokenizer=None):
        from tokenizer import Tokenizer
        self.tokenizer = tokenizer or Tokenizer()

    def parse(self, query_string):
        must_terms = []
        should_terms = []
        exclude_terms = []
        field_terms = {"title": [], "url": [], "body": []}
        site_terms = []

        phrases = self.tokenizer.extract_phrases(query_string)
        remaining = self.tokenizer.remove_phrases(query_string)

        raw_parts = remaining.split()
        for part in raw_parts:
            if not part:
                continue

            field_match = re.match(r'(title|url|site|body):(.+)', part)
            if field_match:
                field_name = field_match.group(1)
                field_value = field_match.group(2)
                tokens = self.tokenizer.tokenize_query(field_value)
                if field_name == "site":
                    site_terms.extend(tokens)
                elif field_name in field_terms:
                    field_terms[field_name].extend(tokens)
                continue

            if part.startswith("+"):
                term_str = part[1:]
                if term_str:
                    tokenized = self.tokenizer.tokenize_query(term_str)
                    must_terms.extend(tokenized)
            elif part.startswith("-"):
                term_str = part[1:]
                if term_str:
                    tokenized = self.tokenizer.tokenize_query(term_str)
                    exclude_terms.extend(tokenized)
            else:
                tokenized = self.tokenizer.tokenize_query(part)
                should_terms.extend(tokenized)

        for phrase in phrases:
            must_terms.extend(phrase)

        return ParsedQuery(
            must_terms=must_terms,
            should_terms=should_terms,
            exclude_terms=exclude_terms,
            phrases=phrases,
            field_terms=field_terms,
            site_terms=site_terms,
        )


def _extract_snippets(text, terms, window_size=40, max_snippets=3):
    text_lower = text.lower()
    snippets = []
    found_positions = []

    for term in terms:
        start = 0
        term_lower = term.lower()
        while True:
            idx = text_lower.find(term_lower, start)
            if idx == -1:
                break
            found_positions.append((idx, idx + len(term)))
            start = idx + 1

    if not found_positions:
        cleaned = re.sub(r'\s+', ' ', text)
        if len(cleaned) > 250:
            snippets.append(cleaned[:250] + "...")
        else:
            snippets.append(cleaned)
        return snippets

    found_positions.sort(key=lambda x: x[0])
    used = [False] * len(found_positions)

    for i, (start, end) in enumerate(found_positions):
        if used[i]:
            continue
        window_start = max(0, start - window_size // 2)
        window_end = min(len(text), end + window_size // 2)

        snippet = text[window_start:window_end]
        snippet = re.sub(r'\s+', ' ', snippet).strip()
        if window_start > 0:
            snippet = "..." + snippet
        if window_end < len(text):
            snippet = snippet + "..."

        for j in range(len(found_positions)):
            s_j, e_j = found_positions[j]
            if s_j >= window_start - 5 and e_j <= window_end + 5:
                used[j] = True

        highlighted = snippet
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(lambda m: f"[[[{m.group()}]]]", highlighted)
        snippets.append(highlighted)

        if len(snippets) >= max_snippets:
            break

    return snippets


def _proximity_score(positions_map, max_distance=8):
    if len(positions_map) < 2:
        return 0.0

    terms = list(positions_map.keys())
    positions_list = [sorted(positions_map[t]) for t in terms]

    total_bonus = 0.0
    pair_count = 0

    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            pos_i = positions_list[i]
            pos_j = positions_list[j]

            if not pos_i or not pos_j:
                continue

            min_dist = float('inf')
            p1, p2 = 0, 0
            while p1 < len(pos_i) and p2 < len(pos_j):
                dist = abs(pos_i[p1] - pos_j[p2])
                min_dist = min(min_dist, dist)
                if pos_i[p1] < pos_j[p2]:
                    p1 += 1
                else:
                    p2 += 1

            if min_dist <= max_distance:
                proximity_bonus = 1.0 / (min_dist + 1) * 2.0
                total_bonus += proximity_bonus
                pair_count += 1

    if pair_count > 0:
        return total_bonus / pair_count * 3.0
    return 0.0


class SearchEngine:
    def __init__(self, index=None, scorer=None, tokenizer=None):
        from tokenizer import Tokenizer
        self.tokenizer = tokenizer or Tokenizer()
        self.index = index or InvertedIndex(tokenizer=self.tokenizer)
        self.scorer = scorer or BM25FieldScorer(self.index)
        self.query_parser = QueryParser(tokenizer=self.tokenizer)

    def search(self, query_string, top_k=10):
        parsed = self.query_parser.parse(query_string)

        must_terms = list(dict.fromkeys(parsed.must_terms))
        should_terms = list(dict.fromkeys(parsed.should_terms))
        exclude_terms = list(dict.fromkeys(parsed.exclude_terms))
        phrases = parsed.phrases
        field_terms = parsed.field_terms
        site_terms = parsed.site_terms

        all_terms = list(dict.fromkeys(must_terms + should_terms))
        for phrase in phrases:
            for t in phrase:
                if t not in all_terms:
                    all_terms.append(t)
        for field, terms in field_terms.items():
            for t in terms:
                if t not in all_terms:
                    all_terms.append(t)

        if not all_terms and not site_terms:
            return []

        candidate_docs = set()

        if site_terms:
            site_docs = self.index.site_filter(site_terms)
            candidate_docs.update(site_docs)

        if must_terms:
            must_docs = self.index.intersect_postings(must_terms, field="body")
            for field, terms in field_terms.items():
                if terms:
                    field_docs = self.index.intersect_postings(terms, field=field)
                    must_docs = must_docs & field_docs if must_docs else field_docs
            if candidate_docs:
                candidate_docs = candidate_docs & must_docs
            else:
                candidate_docs = must_docs
        elif should_terms:
            should_docs = self.index.union_postings(should_terms, field="body")
            for field, terms in field_terms.items():
                if terms:
                    field_docs = self.index.union_postings(terms, field=field)
                    should_docs = should_docs | field_docs
            if candidate_docs:
                candidate_docs = candidate_docs & should_docs
            else:
                candidate_docs = should_docs
        elif phrases:
            for phrase in phrases:
                phrase_docs = self.index.intersect_postings(phrase, field="body")
                candidate_docs.update(phrase_docs)
        else:
            has_field_terms = any(terms for terms in field_terms.values())
            if has_field_terms:
                field_candidates = set()
                for field, terms in field_terms.items():
                    if terms:
                        field_docs = self.index.union_postings(terms, field=field)
                        field_candidates.update(field_docs)
                if candidate_docs:
                    candidate_docs = candidate_docs & field_candidates
                else:
                    candidate_docs = field_candidates

        if exclude_terms:
            exclude_docs = self.index.union_postings(exclude_terms, field="body")
            candidate_docs = candidate_docs - exclude_docs

        results = []
        for doc_id in candidate_docs:
            doc = self.index.documents.get(doc_id)
            if not doc:
                continue

            base_score, breakdown = self.scorer.score_document(all_terms, doc_id)

            must_boost = 0.0
            for term in must_terms:
                term_score = breakdown.per_term.get(term, {}).get("total", 0)
                must_boost += term_score * 1.5
            breakdown.must_boost = must_boost

            phrase_score = 0.0
            phrase_matches = []
            for phrase in phrases:
                found, matches = self.index.check_phrase(phrase, doc_id, field="body")
                if found:
                    phrase_score += len(matches) * 6.0
                    phrase_matches.extend(matches)
                    for t in phrase:
                        if t not in breakdown.per_term:
                            tf = self.index.get_term_freq(t, doc_id, "body")
                            breakdown.per_term[t] = {
                                "body": 0,
                                "title": 0,
                                "url": 0,
                                "total": 0,
                                "tf": tf,
                            }
            breakdown.phrase_boost = phrase_score

            positions_map = {}
            for term in all_terms:
                positions = self.index.get_positions(term, doc_id, field="body")
                if positions:
                    positions_map[term] = positions
            proximity_bonus = _proximity_score(positions_map)
            breakdown.proximity_boost = proximity_bonus

            field_hit_info = {}
            for field in ["title", "body", "url"]:
                field_hit_terms = []
                for term in all_terms:
                    if self.index.get_term_freq(term, doc_id, field) > 0:
                        field_hit_terms.append(term)
                if field_hit_terms:
                    field_hit_info[field] = field_hit_terms

            total_score = base_score + phrase_score + proximity_bonus + must_boost
            breakdown.total = total_score

            term_freqs = {}
            for term in all_terms:
                tf = self.index.get_term_freq(term, doc_id, "body")
                ttf = self.index.get_term_freq(term, doc_id, "title")
                utf = self.index.get_term_freq(term, doc_id, "url")
                if tf > 0 or ttf > 0 or utf > 0:
                    term_freqs[term] = {"body": tf, "title": ttf, "url": utf}

            if total_score > 0:
                snippets = {}
                for term in all_terms:
                    positions = self.index.get_positions(term, doc_id)
                    if positions:
                        snippets[term] = positions[:10]

                highlights = _extract_snippets(doc.text, all_terms)

                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        url=doc.url,
                        title=doc.title,
                        score=total_score,
                        snippets=snippets,
                        term_freqs=term_freqs,
                        doc_length=doc.length,
                        phrase_matches=phrase_matches,
                        highlights=highlights,
                        field_hits=field_hit_info,
                        score_breakdown=breakdown,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def add_document(self, url, title, text):
        doc_id, action = self.index.add_document(url, title, text)
        return doc_id, action

    def add_documents_batch(self, pages):
        return self.index.add_documents_batch(pages)

    def create_snapshot(self, label=""):
        return self.index.create_snapshot(label)

    def rollback_snapshot(self, snapshot_id):
        return self.index.rollback_snapshot(snapshot_id)

    def list_snapshots(self):
        return self.index.list_snapshots()

    def save_index(self, directory):
        self.index.save(directory)

    def load_index(self, directory):
        self.index.load(directory)
        self.scorer = BM25FieldScorer(self.index)
