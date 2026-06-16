import math
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from indexer import InvertedIndex

logger = logging.getLogger(__name__)


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

    def __repr__(self):
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, title={self.title!r})"


class BM25:
    def __init__(self, index, k1=1.8, b=0.75, delta=1.0):
        self.index = index
        self.k1 = k1
        self.b = b
        self.delta = delta

    def _idf(self, term):
        N = self.index.doc_count
        n = self.index.get_doc_freq(term)
        if n == 0:
            return 0.0
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    def score_term(self, term, doc_id):
        tf = self.index.get_term_freq(term, doc_id)
        if tf == 0:
            return 0.0
        doc_len = self.index.get_doc_length(doc_id)
        avgdl = self.index.avg_doc_length
        if avgdl == 0:
            return 0.0
        idf = self._idf(term)
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        return idf * (numerator / denominator + self.delta)

    def score_document(self, terms, doc_id):
        return sum(self.score_term(term, doc_id) for term in terms)


@dataclass
class ParsedQuery:
    must_terms: List[str] = field(default_factory=list)
    should_terms: List[str] = field(default_factory=list)
    exclude_terms: List[str] = field(default_factory=list)
    phrases: List[List[str]] = field(default_factory=list)


class QueryParser:
    def __init__(self, tokenizer=None):
        from tokenizer import Tokenizer
        self.tokenizer = tokenizer or Tokenizer()

    def parse(self, query_string):
        must_terms = []
        should_terms = []
        exclude_terms = []

        phrases = self.tokenizer.extract_phrases(query_string)
        remaining = self.tokenizer.remove_phrases(query_string)

        raw_tokens = remaining.split()
        for token in raw_tokens:
            if token.startswith("+"):
                term = token[1:].strip()
                if term:
                    tokenized = self.tokenizer.tokenize_query(term)
                    must_terms.extend(tokenized)
            elif token.startswith("-"):
                term = token[1:].strip()
                if term:
                    tokenized = self.tokenizer.tokenize_query(term)
                    exclude_terms.extend(tokenized)
            else:
                term = token.strip()
                if term:
                    tokenized = self.tokenizer.tokenize_query(term)
                    should_terms.extend(tokenized)

        phrase_must = []
        for phrase in phrases:
            phrase_must.extend(phrase)

        return ParsedQuery(
            must_terms=must_terms,
            should_terms=should_terms,
            exclude_terms=exclude_terms,
            phrases=phrases,
        )


def _extract_snippets(text, terms, window_size=30, max_snippets=3):
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
        if len(cleaned) > 200:
            snippets.append(cleaned[:200] + "...")
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
        return total_bonus / pair_count
    return 0.0


class SearchEngine:
    def __init__(self, index=None, bm25=None, tokenizer=None):
        from tokenizer import Tokenizer
        self.tokenizer = tokenizer or Tokenizer()
        self.index = index or InvertedIndex(tokenizer=self.tokenizer)
        self.bm25 = bm25 or BM25(self.index)
        self.query_parser = QueryParser(tokenizer=self.tokenizer)

    def search(self, query_string, top_k=10):
        parsed = self.query_parser.parse(query_string)

        must_terms = list(dict.fromkeys(parsed.must_terms))
        should_terms = list(dict.fromkeys(parsed.should_terms))
        exclude_terms = list(dict.fromkeys(parsed.exclude_terms))
        phrases = parsed.phrases

        all_terms = list(dict.fromkeys(must_terms + should_terms))
        for phrase in phrases:
            for t in phrase:
                if t not in all_terms:
                    all_terms.append(t)

        if not all_terms and not phrases:
            return []

        candidate_docs = set()

        if must_terms:
            candidate_docs = self.index.intersect_postings(must_terms)
        elif should_terms:
            candidate_docs = self.index.union_postings(should_terms)
        elif phrases:
            for phrase in phrases:
                phrase_docs = self.index.intersect_postings(phrase)
                candidate_docs.update(phrase_docs)

        for phrase in phrases:
            phrase_docs = self.index.intersect_postings(phrase)
            if must_terms or should_terms:
                candidate_docs = candidate_docs & phrase_docs if candidate_docs else phrase_docs
            else:
                candidate_docs = candidate_docs | phrase_docs

        if exclude_terms:
            exclude_docs = self.index.union_postings(exclude_terms)
            candidate_docs = candidate_docs - exclude_docs

        results = []
        for doc_id in candidate_docs:
            doc = self.index.documents.get(doc_id)
            if not doc:
                continue

            base_score = 0.0
            term_freqs = {}
            positions_map = {}

            for term in all_terms:
                tf = self.index.get_term_freq(term, doc_id)
                if tf > 0:
                    term_score = self.bm25.score_term(term, doc_id)
                    if term in must_terms:
                        term_score *= 2.5
                    base_score += term_score
                    term_freqs[term] = tf
                    positions_map[term] = self.index.get_positions(term, doc_id)

            phrase_score = 0.0
            phrase_matches = []
            for phrase in phrases:
                found, matches = self.index.check_phrase(phrase, doc_id)
                if found:
                    phrase_score += len(matches) * 5.0
                    phrase_matches.extend(matches)
                    for t in phrase:
                        if t not in term_freqs:
                            term_freqs[t] = self.index.get_term_freq(t, doc_id)

            proximity_bonus = _proximity_score(positions_map)
            rare_term_bonus = 0.0
            for term in all_terms:
                if term in term_freqs:
                    df = self.index.get_doc_freq(term)
                    if df > 0:
                        idf_component = math.log((self.index.doc_count + 1) / (df + 1))
                        rare_term_bonus += idf_component * 0.1

            total_score = base_score + phrase_score + proximity_bonus + rare_term_bonus

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
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def add_document(self, url, title, text):
        doc_id, action = self.index.add_document(url, title, text)
        return doc_id, action

    def add_documents_batch(self, pages):
        return self.index.add_documents_batch(pages)

    def save_index(self, directory):
        self.index.save(directory)

    def load_index(self, directory):
        self.index.load(directory)
        self.bm25 = BM25(self.index)
