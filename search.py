import math
import logging
from dataclasses import dataclass, field
from typing import List

from indexer import InvertedIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    doc_id: int
    url: str
    title: str
    score: float
    snippets: dict = field(default_factory=dict)

    def __repr__(self):
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, title={self.title!r})"


class BM25:
    def __init__(self, index, k1=1.5, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b

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
        return idf * numerator / denominator

    def score_document(self, terms, doc_id):
        return sum(self.score_term(term, doc_id) for term in terms)


@dataclass
class ParsedQuery:
    must_terms: List[str] = field(default_factory=list)
    should_terms: List[str] = field(default_factory=list)


class QueryParser:
    def parse(self, query_string):
        must_terms = []
        should_terms = []
        raw_tokens = query_string.strip().split()
        for token in raw_tokens:
            if token.startswith("+"):
                term = token[1:].strip().lower()
                if term:
                    must_terms.append(term)
            elif token.startswith("-"):
                continue
            else:
                term = token.strip().lower()
                if term:
                    should_terms.append(term)
        return ParsedQuery(must_terms=must_terms, should_terms=should_terms)


class SearchEngine:
    def __init__(self, index=None, bm25=None, tokenizer=None):
        from tokenizer import Tokenizer
        self.index = index or InvertedIndex()
        self.bm25 = bm25 or BM25(self.index)
        self.tokenizer = tokenizer or Tokenizer()
        self.query_parser = QueryParser()

    def search(self, query_string, top_k=10):
        parsed = self.query_parser.parse(query_string)

        must_terms = []
        for term in parsed.must_terms:
            tokenized = self.tokenizer.tokenize_query(term)
            must_terms.extend(tokenized)

        should_terms = []
        for term in parsed.should_terms:
            tokenized = self.tokenizer.tokenize_query(term)
            should_terms.extend(tokenized)

        must_terms = list(dict.fromkeys(must_terms))
        should_terms = list(dict.fromkeys(should_terms))

        all_terms = list(dict.fromkeys(must_terms + should_terms))
        if not all_terms:
            return []

        if must_terms:
            candidate_docs = self.index.intersect_postings(must_terms)
        else:
            candidate_docs = self.index.union_postings(should_terms)

        if not candidate_docs and not must_terms:
            candidate_docs = self.index.union_postings(all_terms)

        results = []
        for doc_id in candidate_docs:
            score = 0.0
            for term in all_terms:
                term_score = self.bm25.score_term(term, doc_id)
                if term in must_terms:
                    term_score *= 2.0
                score += term_score

            if score > 0:
                doc = self.index.documents.get(doc_id)
                if doc:
                    snippets = {}
                    for term in all_terms:
                        positions = self.index.get_positions(term, doc_id)
                        if positions:
                            snippets[term] = positions[:5]
                    results.append(
                        SearchResult(
                            doc_id=doc_id,
                            url=doc.url,
                            title=doc.title,
                            score=score,
                            snippets=snippets,
                        )
                    )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def add_document(self, url, title, text):
        return self.index.add_document(url, title, text)

    def add_documents_batch(self, pages):
        return self.index.add_documents_batch(pages)

    def save_index(self, directory):
        self.index.save(directory)

    def load_index(self, directory):
        self.index.load(directory)
        self.bm25 = BM25(self.index)
