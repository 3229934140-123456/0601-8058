import json
import os
import logging
from collections import defaultdict

from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Document:
    __slots__ = ("doc_id", "url", "title", "length")

    def __init__(self, doc_id, url, title, length):
        self.doc_id = doc_id
        self.url = url
        self.title = title
        self.length = length

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["doc_id"], d["url"], d["title"], d["length"])


class InvertedIndex:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or Tokenizer()
        self.postings = defaultdict(lambda: defaultdict(list))
        self.documents = {}
        self._next_id = 0
        self._url_to_id = {}
        self._total_length = 0
        self._corpus_length = 0

    @property
    def doc_count(self):
        return len(self.documents)

    @property
    def avg_doc_length(self):
        if self.doc_count == 0:
            return 0
        return self._total_length / self.doc_count

    def get_term_freq(self, term, doc_id):
        positions = self.postings.get(term, {}).get(doc_id, [])
        return len(positions)

    def get_doc_freq(self, term):
        return len(self.postings.get(term, {}))

    def get_doc_length(self, doc_id):
        doc = self.documents.get(doc_id)
        return doc.length if doc else 0

    def add_document(self, url, title, text):
        if url in self._url_to_id:
            self.remove_document(self._url_to_id[url])

        doc_id = self._next_id
        self._next_id += 1

        tokens = self.tokenizer.tokenize(text)
        doc_length = len(tokens)

        for term, position in tokens:
            self.postings[term][doc_id].append(position)

        self.documents[doc_id] = Document(doc_id, url, title, doc_length)
        self._url_to_id[url] = doc_id
        self._total_length += doc_length
        self._corpus_length += 1

        logger.info(
            "Indexed doc %d: %s (%d tokens, %d unique terms)",
            doc_id, title[:30], doc_length, sum(1 for t in tokens),
        )
        return doc_id

    def remove_document(self, doc_id):
        if doc_id not in self.documents:
            return
        doc = self.documents[doc_id]
        terms_to_remove = []
        for term, doc_postings in self.postings.items():
            if doc_id in doc_postings:
                del doc_postings[doc_id]
                if not doc_postings:
                    terms_to_remove.append(term)

        for term in terms_to_remove:
            del self.postings[term]

        self._total_length -= doc.length
        self._corpus_length -= 1
        if doc.url in self._url_to_id:
            del self._url_to_id[doc.url]
        del self.documents[doc_id]
        logger.info("Removed doc %d from index", doc_id)

    def add_documents_batch(self, pages):
        doc_ids = []
        for page in pages:
            doc_id = self.add_document(page.url, page.title, page.text)
            doc_ids.append(doc_id)
        logger.info("Batch indexed %d documents, total: %d", len(doc_ids), self.doc_count)
        return doc_ids

    def intersect_postings(self, terms):
        if not terms:
            return set()
        posting_sets = []
        for term in terms:
            doc_ids = set(self.postings.get(term, {}).keys())
            if not doc_ids:
                return set()
            posting_sets.append(doc_ids)
        posting_sets.sort(key=len)
        result = posting_sets[0]
        for s in posting_sets[1:]:
            result = result & s
            if not result:
                return set()
        return result

    def union_postings(self, terms):
        result = set()
        for term in terms:
            doc_ids = self.postings.get(term, {}).keys()
            result.update(doc_ids)
        return result

    def get_positions(self, term, doc_id):
        return self.postings.get(term, {}).get(doc_id, [])

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        postings_serializable = {}
        for term, doc_map in self.postings.items():
            postings_serializable[term] = {
                str(doc_id): positions for doc_id, positions in doc_map.items()
            }
        with open(os.path.join(directory, "postings.json"), "w", encoding="utf-8") as f:
            json.dump(postings_serializable, f, ensure_ascii=False)

        docs_serializable = {
            str(doc_id): doc.to_dict() for doc_id, doc in self.documents.items()
        }
        with open(os.path.join(directory, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(docs_serializable, f, ensure_ascii=False)

        meta = {
            "next_id": self._next_id,
            "total_length": self._total_length,
            "corpus_length": self._corpus_length,
            "url_to_id": self._url_to_id,
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.info("Index saved to %s (%d docs, %d terms)", directory, self.doc_count, len(self.postings))

    def load(self, directory):
        with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._next_id = meta["next_id"]
        self._total_length = meta["total_length"]
        self._corpus_length = meta["corpus_length"]
        self._url_to_id = meta["url_to_id"]

        with open(os.path.join(directory, "documents.json"), "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        for doc_id_str, doc_dict in docs_data.items():
            doc_id = int(doc_id_str)
            self.documents[doc_id] = Document.from_dict(doc_dict)

        with open(os.path.join(directory, "postings.json"), "r", encoding="utf-8") as f:
            postings_data = json.load(f)
        self.postings = defaultdict(lambda: defaultdict(list))
        for term, doc_map in postings_data.items():
            for doc_id_str, positions in doc_map.items():
                self.postings[term][int(doc_id_str)] = positions

        logger.info("Index loaded from %s (%d docs, %d terms)", directory, self.doc_count, len(self.postings))
