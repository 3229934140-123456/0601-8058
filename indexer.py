import json
import os
import hashlib
import time
import logging
from collections import defaultdict, Counter

from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Document:
    __slots__ = ("doc_id", "url", "title", "text", "length", "content_hash", "updated_at", "term_freq")

    def __init__(self, doc_id, url, title, text, length, content_hash, updated_at=None, term_freq=None):
        self.doc_id = doc_id
        self.url = url
        self.title = title
        self.text = text
        self.length = length
        self.content_hash = content_hash
        self.updated_at = updated_at if updated_at is not None else time.time()
        self.term_freq = term_freq if term_freq is not None else {}

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "length": self.length,
            "content_hash": self.content_hash,
            "updated_at": self.updated_at,
            "term_freq": self.term_freq,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["doc_id"],
            d["url"],
            d["title"],
            d.get("text", ""),
            d["length"],
            d.get("content_hash", ""),
            d.get("updated_at", time.time()),
            d.get("term_freq", {}),
        )


def _content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class InvertedIndex:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or Tokenizer()
        self.postings = defaultdict(lambda: defaultdict(list))
        self.documents = {}
        self._next_id = 0
        self._url_to_id = {}
        self._total_length = 0
        self._corpus_length = 0
        self._last_updated = time.time()

    @property
    def doc_count(self):
        return len(self.documents)

    @property
    def term_count(self):
        return len(self.postings)

    @property
    def avg_doc_length(self):
        if self.doc_count == 0:
            return 0
        return self._total_length / self.doc_count

    @property
    def last_updated(self):
        return self._last_updated

    def get_term_freq(self, term, doc_id):
        positions = self.postings.get(term, {}).get(doc_id, [])
        return len(positions)

    def get_doc_freq(self, term):
        return len(self.postings.get(term, {}))

    def get_doc_length(self, doc_id):
        doc = self.documents.get(doc_id)
        return doc.length if doc else 0

    def get_content_hash(self, url):
        doc_id = self._url_to_id.get(url)
        if doc_id is None:
            return None
        return self.documents[doc_id].content_hash

    def add_document(self, url, title, text):
        content_hash = _content_hash(text)
        existing_id = self._url_to_id.get(url)

        if existing_id is not None:
            existing_doc = self.documents[existing_id]
            if existing_doc.content_hash == content_hash:
                logger.debug("Document unchanged, skipping: %s", url)
                return existing_id, "skipped"
            else:
                logger.info("Document updated, reindexing: %s", url)
                self.remove_document(existing_id)
                action = "updated"
        else:
            action = "added"

        doc_id = self._next_id
        self._next_id += 1

        tokens, original_tokens = self.tokenizer.tokenize_with_original(text)
        doc_length = len(tokens)

        term_freq = Counter()
        for term, position in tokens:
            self.postings[term][doc_id].append(position)
            term_freq[term] += 1

        doc = Document(
            doc_id=doc_id,
            url=url,
            title=title,
            text=text,
            length=doc_length,
            content_hash=content_hash,
            term_freq=dict(term_freq),
        )
        self.documents[doc_id] = doc
        self._url_to_id[url] = doc_id
        self._total_length += doc_length
        self._corpus_length += 1
        self._last_updated = time.time()

        unique_terms = sum(1 for t in term_freq)
        logger.info(
            "Indexed doc %d: %s (%d tokens, %d unique terms, action=%s)",
            doc_id, title[:30], doc_length, unique_terms, action,
        )
        return doc_id, action

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
        self._last_updated = time.time()
        logger.debug("Removed doc %d from index", doc_id)

    def add_documents_batch(self, pages):
        doc_ids = []
        stats = {"added": 0, "updated": 0, "skipped": 0}
        for page in pages:
            doc_id, action = self.add_document(page.url, page.title, page.text)
            doc_ids.append(doc_id)
            stats[action] = stats.get(action, 0) + 1
        logger.info(
            "Batch indexed: added=%d, updated=%d, skipped=%d, total docs=%d",
            stats["added"], stats["updated"], stats["skipped"], self.doc_count,
        )
        return doc_ids, stats

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

    def check_phrase(self, phrase_tokens, doc_id):
        if not phrase_tokens:
            return False, []
        positions_list = []
        for term in phrase_tokens:
            positions = self.get_positions(term, doc_id)
            if not positions:
                return False, []
            positions_list.append(positions)

        matches = []
        first_term = phrase_tokens[0]
        for start_pos in positions_list[0]:
            match = True
            phrase_positions = [start_pos]
            for i in range(1, len(phrase_tokens)):
                expected = start_pos + i
                if expected not in positions_list[i]:
                    match = False
                    break
                phrase_positions.append(expected)
            if match:
                matches.append(phrase_positions)

        return len(matches) > 0, matches

    def get_overview(self):
        return {
            "doc_count": self.doc_count,
            "term_count": self.term_count,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "last_updated": self._last_updated,
            "total_tokens": self._total_length,
        }

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
            "last_updated": self._last_updated,
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
        self._last_updated = meta.get("last_updated", time.time())

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
