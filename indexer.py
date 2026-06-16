import json
import os
import hashlib
import time
import copy
import logging
import re
from collections import defaultdict, Counter
from urllib.parse import urlparse

from tokenizer import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_FIELD_WEIGHTS = {
    "title": 3.0,
    "body": 1.0,
    "url": 2.0,
}


class Document:
    __slots__ = (
        "doc_id", "url", "title", "text",
        "length", "content_hash", "updated_at",
        "field_lengths", "field_term_freq",
    )

    def __init__(
        self,
        doc_id,
        url,
        title,
        text,
        length,
        content_hash,
        updated_at=None,
        field_lengths=None,
        field_term_freq=None,
    ):
        self.doc_id = doc_id
        self.url = url
        self.title = title
        self.text = text
        self.length = length
        self.content_hash = content_hash
        self.updated_at = updated_at if updated_at is not None else time.time()
        self.field_lengths = field_lengths or {"title": 0, "body": 0, "url": 0}
        self.field_term_freq = field_term_freq or {"title": {}, "body": {}, "url": {}}

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "length": self.length,
            "content_hash": self.content_hash,
            "updated_at": self.updated_at,
            "field_lengths": self.field_lengths,
            "field_term_freq": self.field_term_freq,
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
            d.get("field_lengths", {"title": 0, "body": 0, "url": 0}),
            d.get("field_term_freq", {"title": {}, "body": {}, "url": {}}),
        )


def _content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_domain(url):
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        if parsed.scheme:
            return parsed.hostname or ""
        if "/" in url:
            url = url.split("/")[0]
        return url.lower()
    except Exception:
        return ""


def domain_matches(domain, pattern):
    if not domain or not pattern:
        return False
    domain = domain.lower()
    pattern = pattern.lower()
    if domain == pattern:
        return True
    if domain.endswith("." + pattern):
        return True
    return False


class InvertedIndex:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or Tokenizer()
        self.field_postings = {
            "title": defaultdict(lambda: defaultdict(list)),
            "body": defaultdict(lambda: defaultdict(list)),
            "url": defaultdict(lambda: defaultdict(list)),
        }
        self.documents = {}
        self._next_id = 0
        self._url_to_id = {}
        self._field_total_length = {"title": 0, "body": 0, "url": 0}
        self._corpus_length = 0
        self._last_updated = time.time()
        self._snapshots = []
        self._snapshot_dir = None

    @property
    def doc_count(self):
        return len(self.documents)

    @property
    def term_count(self):
        all_terms = set()
        for field in self.field_postings:
            all_terms.update(self.field_postings[field].keys())
        return len(all_terms)

    @property
    def avg_doc_length(self):
        if self.doc_count == 0:
            return 0
        return self._field_total_length["body"] / self.doc_count

    @property
    def last_updated(self):
        return self._last_updated

    def get_field_avg_length(self, field):
        if self.doc_count == 0:
            return 0
        return self._field_total_length.get(field, 0) / self.doc_count

    def get_term_freq(self, term, doc_id, field="body"):
        positions = self.field_postings[field].get(term, {}).get(doc_id, [])
        return len(positions)

    def get_doc_freq(self, term, field="body"):
        return len(self.field_postings[field].get(term, {}))

    def get_doc_length(self, doc_id):
        doc = self.documents.get(doc_id)
        return doc.length if doc else 0

    def get_field_doc_length(self, doc_id, field):
        doc = self.documents.get(doc_id)
        if doc is None:
            return 0
        return doc.field_lengths.get(field, 0)

    def get_content_hash(self, url):
        doc_id = self._url_to_id.get(url)
        if doc_id is None:
            return None
        return self.documents[doc_id].content_hash

    def _tokenize_field(self, text, field_name):
        tokens = self.tokenizer.tokenize(text)
        term_freq = Counter()
        for term, pos in tokens:
            term_freq[term] += 1
        return tokens, dict(term_freq), len(tokens)

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

        body_tokens, body_tf, body_len = self._tokenize_field(text, "body")
        title_tokens, title_tf, title_len = self._tokenize_field(title, "title")
        url_tokens, url_tf, url_len = self._tokenize_field(url, "url")

        for term, position in body_tokens:
            self.field_postings["body"][term][doc_id].append(position)
        for term, position in title_tokens:
            self.field_postings["title"][term][doc_id].append(position)
        for term, position in url_tokens:
            self.field_postings["url"][term][doc_id].append(position)

        total_length = body_len

        field_lengths = {"title": title_len, "body": body_len, "url": url_len}
        field_term_freq = {"title": title_tf, "body": body_tf, "url": url_tf}

        doc = Document(
            doc_id=doc_id,
            url=url,
            title=title,
            text=text,
            length=total_length,
            content_hash=content_hash,
            field_lengths=field_lengths,
            field_term_freq=field_term_freq,
        )
        self.documents[doc_id] = doc
        self._url_to_id[url] = doc_id
        self._field_total_length["title"] += title_len
        self._field_total_length["body"] += body_len
        self._field_total_length["url"] += url_len
        self._corpus_length += 1
        self._last_updated = time.time()

        unique_body_terms = len(body_tf)
        logger.info(
            "Indexed doc %d: %s (body=%d tokens/%d unique, action=%s)",
            doc_id, title[:30], body_len, unique_body_terms, action,
        )
        return doc_id, action

    def remove_document(self, doc_id):
        if doc_id not in self.documents:
            return
        doc = self.documents[doc_id]

        for field in self.field_postings:
            terms_to_remove = []
            for term, doc_postings in self.field_postings[field].items():
                if doc_id in doc_postings:
                    del doc_postings[doc_id]
                    if not doc_postings:
                        terms_to_remove.append(term)
            for term in terms_to_remove:
                del self.field_postings[field][term]

        self._field_total_length["title"] -= doc.field_lengths.get("title", 0)
        self._field_total_length["body"] -= doc.field_lengths.get("body", 0)
        self._field_total_length["url"] -= doc.field_lengths.get("url", 0)
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

    def intersect_postings(self, terms, field="body"):
        if not terms:
            return set()
        posting_sets = []
        for term in terms:
            doc_ids = set(self.field_postings[field].get(term, {}).keys())
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

    def union_postings(self, terms, field="body"):
        result = set()
        for term in terms:
            doc_ids = self.field_postings[field].get(term, {}).keys()
            result.update(doc_ids)
        return result

    def get_positions(self, term, doc_id, field="body"):
        return self.field_postings[field].get(term, {}).get(doc_id, [])

    def check_phrase(self, phrase_tokens, doc_id, field="body"):
        if not phrase_tokens:
            return False, []
        positions_list = []
        for term in phrase_tokens:
            positions = self.get_positions(term, doc_id, field)
            if not positions:
                return False, []
            positions_list.append(positions)

        matches = []
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

    def site_filter(self, site_terms, doc_ids=None):
        if not site_terms:
            return doc_ids if doc_ids is not None else set()

        if doc_ids is None:
            doc_ids = set(self.documents.keys())

        matched = set()
        for doc_id in doc_ids:
            doc = self.documents.get(doc_id)
            if doc is None:
                continue
            domain = extract_domain(doc.url)
            for site_term in site_terms:
                if domain_matches(domain, site_term):
                    matched.add(doc_id)
                    break
        return matched

    def set_snapshot_dir(self, directory):
        self._snapshot_dir = directory
        if directory:
            os.makedirs(directory, exist_ok=True)
            self._scan_snapshots()

    def _snapshot_filename(self, snap_id, timestamp):
        ts = int(timestamp)
        return f"snapshot_{snap_id:04d}_{ts}.json"

    def _scan_snapshots(self):
        if not self._snapshot_dir or not os.path.isdir(self._snapshot_dir):
            return
        pattern = re.compile(r"^snapshot_(\d+)_(\d+)\.json$")
        files = []
        for fname in os.listdir(self._snapshot_dir):
            m = pattern.match(fname)
            if m:
                files.append((int(m.group(1)), int(m.group(2)), fname))
        files.sort(key=lambda x: x[0])

        self._snapshots = []
        for snap_id, ts, fname in files:
            fpath = os.path.join(self._snapshot_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._snapshots.append({
                    "id": snap_id,
                    "timestamp": ts,
                    "label": meta.get("label", ""),
                    "doc_count": meta.get("doc_count", 0),
                    "term_count": meta.get("term_count", 0),
                    "filename": fname,
                    "from_file": True,
                })
            except Exception:
                logger.warning("Failed to load snapshot %s", fname)

        if self._snapshots:
            self._next_snapshot_id = self._snapshots[-1]["id"] + 1
        else:
            self._next_snapshot_id = 0

    def _snapshot_to_dict(self):
        return {
            "label": "",
            "doc_count": self.doc_count,
            "term_count": self.term_count,
            "data": {
                "field_postings": {
                    field: {
                        term: {str(did): pos for did, pos in doc_map.items()}
                        for term, doc_map in postings.items()
                    }
                    for field, postings in self.field_postings.items()
                },
                "documents": {str(k): v.to_dict() for k, v in self.documents.items()},
                "next_id": self._next_id,
                "url_to_id": dict(self._url_to_id),
                "field_total_length": dict(self._field_total_length),
                "corpus_length": self._corpus_length,
                "last_updated": self._last_updated,
            },
        }

    def _load_snapshot_data(self, snap_id):
        snap = None
        for s in self._snapshots:
            if s["id"] == snap_id:
                snap = s
                break
        if snap is None:
            return None

        if "data" in snap:
            return snap["data"]

        if self._snapshot_dir and snap.get("from_file") and snap.get("filename"):
            fpath = os.path.join(self._snapshot_dir, snap["filename"])
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    sd = json.load(f)
                return sd.get("data")
            except Exception:
                logger.warning("Failed to load snapshot file %s", fpath)
        return None

    def create_snapshot(self, label=""):
        snap_id = self._next_snapshot_id
        timestamp = time.time()

        snapshot = {
            "id": snap_id,
            "timestamp": timestamp,
            "label": label,
            "doc_count": self.doc_count,
            "term_count": self.term_count,
        }

        if self._snapshot_dir:
            snap_data = self._snapshot_to_dict()
            snap_data["label"] = label
            fname = self._snapshot_filename(snap_id, timestamp)
            fpath = os.path.join(self._snapshot_dir, fname)
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(snap_data, f, ensure_ascii=False)
                snapshot["filename"] = fname
                snapshot["from_file"] = True
            except Exception:
                logger.error("Failed to save snapshot to %s", fpath)
        else:
            snapshot["data"] = self._snapshot_to_dict()["data"]

        self._snapshots.append(snapshot)
        self._next_snapshot_id += 1

        logger.info("Created snapshot %d: %s (%d docs)", snap_id, label or "auto", self.doc_count)
        return snap_id

    def delete_snapshot(self, snapshot_id):
        idx = None
        for i, s in enumerate(self._snapshots):
            if s["id"] == snapshot_id:
                idx = i
                break
        if idx is None:
            return False

        snap = self._snapshots[idx]
        if self._snapshot_dir and snap.get("filename"):
            fpath = os.path.join(self._snapshot_dir, snap["filename"])
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception:
                logger.warning("Failed to delete snapshot file %s", fpath)

        del self._snapshots[idx]
        logger.info("Deleted snapshot %d", snapshot_id)
        return True

    def rollback_snapshot(self, snapshot_id):
        data = self._load_snapshot_data(snapshot_id)
        if data is None:
            return False

        idx = None
        for i, s in enumerate(self._snapshots):
            if s["id"] == snapshot_id:
                idx = i
                break
        if idx is None:
            return False
        snap = self._snapshots[idx]

        self.field_postings = {}
        for field in ["title", "body", "url"]:
            self.field_postings[field] = defaultdict(lambda: defaultdict(list))
        for field, postings in data.get("field_postings", {}).items():
            if field not in self.field_postings:
                continue
            for term, doc_map in postings.items():
                for doc_id_str, positions in doc_map.items():
                    self.field_postings[field][term][int(doc_id_str)] = list(positions)

        self.documents = {}
        for doc_id_str, doc_dict in data.get("documents", {}).items():
            self.documents[int(doc_id_str)] = Document.from_dict(doc_dict)

        self._next_id = data["next_id"]
        self._url_to_id = dict(data["url_to_id"])
        self._field_total_length = dict(data["field_total_length"])
        self._corpus_length = data["corpus_length"]
        self._last_updated = data["last_updated"]

        self._snapshots = self._snapshots[: idx + 1]
        self._next_snapshot_id = snapshot_id + 1

        logger.info(
            "Rolled back to snapshot %d (%s), %d docs",
            snapshot_id, snap.get("label") or "auto", self.doc_count,
        )
        return True

    def export_snapshot(self, snapshot_id, output_path):
        data = self._load_snapshot_data(snapshot_id)
        if data is None:
            return False
        snap = None
        for s in self._snapshots:
            if s["id"] == snapshot_id:
                snap = s
                break

        export_data = {
            "version": 1,
            "snapshot_id": snapshot_id,
            "label": snap.get("label", "") if snap else "",
            "timestamp": snap.get("timestamp", time.time()) if snap else time.time(),
            "doc_count": snap.get("doc_count", self.doc_count) if snap else self.doc_count,
            "term_count": snap.get("term_count", self.term_count) if snap else self.term_count,
            "data": data,
        }

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error("Failed to export snapshot: %s", e)
            return False

    def import_snapshot(self, input_path, label=None):
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                sd = json.load(f)
        except Exception as e:
            logger.error("Failed to import snapshot: %s", e)
            return None

        if "data" not in sd:
            logger.error("Invalid snapshot file: missing data")
            return None

        snap_id = self._next_snapshot_id
        timestamp = sd.get("timestamp", time.time())
        doc_count = sd.get("doc_count", 0)
        term_count = sd.get("term_count", 0)
        snap_label = label if label is not None else sd.get("label", "imported")

        snapshot = {
            "id": snap_id,
            "timestamp": timestamp,
            "label": snap_label,
            "doc_count": doc_count,
            "term_count": term_count,
        }

        if self._snapshot_dir:
            fname = self._snapshot_filename(snap_id, timestamp)
            fpath = os.path.join(self._snapshot_dir, fname)
            export_data = {
                "label": snap_label,
                "doc_count": doc_count,
                "term_count": term_count,
                "data": sd["data"],
            }
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False)
                snapshot["filename"] = fname
                snapshot["from_file"] = True
            except Exception:
                logger.error("Failed to save imported snapshot to %s", fpath)
                snapshot["data"] = sd["data"]
        else:
            snapshot["data"] = sd["data"]

        self._snapshots.append(snapshot)
        self._next_snapshot_id += 1

        logger.info("Imported snapshot %d: %s", snap_id, snap_label)
        return snap_id

    def list_snapshots(self):
        return [
            {
                "id": s["id"],
                "timestamp": s["timestamp"],
                "label": s["label"],
                "doc_count": s["doc_count"],
                "term_count": s["term_count"],
            }
            for s in self._snapshots
        ]

    def get_overview(self):
        return {
            "doc_count": self.doc_count,
            "term_count": self.term_count,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "last_updated": self._last_updated,
            "total_tokens_body": self._field_total_length["body"],
            "total_tokens_title": self._field_total_length["title"],
            "total_tokens_url": self._field_total_length["url"],
            "snapshot_count": len(self._snapshots),
        }

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)

        postings_serializable = {}
        for field, postings in self.field_postings.items():
            postings_serializable[field] = {}
            for term, doc_map in postings.items():
                postings_serializable[field][term] = {
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
            "field_total_length": self._field_total_length,
            "corpus_length": self._corpus_length,
            "url_to_id": self._url_to_id,
            "last_updated": self._last_updated,
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.info("Index saved to %s (%d docs, %d terms)", directory, self.doc_count, self.term_count)

    def load(self, directory):
        with open(os.path.join(directory, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._next_id = meta["next_id"]
        self._field_total_length = meta.get(
            "field_total_length", {"title": 0, "body": 0, "url": 0}
        )
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

        if isinstance(next(iter(postings_data.values())) if postings_data else {}, dict):
            self.field_postings = {}
            for field in ["title", "body", "url"]:
                self.field_postings[field] = defaultdict(lambda: defaultdict(list))
            for field, field_data in postings_data.items():
                if field not in self.field_postings:
                    continue
                for term, doc_map in field_data.items():
                    for doc_id_str, positions in doc_map.items():
                        self.field_postings[field][term][int(doc_id_str)] = positions
        else:
            self.field_postings = {
                "title": defaultdict(lambda: defaultdict(list)),
                "body": defaultdict(lambda: defaultdict(list)),
                "url": defaultdict(lambda: defaultdict(list)),
            }
            for term, doc_map in postings_data.items():
                for doc_id_str, positions in doc_map.items():
                    self.field_postings["body"][term][int(doc_id_str)] = positions

        logger.info("Index loaded from %s (%d docs, %d terms)", directory, self.doc_count, self.term_count)
