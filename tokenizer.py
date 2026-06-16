import re
import math
import logging

import jieba

logger = logging.getLogger(__name__)

STOP_WORDS_EN = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "not", "no",
    "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "if", "then", "than", "so", "very", "just",
    "about", "up", "out", "into", "over", "after", "also", "more",
    "most", "other", "some", "such", "only", "own", "same", "than",
    "too", "s", "t", "don", "didn", "doesn", "isn", "aren", "wasn",
    "weren", "hasn", "haven", "hadn", "won", "wouldn", "couldn",
    "shouldn", "mustn", "needn", "ll", "ve", "re", "d", "m",
}

STOP_WORDS_ZH = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
    "们", "那", "这个", "那个", "什么", "吗", "呢", "吧", "啊",
    "哦", "嗯", "么", "把", "被", "让", "给", "对", "而", "但",
    "却", "又", "还", "已", "已经", "于", "与", "从", "向", "为",
    "因为", "所以", "如果", "虽然", "可以", "能", "得", "地",
}

STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_ZH

_EN_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


class Tokenizer:
    def __init__(self, use_stop_words=True, min_len=1):
        self.use_stop_words = use_stop_words
        self.min_len = min_len
        jieba.setLogLevel(logging.WARNING)

    def tokenize(self, text):
        tokens = []
        position = 0
        chunks = _EN_WORD_RE.split(text)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            segs = jieba.cut(chunk)
            for seg in segs:
                seg = seg.strip()
                if not seg:
                    continue
                term = seg.lower()
                if len(term) < self.min_len:
                    continue
                if self.use_stop_words and term in STOP_WORDS:
                    continue
                tokens.append((term, position))
                position += 1
        en_matches = _EN_WORD_RE.findall(text)
        for word in en_matches:
            term = word.lower()
            if len(term) < self.min_len:
                continue
            if self.use_stop_words and term in STOP_WORDS:
                continue
            tokens.append((term, position))
            position += 1

        seen = {}
        deduped = []
        for term, pos in tokens:
            if term not in seen:
                seen[term] = True
                deduped.append((term, pos))
        deduped.sort(key=lambda x: x[1])
        return deduped

    def tokenize_query(self, query):
        tokens = []
        position = 0
        parts = _EN_WORD_RE.split(query)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            for seg in jieba.cut(part):
                seg = seg.strip()
                if not seg:
                    continue
                term = seg.lower()
                if len(term) < self.min_len:
                    continue
                if self.use_stop_words and term in STOP_WORDS:
                    continue
                tokens.append(term)
                position += 1
        for word in _EN_WORD_RE.findall(query):
            term = word.lower()
            if len(term) < self.min_len:
                continue
            if self.use_stop_words and term in STOP_WORDS:
                continue
            tokens.append(term)
            position += 1

        seen = {}
        unique = []
        for t in tokens:
            if t not in seen:
                seen[t] = True
                unique.append(t)
        return unique
