import re
import logging
from collections import Counter

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

_EN_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:['_-][a-zA-Z0-9]+)*")
_PHRASE_RE = re.compile(r'"([^"]+)"')


class Tokenizer:
    def __init__(self, use_stop_words=True, min_len=1):
        self.use_stop_words = use_stop_words
        self.min_len = min_len
        jieba.setLogLevel(logging.WARNING)

    def _process_token(self, seg):
        seg = seg.strip()
        if not seg:
            return None
        term = seg.lower()
        if len(term) < self.min_len:
            return None
        if self.use_stop_words and term in STOP_WORDS:
            return None
        return term

    def tokenize(self, text):
        tokens = []
        position = 0

        for seg in jieba.cut(text):
            en_matches = _EN_WORD_RE.findall(seg)
            if en_matches:
                for word in en_matches:
                    term = self._process_token(word)
                    if term is not None:
                        tokens.append((term, position))
                        position += 1
            else:
                term = self._process_token(seg)
                if term is not None:
                    tokens.append((term, position))
                    position += 1

        return tokens

    def tokenize_with_original(self, text):
        tokens = []
        original_tokens = []
        position = 0

        for seg in jieba.cut(text):
            en_matches = _EN_WORD_RE.findall(seg)
            if en_matches:
                for word in en_matches:
                    term = self._process_token(word)
                    if term is not None:
                        tokens.append((term, position))
                        original_tokens.append(word)
                        position += 1
            else:
                term = self._process_token(seg)
                if term is not None:
                    tokens.append((term, position))
                    original_tokens.append(seg)
                    position += 1

        return tokens, original_tokens

    def tokenize_query(self, query):
        tokens = []
        position = 0

        for seg in jieba.cut(query):
            en_matches = _EN_WORD_RE.findall(seg)
            if en_matches:
                for word in en_matches:
                    term = self._process_token(word)
                    if term is not None:
                        tokens.append(term)
                        position += 1
            else:
                term = self._process_token(seg)
                if term is not None:
                    tokens.append(term)
                    position += 1

        return tokens

    def count_term_freq(self, text):
        tokens = self.tokenize(text)
        counter = Counter()
        for term, _ in tokens:
            counter[term] += 1
        return counter

    def extract_phrases(self, query_string):
        phrases = _PHRASE_RE.findall(query_string)
        phrase_tokens_list = []
        for phrase in phrases:
            tokens = self.tokenize_query(phrase)
            if tokens:
                phrase_tokens_list.append(tokens)
        return phrase_tokens_list

    def remove_phrases(self, query_string):
        return _PHRASE_RE.sub("", query_string).strip()
