import re
import hashlib
import time
import logging
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CrawlResult:
    __slots__ = ("url", "title", "text", "links", "depth")

    def __init__(self, url, title, text, links, depth):
        self.url = url
        self.title = title
        self.text = text
        self.links = links
        self.depth = depth

    def to_dict(self):
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "links": self.links,
            "depth": self.depth,
        }


def _normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path
    if path.endswith("/"):
        path = path[:-1]
    if not path:
        path = "/"
    query = parsed.query
    qparts = sorted(query.split("&")) if query else []
    normalized = f"{scheme}://{netloc}{path}"
    if qparts:
        normalized += "?" + "&".join(qparts)
    return normalized


def _url_fingerprint(url):
    normalized = _normalize_url(url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Crawler:
    def __init__(
        self,
        max_depth=2,
        max_pages=100,
        max_per_domain=20,
        delay=1.0,
        timeout=10,
        user_agent="MiniSearchBot/1.0",
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_per_domain = max_per_domain
        self.delay = delay
        self.timeout = timeout
        self.user_agent = user_agent

        self._visited = {}
        self._domain_last_access = {}
        self._domain_count = {}
        self._robots_cache = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._session.headers.update(
            {"Accept": "text/html,application/xhtml+xml"}
        )
        self._session.headers.update(
            {"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
        )

    def _can_fetch(self, url):
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if robots_url not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                rp = None
            self._robots_cache[robots_url] = rp
        rp = self._robots_cache[robots_url]
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)

    def _polite_wait(self, domain):
        last = self._domain_last_access.get(domain, 0)
        elapsed = time.time() - last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._domain_last_access[domain] = time.time()

    def _extract_links(self, soup, base_url):
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            absolute = urljoin(base_url, href)
            absolute, _ = urldefrag(absolute)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            links.append(absolute)
        return links

    def _extract_content(self, soup):
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        body = soup.find("body")
        if body:
            text = body.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return title, text

    def fetch(self, url):
        if not self._can_fetch(url):
            logger.info("Blocked by robots.txt: %s", url)
            return None
        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
            resp.encoding = resp.apparent_encoding
            return resp.text
        except requests.RequestException as e:
            logger.warning("Fetch failed %s: %s", url, e)
            return None

    def crawl(self, seed_urls):
        results = []
        queue = deque()
        for url in seed_urls:
            norm = _normalize_url(url)
            fp = _url_fingerprint(norm)
            if fp not in self._visited:
                self._visited[fp] = norm
                queue.append((norm, 0))

        page_count = 0
        while queue and page_count < self.max_pages:
            url, depth = queue.popleft()
            if depth > self.max_depth:
                continue

            domain = urlparse(url).netloc
            domain_count = self._domain_count.get(domain, 0)
            if domain_count >= self.max_per_domain:
                continue

            self._polite_wait(domain)
            logger.info("Crawling (depth=%d): %s", depth, url)

            html = self.fetch(url)
            if html is None:
                continue

            soup = BeautifulSoup(html, "html.parser")
            title, text = self._extract_content(soup)
            if len(text) < 50:
                continue

            links = self._extract_links(soup, url)
            result = CrawlResult(url, title, text, links, depth)
            results.append(result)
            page_count += 1
            self._domain_count[domain] = domain_count + 1

            if depth < self.max_depth:
                for link in links:
                    norm_link = _normalize_url(link)
                    fp = _url_fingerprint(norm_link)
                    if fp not in self._visited:
                        self._visited[fp] = norm_link
                        queue.append((norm_link, depth + 1))

        logger.info("Crawl finished: %d pages fetched", len(results))
        return results
