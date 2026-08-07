"""
Storynory scraper — downloads MP3 audio + text transcript pairs for training.

Usage:
    python scrape_storynory.py --output raw_data/ --max-stories 100
    python scrape_storynory.py --output raw_data/  # scrape everything

Saves matching pairs:
    raw_data/<slug>.mp3
    raw_data/<slug>.txt

Already-downloaded stories are skipped automatically (safe to re-run).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://storynory.com"
BASE_URL_WWW = "https://www.storynory.com"

# Storynory uses a 3-level structure:
#   1. Archive page  → lists series (category) pages
#   2. Series page   → lists individual story pages (paginated)
#   3. Story page    → has MP3 + text transcript
ARCHIVE_URLS = {
    "original":    "https://storynory.com/archives/original-stories-for-children/",
    "fairy-tales": "https://storynory.com/archives/fairy-tales/",
    "myths":       "https://storynory.com/archives/myths-world-stories/",
    "educational": "https://storynory.com/archives/educational-stories/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; educational-tts-research/1.0; "
        "children-story audio+text dataset)"
    )
}

DELAY_BETWEEN_PAGES   = 1.5   # seconds between listing page fetches
DELAY_BETWEEN_STORIES = 2.0   # seconds between individual story fetches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  [HTTP error] {e} — retrying in {wait}s")
            time.sleep(wait)
    print(f"  [SKIP] Failed after {retries} attempts: {url}")
    return None


def _slug_from_url(url: str) -> str:
    """Extract a filesystem-safe slug from a story URL."""
    path = urlparse(url).path.strip("/")
    slug = path.split("/")[-1]
    slug = re.sub(r"[^\w\-]", "_", slug)
    return slug[:120]  # cap length


def _is_storynory(url: str) -> bool:
    """True for any storynory.com URL (with or without www)."""
    host = urlparse(url).netloc.lower().lstrip("www.")
    return host == "storynory.com"


def _is_story_url(url: str) -> bool:
    """
    Individual story pages sit at the root: storynory.com/<slug>/
    Series/category/archive pages have multi-segment paths or known prefixes.
    """
    parts = [p for p in urlparse(url).path.split("/") if p]
    non_story = {"category", "archives", "tag", "author", "page", "feed",
                 "wp-content", "wp-includes", "wp-admin", "cart", "shop"}
    return len(parts) == 1 and parts[0] not in non_story


# ---------------------------------------------------------------------------
# Story link discovery  (3-level: archive → series → individual story)
# ---------------------------------------------------------------------------

def _get_series_links(archive_url: str) -> list[str]:
    """Fetch an archive page and return all series (category) URLs listed on it."""
    resp = _get(archive_url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    series: list[str] = []
    for a in soup.find_all("a", href=True):
        # Resolve relative URLs (e.g. /category/...) to absolute
        href = urljoin(archive_url, a["href"])
        if not _is_storynory(href):
            continue
        parts = [p for p in urlparse(href).path.split("/") if p]
        # Series pages look like /category/<genre>/<series-name>/
        if parts and parts[0] == "category" and href not in series:
            series.append(href)
    return series


def _get_story_links_from_series(series_url: str) -> list[str]:
    """
    Paginate through a series page and return all individual story URLs.
    Follows /page/N/ pagination automatically.
    """
    links: list[str] = []
    page_url = series_url

    while page_url:
        print(f"    [Series page] {page_url}")
        resp = _get(page_url)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = urljoin(series_url, a["href"])
            if _is_storynory(href) and _is_story_url(href) and href not in links:
                links.append(href)

        # Next page link
        next_tag = (
            soup.find("a", class_=re.compile(r"next", re.I)) or
            soup.find("a", string=re.compile(r"next|›|»", re.I))
        )
        next_url = next_tag["href"] if next_tag else None
        if next_url and next_url != page_url:
            next_url = urljoin(series_url, next_url)
            if _is_storynory(next_url):
                page_url = next_url
                time.sleep(DELAY_BETWEEN_PAGES)
                continue
        page_url = None

    return links


def _collect_story_links(archive_url: str) -> list[str]:
    """
    Full two-step crawl: archive page → series pages → individual story URLs.
    """
    series_links = _get_series_links(archive_url)
    print(f"  Found {len(series_links)} series under {archive_url}")

    all_stories: list[str] = []
    seen: set[str] = set()
    for series_url in series_links:
        stories = _get_story_links_from_series(series_url)
        for url in stories:
            if url not in seen:
                seen.add(url)
                all_stories.append(url)
        time.sleep(DELAY_BETWEEN_PAGES)

    return all_stories


# ---------------------------------------------------------------------------
# Individual story scraping
# ---------------------------------------------------------------------------

def _extract_mp3_url(soup: BeautifulSoup, page_url: str) -> str | None:
    """Find the MP3 URL on a story page."""

    # 1. <audio> tag with src or child <source src="...">
    for audio_tag in soup.find_all("audio"):
        src = audio_tag.get("src", "")
        if src.lower().endswith(".mp3"):
            return urljoin(page_url, src)
        for source in audio_tag.find_all("source"):
            src = source.get("src", "")
            if src.lower().endswith(".mp3"):
                return urljoin(page_url, src)

    # 2. Direct <a href="...mp3"> links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".mp3"):
            return urljoin(page_url, href)

    # 3. Data attributes (some WP audio players use data-url or data-source)
    for tag in soup.find_all(True):
        for attr in ("data-url", "data-source", "data-audio", "data-mp3"):
            val = tag.get(attr, "")
            if val.lower().endswith(".mp3"):
                return urljoin(page_url, val)

    # 4. Inline JS: look for quoted .mp3 URLs in <script> tags
    for script in soup.find_all("script"):
        if script.string:
            match = re.search(r'["\']([^"\']+\.mp3)["\']', script.string)
            if match:
                return urljoin(page_url, match.group(1))

    return None


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract the story transcript text from a story page."""
    # Remove audio players, nav, sidebar, ads
    for unwanted in soup.select(
        "script, style, nav, header, footer, aside, .sidebar, "
        ".widget, .sharedaddy, .jp-relatedposts, .navigation, "
        "audio, iframe, .audio-player, .wp-audio-shortcode, "
        ".post-meta, .entry-meta, .tags, .categories"
    ):
        unwanted.decompose()

    # Try common WordPress content containers (most specific first)
    candidates = [
        soup.select_one(".entry-content"),
        soup.select_one(".post-content"),
        soup.select_one(".story-content"),
        soup.select_one("article .content"),
        soup.select_one("article"),
        soup.select_one("main"),
    ]

    content_div = next((c for c in candidates if c is not None), None)
    if content_div is None:
        return ""

    # Extract paragraphs only — avoids pulling in menu text, etc.
    paragraphs = [p.get_text(separator=" ", strip=True)
                  for p in content_div.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if len(p) > 30)
    return text.strip()


def scrape_story(url: str, output_dir: Path) -> bool:
    """
    Download audio and extract text for one story.
    Returns True on success, False if skipped or failed.
    """
    slug = _slug_from_url(url)
    mp3_path = output_dir / f"{slug}.mp3"
    txt_path = output_dir / f"{slug}.txt"

    # Skip if both files already exist
    if mp3_path.exists() and txt_path.exists():
        print(f"  [skip]  {slug} (already downloaded)")
        return False

    resp = _get(url)
    if resp is None:
        return False

    soup = BeautifulSoup(resp.text, "html.parser")

    # -- Text --
    if not txt_path.exists():
        text = _extract_text(soup)
        if len(text) < 200:
            print(f"  [SKIP]  {slug} — text too short ({len(text)} chars)")
            return False
        txt_path.write_text(text, encoding="utf-8")
        print(f"  [text]  {slug}.txt ({len(text):,} chars)")

    # -- Audio --
    if not mp3_path.exists():
        mp3_url = _extract_mp3_url(soup, url)
        if mp3_url is None:
            print(f"  [SKIP]  {slug} — no MP3 found on page")
            return False
        audio_resp = _get(mp3_url)
        if audio_resp is None:
            return False
        mp3_path.write_bytes(audio_resp.content)
        size_mb = len(audio_resp.content) / 1_048_576
        print(f"  [audio] {slug}.mp3 ({size_mb:.1f} MB)")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Storynory children's stories: MP3 + text transcript pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_storynory.py --output raw_data/
  python scrape_storynory.py --output raw_data/ --max-stories 50
  python scrape_storynory.py --output raw_data/ --categories fairy-tales myths
        """,
    )
    parser.add_argument(
        "--output", default="raw_data",
        help="Directory to save MP3 + TXT pairs (default: raw_data/)",
    )
    parser.add_argument(
        "--max-stories", type=int, default=None,
        help="Stop after downloading this many NEW stories (default: no limit)",
    )
    parser.add_argument(
        "--categories", nargs="+",
        choices=["original", "fairy-tales", "myths", "educational"],
        default=["original", "fairy-tales", "myths", "educational"],
        help="Which Storynory categories to scrape",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all story URLs across chosen categories
    all_links: list[str] = []
    seen: set[str] = set()
    for cat in args.categories:
        cat_url = ARCHIVE_URLS[cat]
        print(f"\n[Category] {cat} → {cat_url}")
        links = _collect_story_links(cat_url)
        for link in links:
            if link not in seen:
                seen.add(link)
                all_links.append(link)

    print(f"\n[Scraper] Found {len(all_links)} unique story URLs")

    downloaded = 0
    for i, url in enumerate(all_links):
        if args.max_stories and downloaded >= args.max_stories:
            print(f"\n[Scraper] Reached --max-stories={args.max_stories}, stopping.")
            break
        print(f"\n[{i + 1}/{len(all_links)}] {url}")
        success = scrape_story(url, output_dir)
        if success:
            downloaded += 1
        time.sleep(DELAY_BETWEEN_STORIES)

    print(f"\n[Scraper] Done. {downloaded} new stories downloaded to {output_dir}/")
    print(f"[Scraper] Run next:")
    print(f"  python data_pipeline.py --input {output_dir}/ --output data/")


if __name__ == "__main__":
    main()
