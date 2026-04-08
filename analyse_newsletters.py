"""
Ennismore Fund Management Newsletter Stock Performance Analyser
===============================================================
Scrapes newsletters from ennismorefunds.com, extracts stock tickers
from PDF commentary, then analyses 2-year price performance from
the date of first mention.

Usage:
    python analyse_newsletters.py [--output-dir OUTPUT_DIR] [--max-newsletters N]

Output:
    - results/stock_performance.csv  – per-ticker performance table
    - results/performance_chart.png  – bar chart of 2-year returns
    - results/mention_log.csv        – every mention with newsletter date
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEWSLETTER_PAGES = [
    {
        "fund": "Global Equity Fund",
        "url": "https://ennismorefunds.com/global-equity-fund/newsletters",
    },
    {
        "fund": "European Smaller Companies Fund",
        "url": "https://ennismorefunds.com/european-smaller-companies-fund/newsletters",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Regex patterns for extracting ticker symbols from PDF text.
# Ennismore newsletters typically write: "Company Name (TICKER)" or
# reference tickers in stock-comment headers.
TICKER_PATTERNS = [
    # "Company Name (TICKER)" – most common format
    re.compile(r"(?<!\w)\(([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\)(?!\w)"),
    # Explicit tag formats: "ticker: AAPL" or "Ticker AAPL"
    re.compile(r"(?i)\bticker[:\s]+([A-Z]{1,5}(?:\.[A-Z]{1,2})?)"),
    # Dollar-prefixed: $AAPL
    re.compile(r"\$([A-Z]{1,5})"),
]

# Words that look like tickers but are common English abbreviations / units
TICKER_BLACKLIST = {
    "A", "I", "IT", "US", "UK", "EU", "CEO", "CFO", "COO", "CTO",
    "PE", "PB", "EV", "FCF", "NAV", "AUM", "IPO", "EPS", "GDP",
    "CPI", "ESG", "AI", "DM", "EM", "ETF", "TV", "Q1", "Q2", "Q3", "Q4",
    "H1", "H2", "FY", "YOY", "QOQ", "MOM", "EBIT", "EBITDA", "NET",
    "LTM", "NTM", "ROE", "ROA", "RoC", "LBO", "DCF", "IRR", "NPV",
    "AND", "OR", "NOT", "THE", "FOR", "IN", "OF", "ON", "AT", "BY",
    "NEW", "OLD", "LOW", "HIGH", "MAX", "MIN", "AVG", "YTD", "MTD",
    "USD", "GBP", "EUR", "CHF", "SEK", "NOK", "DKK", "JPY", "HKD",
    "AUD", "CAD", "NZD", "SGD", "CNY", "BPS", "BP", "PL", "P",
    "AGM", "LLC", "PLC", "LTD", "INC", "SA", "AG", "NV", "AB",
}

# ---------------------------------------------------------------------------
# 1. Newsletter discovery (Playwright)
# ---------------------------------------------------------------------------

def discover_newsletter_pdfs(pages: list[dict], max_newsletters: int | None = None) -> list[dict]:
    """
    Use Playwright to load each fund's newsletter listing page and
    collect all PDF links with their associated publication dates.

    Returns a list of dicts:
        {"fund": str, "date": datetime, "url": str, "title": str}
    """
    from playwright.sync_api import sync_playwright

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="en-GB",
        )
        page = context.new_page()

        for fund_info in pages:
            fund_name = fund_info["fund"]
            fund_url = fund_info["url"]
            print(f"\n[Scraper] Loading {fund_url}")

            try:
                page.goto(fund_url, wait_until="networkidle", timeout=30_000)
                # Let any JS-rendered content settle
                page.wait_for_timeout(2_000)
            except Exception as exc:
                print(f"[Scraper] Warning: could not load page – {exc}")
                continue

            # Extract all PDF links on the page
            pdf_links = page.evaluate("""
                () => {
                    const links = [];
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.href || '';
                        if (href.toLowerCase().includes('.pdf')) {
                            links.push({
                                href: href,
                                text: a.textContent.trim(),
                                title: a.getAttribute('title') || ''
                            });
                        }
                    });
                    return links;
                }
            """)

            if not pdf_links:
                # Fallback: grab all links and filter
                all_links = page.evaluate("""
                    () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                        href: a.href, text: a.textContent.trim()
                    }))
                """)
                pdf_links = [l for l in all_links if ".pdf" in l.get("href", "").lower()]

            print(f"[Scraper] Found {len(pdf_links)} PDF links for {fund_name}")

            for link in pdf_links:
                href = link.get("href", "")
                text = link.get("text", "") or link.get("title", "")
                pub_date = _parse_date_from_url_or_text(href, text)
                if pub_date is None:
                    continue
                results.append({
                    "fund": fund_name,
                    "date": pub_date,
                    "url": href,
                    "title": text or href.split("/")[-1],
                })

        browser.close()

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    # Sort chronologically
    unique.sort(key=lambda x: x["date"])

    if max_newsletters:
        unique = unique[:max_newsletters]

    print(f"\n[Scraper] Total unique newsletters discovered: {len(unique)}")
    return unique


def _parse_date_from_url_or_text(url: str, text: str) -> datetime | None:
    """
    Attempt to extract a publication date from the PDF URL or link text.
    Handles patterns like:
      - "October-2025", "October_2025", "October 2025"
      - "2025-10", "Oct-2025"
      - Legacy: "NL-EGF-March-2021"
    """
    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    combined = f"{url} {text}".lower().replace("---", " ").replace("-", " ").replace("_", " ")

    # Try "Month Year" pattern
    for month_str, month_num in MONTHS.items():
        pattern = re.compile(rf"\b{month_str}\b\s*(\d{{4}})")
        m = pattern.search(combined)
        if m:
            year = int(m.group(1))
            if 2010 <= year <= datetime.now().year + 1:
                return datetime(year, month_num, 1)

    # Try YYYY-MM or YYYY MM
    m = re.search(r"\b(20\d{2})\s*(\d{2})\b", combined)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return datetime(year, month, 1)

    return None


# ---------------------------------------------------------------------------
# 2. PDF download & stock ticker extraction
# ---------------------------------------------------------------------------

def download_pdf(url: str, cache_dir: Path) -> Path | None:
    """Download a PDF to cache_dir; return local path or None on failure."""
    filename = re.sub(r"[^\w\-.]", "_", url.split("/")[-1])
    local_path = cache_dir / filename

    if local_path.exists():
        return local_path

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        time.sleep(0.5)  # polite crawl delay
        return local_path
    except Exception as exc:
        print(f"  [PDF] Failed to download {url}: {exc}")
        return None


def extract_tickers_from_pdf(pdf_path: Path) -> list[tuple[str, str]]:
    """
    Parse a PDF and return a list of (ticker, context_sentence) tuples.
    Applies all TICKER_PATTERNS and filters via TICKER_BLACKLIST.
    """
    found = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                # Split into sentences for context capture
                sentences = re.split(r"(?<=[.!?])\s+", text)
                for sentence in sentences:
                    for pattern in TICKER_PATTERNS:
                        for match in pattern.finditer(sentence):
                            ticker = match.group(1).upper()
                            if ticker in TICKER_BLACKLIST:
                                continue
                            if len(ticker) < 2:
                                continue
                            # Capture context (up to 200 chars around the match)
                            start = max(0, match.start() - 80)
                            end = min(len(sentence), match.end() + 80)
                            context = sentence[start:end].strip()
                            found.append((ticker, context))
    except Exception as exc:
        print(f"  [PDF] Error reading {pdf_path.name}: {exc}")

    # Deduplicate within this PDF while preserving first-occurrence context
    seen: dict[str, str] = {}
    for ticker, context in found:
        if ticker not in seen:
            seen[ticker] = context
    return list(seen.items())


# ---------------------------------------------------------------------------
# 3. Build mention log
# ---------------------------------------------------------------------------

def build_mention_log(newsletters: list[dict], cache_dir: Path) -> pd.DataFrame:
    """
    For each newsletter, download the PDF and extract tickers.
    Returns a DataFrame with columns:
        ticker, mention_date, fund, newsletter_title, newsletter_url, context
    """
    rows = []

    for i, nl in enumerate(newsletters, 1):
        print(f"[{i}/{len(newsletters)}] {nl['fund']} – {nl['date'].strftime('%B %Y')} – {nl['title'][:60]}")
        pdf_path = download_pdf(nl["url"], cache_dir)
        if pdf_path is None:
            print("  Skipped (download failed)")
            continue

        tickers = extract_tickers_from_pdf(pdf_path)
        print(f"  Extracted {len(tickers)} tickers: {[t for t, _ in tickers]}")

        for ticker, context in tickers:
            rows.append({
                "ticker": ticker,
                "mention_date": nl["date"],
                "fund": nl["fund"],
                "newsletter_title": nl["title"],
                "newsletter_url": nl["url"],
                "context": context,
            })

    df = pd.DataFrame(rows, columns=[
        "ticker", "mention_date", "fund", "newsletter_title",
        "newsletter_url", "context",
    ])
    return df


# ---------------------------------------------------------------------------
# 4. Price performance analysis
# ---------------------------------------------------------------------------

def _get_price_on_date(ticker_obj: yf.Ticker, target_date: datetime) -> float | None:
    """
    Retrieve the closing price nearest to target_date (within ±5 trading days).
    """
    start = target_date - timedelta(days=7)
    end = target_date + timedelta(days=7)
    try:
        hist = ticker_obj.history(start=start.strftime("%Y-%m-%d"),
                                   end=end.strftime("%Y-%m-%d"))
        if hist.empty:
            return None
        # Find closest date
        hist.index = hist.index.tz_localize(None) if hist.index.tzinfo else hist.index
        target_ts = pd.Timestamp(target_date)
        closest_idx = (hist.index - target_ts).abs().argmin()
        return float(hist["Close"].iloc[closest_idx])
    except Exception:
        return None


def analyse_performance(mention_log: pd.DataFrame, today: datetime) -> pd.DataFrame:
    """
    For each ticker's *first* mention date, look up:
      - price at mention date
      - price 2 years later (or today if < 2 years have elapsed)
      - % return over that period
      - whether the 2-year window is complete

    Returns a summary DataFrame sorted by 2-year return (descending).
    """
    # Only keep the first mention per ticker
    first_mentions = (
        mention_log.sort_values("mention_date")
        .drop_duplicates(subset="ticker", keep="first")
        .reset_index(drop=True)
    )

    results = []

    for _, row in first_mentions.iterrows():
        ticker = row["ticker"]
        mention_date = row["mention_date"]
        two_year_date = mention_date + timedelta(days=730)
        window_complete = two_year_date <= today
        analysis_end = two_year_date if window_complete else today

        print(f"[Price] {ticker:8s}  first mention: {mention_date.strftime('%Y-%m-%d')}  "
              f"  end: {analysis_end.strftime('%Y-%m-%d')}{'*' if not window_complete else ''}")

        try:
            tk = yf.Ticker(ticker)
            price_start = _get_price_on_date(tk, mention_date)
            price_end = _get_price_on_date(tk, analysis_end)

            if price_start and price_end and price_start > 0:
                pct_return = (price_end - price_start) / price_start * 100
            else:
                pct_return = None
        except Exception as exc:
            print(f"  yfinance error for {ticker}: {exc}")
            price_start = price_end = pct_return = None

        results.append({
            "ticker": ticker,
            "first_mention_date": mention_date.strftime("%Y-%m-%d"),
            "fund": row["fund"],
            "newsletter": row["newsletter_title"],
            "price_at_mention": round(price_start, 4) if price_start else None,
            "price_at_end": round(price_end, 4) if price_end else None,
            "analysis_end_date": analysis_end.strftime("%Y-%m-%d"),
            "window_complete_2yr": window_complete,
            "return_pct": round(pct_return, 2) if pct_return is not None else None,
            "context_snippet": row["context"][:200],
        })

    df = pd.DataFrame(results)
    if not df.empty and "return_pct" in df.columns:
        df = df.sort_values("return_pct", ascending=False, na_position="last")
    return df


# ---------------------------------------------------------------------------
# 5. Report generation
# ---------------------------------------------------------------------------

def save_csv_reports(performance: pd.DataFrame, mention_log: pd.DataFrame, output_dir: Path) -> None:
    perf_path = output_dir / "stock_performance.csv"
    log_path = output_dir / "mention_log.csv"
    performance.to_csv(perf_path, index=False)
    mention_log.to_csv(log_path, index=False)
    print(f"\n[Report] Saved performance table → {perf_path}")
    print(f"[Report] Saved mention log        → {log_path}")


def plot_performance(performance: pd.DataFrame, output_dir: Path) -> None:
    """Generate a horizontal bar chart of 2-year returns per ticker."""
    df = performance.dropna(subset=["return_pct"]).copy()
    if df.empty:
        print("[Report] No price data available – skipping chart.")
        return

    df = df.sort_values("return_pct")
    colors = ["#d62728" if r < 0 else "#2ca02c" for r in df["return_pct"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.45)))
    bars = ax.barh(df["ticker"], df["return_pct"], color=colors, edgecolor="white", linewidth=0.5)

    # Annotate bars
    for bar, val in zip(bars, df["return_pct"]):
        x_pos = bar.get_width() + (1 if val >= 0 else -1)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Return (%)")
    ax.set_title(
        "Stock Performance: 2-Year Return from First Newsletter Mention\n"
        "Ennismore Fund Management",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=9)

    # Footnote for incomplete windows
    incomplete = performance[performance["window_complete_2yr"] == False]["ticker"].tolist()
    if incomplete:
        fig.text(
            0.01, 0.01,
            f"* Incomplete 2-year window (data to today): {', '.join(incomplete)}",
            fontsize=7, color="grey",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    chart_path = output_dir / "performance_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[Report] Saved chart               → {chart_path}")


def print_summary(performance: pd.DataFrame) -> None:
    """Print a readable summary table to stdout."""
    df = performance.copy()
    if df.empty:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 80)
    print("ENNISMORE NEWSLETTER STOCK PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Ticker':<10} {'First Mention':<15} {'Fund':<30} {'Start $':>8} {'End $':>8} {'Return':>8} {'Complete':>9}")
    print("-" * 80)
    for _, row in df.iterrows():
        complete_marker = "Yes" if row["window_complete_2yr"] else "No*"
        ret_str = f"{row['return_pct']:+.1f}%" if pd.notna(row["return_pct"]) else "N/A"
        start_str = f"{row['price_at_mention']:.2f}" if pd.notna(row.get("price_at_mention")) else "N/A"
        end_str = f"{row['price_at_end']:.2f}" if pd.notna(row.get("price_at_end")) else "N/A"
        fund_short = row["fund"][:28]
        print(f"{row['ticker']:<10} {row['first_mention_date']:<15} {fund_short:<30} {start_str:>8} {end_str:>8} {ret_str:>8} {complete_marker:>9}")
    print("-" * 80)
    valid = df.dropna(subset=["return_pct"])
    if not valid.empty:
        avg = valid["return_pct"].mean()
        median = valid["return_pct"].median()
        winners = (valid["return_pct"] > 0).sum()
        print(f"\nAverage return : {avg:+.1f}%")
        print(f"Median return  : {median:+.1f}%")
        print(f"Winners (>0%)  : {winners}/{len(valid)}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_mention_log_from_local_pdfs(pdf_dir: Path) -> pd.DataFrame:
    """
    Fallback: parse all PDFs in pdf_dir directly, inferring the newsletter
    date from the filename.  Filenames should contain a month and year, e.g.
        Ennismore-Global-Equity-Fund---October-2023.pdf
        NL-EGF-March-2021.pdf
    """
    rows = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pub_date = _parse_date_from_url_or_text(pdf_path.name, "")
        if pub_date is None:
            print(f"  [LocalPDF] Cannot infer date from filename: {pdf_path.name} – skipping")
            continue

        # Guess fund name from filename
        name_lower = pdf_path.name.lower()
        if "global" in name_lower:
            fund = "Global Equity Fund"
        elif "european" in name_lower or "escf" in name_lower:
            fund = "European Smaller Companies Fund"
        else:
            fund = "Unknown Fund"

        print(f"  [LocalPDF] {fund} – {pub_date.strftime('%B %Y')} – {pdf_path.name}")
        tickers = extract_tickers_from_pdf(pdf_path)
        print(f"    Extracted {len(tickers)} tickers: {[t for t, _ in tickers]}")

        for ticker, context in tickers:
            rows.append({
                "ticker": ticker,
                "mention_date": pub_date,
                "fund": fund,
                "newsletter_title": pdf_path.name,
                "newsletter_url": str(pdf_path),
                "context": context,
            })

    return pd.DataFrame(rows, columns=[
        "ticker", "mention_date", "fund", "newsletter_title",
        "newsletter_url", "context",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Analyse stock price performance following Ennismore newsletter mentions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run – scrape website, download PDFs, analyse prices:
  python analyse_newsletters.py

  # Limit to 5 newsletters (useful for a quick test):
  python analyse_newsletters.py --max-newsletters 5

  # Use pre-downloaded PDFs in a local folder:
  python analyse_newsletters.py --pdf-dir /path/to/pdfs

  # Re-run price analysis only (skip scraping, use cached mention_log.csv):
  python analyse_newsletters.py --skip-scrape
        """,
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for CSV and chart output (default: results/)",
    )
    parser.add_argument(
        "--max-newsletters", type=int, default=None,
        help="Limit number of newsletters to process (useful for testing)",
    )
    parser.add_argument(
        "--cache-dir", default=".pdf_cache",
        help="Directory to cache downloaded PDFs (default: .pdf_cache/)",
    )
    parser.add_argument(
        "--pdf-dir", default=None,
        help="Read PDFs from this local directory instead of scraping the website",
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip scraping; reload mention_log.csv from --output-dir",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # ---- Phase 1: discover and parse newsletters ----
    if args.skip_scrape:
        log_path = output_dir / "mention_log.csv"
        if not log_path.exists():
            sys.exit(f"[Error] --skip-scrape specified but {log_path} not found.")
        print(f"[Main] Loading existing mention log from {log_path}")
        mention_log = pd.read_csv(log_path, parse_dates=["mention_date"])

    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.is_dir():
            sys.exit(f"[Error] --pdf-dir path does not exist: {pdf_dir}")
        print(f"[Main] Reading PDFs from local directory: {pdf_dir}")
        mention_log = build_mention_log_from_local_pdfs(pdf_dir)
        if mention_log.empty:
            sys.exit("[Main] No tickers extracted from local PDFs.")

    else:
        print("[Main] Scraping Ennismore Fund Management newsletter pages...")
        newsletters = discover_newsletter_pdfs(
            NEWSLETTER_PAGES, max_newsletters=args.max_newsletters
        )
        if not newsletters:
            print(
                "[Main] No newsletters discovered.\n"
                "       Possible causes:\n"
                "         • No internet access to ennismorefunds.com\n"
                "         • Site layout has changed\n"
                "       Workaround: download PDFs manually and use --pdf-dir"
            )
            sys.exit(1)

        mention_log = build_mention_log(newsletters, cache_dir)

        if mention_log.empty:
            print("[Main] No tickers extracted from newsletters. "
                  "PDFs may be image-based or ticker patterns need tuning.")
            sys.exit(1)

    # ---- Phase 2: price performance analysis ----
    print(f"\n[Main] Analysing performance for {mention_log['ticker'].nunique()} unique tickers...")
    performance = analyse_performance(mention_log, today)

    # ---- Phase 3: output ----
    save_csv_reports(performance, mention_log, output_dir)
    plot_performance(performance, output_dir)
    print_summary(performance)


if __name__ == "__main__":
    main()
