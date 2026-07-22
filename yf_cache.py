"""Disk-backed cache around yfinance.

Every notebook in this pipeline re-fetches overlapping price/fundamentals data
from yfinance on every run (same ~250-470 tickers, mostly overlapping date
ranges, across 8 notebooks). That's redundant network I/O and is one of the
slowest parts of re-running the pipeline. This module is a drop-in cache:

    import yf_cache
    px_raw = yf_cache.download(tickers_ns, start=..., end=..., auto_adjust=True)
    info   = yf_cache.Ticker(ticker).info

`download` and `Ticker` mirror `yfinance.download` / `yfinance.Ticker`'s
interface for the calls this project makes. A cache hit returns the exact
object saved from the original live fetch; a miss fetches once and saves it.
Cached data does not expire — delete files under data/raw/yfinance_cache/ (or
the whole directory) to force a re-fetch.
"""
from pathlib import Path
import hashlib
import pickle
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent / 'data' / 'raw' / 'yfinance_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key(*parts):
    raw = '|'.join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def download(tickers, start=None, end=None, **kwargs):
    """Cached drop-in for yf.download(tickers, start=start, end=end, **kwargs)."""
    ticker_key = tickers if isinstance(tickers, str) else ','.join(sorted(tickers))
    path = CACHE_DIR / f'download_{_key(ticker_key, start, end, sorted(kwargs.items()))}.pkl'
    if path.exists():
        return _load(path)
    df = yf.download(tickers, start=start, end=end, **kwargs)
    _save(path, df)
    return df


class Ticker:
    """Cached stand-in for yf.Ticker — covers .info/.financials/.income_stmt/.balance_sheet."""

    def __init__(self, ticker):
        self._ticker = ticker
        self._live_ticker = None

    def _live(self):
        if self._live_ticker is None:
            self._live_ticker = yf.Ticker(self._ticker)
        return self._live_ticker

    def _cached_attr(self, name):
        path = CACHE_DIR / f'{name}_{self._ticker.replace("/", "_")}.pkl'
        if path.exists():
            return _load(path)
        value = getattr(self._live(), name)
        _save(path, value)
        return value

    @property
    def info(self):
        return self._cached_attr('info')

    @property
    def financials(self):
        return self._cached_attr('financials')

    @property
    def income_stmt(self):
        return self._cached_attr('income_stmt')

    @property
    def balance_sheet(self):
        return self._cached_attr('balance_sheet')
