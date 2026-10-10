"""Minimal mock WebShop Flask server compatible with minja.py webshop_text parser.

Serves product search/browse/purchase for 1000 products from the small dataset.
Matches the exact HTML structure expected by minja.py's webshop_text() function.
"""
from __future__ import annotations
import json
import os
import re
import sys
import random
from typing import Optional
from pathlib import Path
from flask import Flask, request

# Rank BM25 for search (already installed in rap_webshop env)
from rank_bm25 import BM25Okapi

# ─── Data loading ────────────────────────────────────────────────────────────
WEBSHOP_DIR = Path(os.environ.get("WEBSHOP_DATA_DIR",
                                  str(Path(__file__).parent / "webshop_setup" / "WebShop")))
DATA_DIR = WEBSHOP_DIR / "data"

def _load_json(path):
    with open(path) as f:
        return json.load(f)

print("Loading product data...", file=sys.stderr)
try:
    # Prefer FULL dataset when available (contains MINJA's 9 target products);
    # fall back to 1000-product subset for quick smoke testing.
    if (DATA_DIR / "items_shuffle.json").exists():
        ALL_PRODUCTS = _load_json(DATA_DIR / "items_shuffle.json")
        print(f"  Loaded {len(ALL_PRODUCTS):,} products (FULL dataset)", file=sys.stderr)
    else:
        ALL_PRODUCTS = _load_json(DATA_DIR / "items_shuffle_1000.json")
        print(f"  Loaded {len(ALL_PRODUCTS)} products (small subset)", file=sys.stderr)
    if (DATA_DIR / "items_ins_v2.json").exists():
        ATTRS = _load_json(DATA_DIR / "items_ins_v2.json")
    else:
        ATTRS = _load_json(DATA_DIR / "items_ins_v2_1000.json")
    HUMAN_INS = _load_json(DATA_DIR / "items_human_ins.json")
    print("  Loaded attributes and instructions", file=sys.stderr)
except Exception as e:
    print(f"ERROR loading data: {e}", file=sys.stderr)
    sys.exit(1)

# Build product index by ASIN
PRODUCT_BY_ASIN: dict[str, dict] = {}
for p in ALL_PRODUCTS:
    asin = p.get("asin", p.get("ASIN", ""))
    if asin:
        PRODUCT_BY_ASIN[asin] = p

# Build BM25 index for search
def _tokenize(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

_corpus_asins = list(PRODUCT_BY_ASIN.keys())
_corpus_texts = []
for asin in _corpus_asins:
    p = PRODUCT_BY_ASIN[asin]
    title = p.get("Title", p.get("name", p.get("title", "")))
    desc = p.get("Description", p.get("full_description", p.get("description", "")))
    if isinstance(desc, list):
        desc = " ".join(desc)
    small_desc = p.get("small_description", "")
    if isinstance(small_desc, list):
        small_desc = " ".join(small_desc)
    category = p.get("MainCategory", p.get("product_category", p.get("category", "")))
    query = p.get("query", "")
    text = f"{title} {desc} {small_desc} {category} {query}"
    _corpus_texts.append(_tokenize(text))

print("  Building BM25 index...", file=sys.stderr)
_bm25 = BM25Okapi(_corpus_texts)
print("  BM25 index ready", file=sys.stderr)

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4096)
def _cached_search_asins(query: str, n: int = 50) -> tuple:
    """LRU-cached BM25 top-N ASINs for a normalized query string.

    BM25 over 1.18M products takes ~3s/query — caching cuts repeated searches
    (same `search[target]` across many inject sessions) to microseconds.
    Returns a tuple (immutable, hashable) so we can stash in lru_cache.
    """
    tokens = _tokenize(query)
    if not tokens:
        return tuple(_corpus_asins[:n])
    scores = _bm25.get_scores(tokens)
    # Use argpartition for O(N) top-K instead of O(N log N) full sort.
    if len(scores) > n:
        top_idx = np.argpartition(-scores, n)[:n]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
    else:
        top_idx = np.argsort(-scores)
    asins = []
    for i in top_idx:
        if scores[i] > 0:
            asins.append(_corpus_asins[i])
    if not asins:
        asins = _corpus_asins[:n]
    return tuple(asins[:n])


def search_products(query: str, n: int = 50) -> list[dict]:
    """Search products using BM25 and return top N matches (LRU-cached)."""
    asins = _cached_search_asins(query, n)
    return [PRODUCT_BY_ASIN[a] for a in asins]

# ─── Session state ───────────────────────────────────────────────────────────
SESSION_DATA: dict[str, dict] = {}

def _get_product_title(p: dict) -> str:
    """Get product title (handles both 'Title' and 'name' field names)."""
    return p.get("Title", p.get("name", p.get("title", "Unknown Product")))

def _get_product_price(p: dict) -> float:
    """Extract numeric price from product."""
    raw = p.get("Price", p.get("price", p.get("pricing", "$10.00")))
    if isinstance(raw, (int, float)):
        return float(raw)
    m = re.search(r'[\d.]+', str(raw))
    return float(m.group()) if m else 10.00

def _get_product_options(p: dict) -> dict:
    """Extract options from product (color, size, etc.)."""
    asin = p.get("asin", p.get("ASIN", ""))
    # Try customization_options first (rich format)
    raw = p.get("customization_options", p.get("options", None))
    if raw and isinstance(raw, dict):
        result = {}
        for opt_type, opt_vals in raw.items():
            if isinstance(opt_vals, list):
                values = []
                for v in opt_vals:
                    if isinstance(v, dict):
                        val = v.get("value", "")
                        if val:
                            values.append(str(val))
                    elif isinstance(v, str):
                        values.append(v)
                if values:
                    result[opt_type] = values
            elif opt_vals is None:
                pass
        if result:
            return result
    # Fallback: human_ins options (format: ['color: blue'])
    if asin in HUMAN_INS:
        h = HUMAN_INS[asin]
        if isinstance(h, list) and h:
            h = h[0]
        raw_opts = h.get("options", [])
        if isinstance(raw_opts, list):
            result = {}
            for opt_str in raw_opts:
                if ":" in opt_str:
                    k, v = opt_str.split(":", 1)
                    result.setdefault(k.strip(), []).append(v.strip())
            if result:
                return result
    return {}

# ─── HTML templates ──────────────────────────────────────────────────────────
# These are designed so that minja.py's webshop_text() BeautifulSoup parser
# produces the exact text format expected by the RAP agent.

def _init_html(session_id: str, instruction: str) -> str:
    """Init page: shows instruction text + Search button."""
    return f"""<html><body>
<p>Instruction: {instruction}</p>
<button>Search</button>
</body></html>"""

def _search_html(session_id: str, instruction: str, products: list[dict],
                 query: str, page: int = 1) -> str:
    """Search results page: shows product list."""
    total = len(products)
    # Show max 3 products per page (to match RAP agent's prod_cnt >= 3 limit)
    start = (page - 1) * 10
    page_products = products[start:start + 10]

    rows = ""
    for p in page_products:
        asin = p.get("asin", p.get("ASIN", "UNKNOWN"))
        title = _get_product_title(p)
        price = _get_product_price(p)
        rows += f"""
<a class="product-link">{asin}</a>
<p>{title}</p>
<p>${price:.2f}</p>"""

    has_next = (start + 10) < total
    next_btn = "<button>Next ></button>" if has_next else ""

    return f"""<html><body>
<p>{session_id}</p>
<button>Back to Search</button>
<p>Page {page} (Total results: {total})</p>
{next_btn}
{rows}
</body></html>"""

def _item_html(session_id: str, instruction: str, p: dict,
               query: str, selected_options: dict) -> str:
    """Item detail page with options and Buy Now."""
    asin = p.get("asin", p.get("ASIN", "UNKNOWN"))
    title = _get_product_title(p)
    price = _get_product_price(p)
    rating = p.get("Rating", p.get("average_rating", p.get("rating", "N.A.")))

    opts = _get_product_options(p)
    options_html = ""
    for opt_type, opt_values in opts.items():
        options_html += f"\n<p>{opt_type}</p>"
        if isinstance(opt_values, list):
            for v in opt_values:
                selected = selected_options.get(opt_type) == v
                if selected:
                    options_html += f"<label>[[{v}]]</label>"
                else:
                    options_html += f"<label>{v}</label>"

    return f"""<html><body>
<p>{session_id}</p>
<button>Back to Search</button>
<button>&lt; Prev</button>
{options_html}
<p>{title}</p>
<p>Price: ${price:.2f}</p>
<p>Rating: {rating}</p>
<button>Description</button>
<button>Features</button>
<button>Reviews</button>
<button>Buy Now</button>
</body></html>"""

def _done_html(session_id: str, score: float, asin: str) -> str:
    """Purchase complete page with reward score."""
    return f"""<html><body><p>Your score (min 0.0, max 1.0)</p><p>{score:.2f}</p></body></html>"""

def _sub_html(session_id: str, instruction: str, p: dict, subpage: str) -> str:
    """Item sub-page (Description/Features/Reviews)."""
    title = _get_product_title(p)
    if subpage == "Description":
        desc = p.get("Description", p.get("full_description", p.get("description", "No description available.")))
        if isinstance(desc, list):
            desc = " ".join(desc)
        content = f"<p>{desc[:500] if desc else 'No description.'}</p>"
    elif subpage == "Features":
        feats = p.get("BulletPoints", p.get("small_description", p.get("features", [])))
        if isinstance(feats, str):
            feats = [feats]
        content = "".join(f"<p>{f}</p>" for f in feats[:5])
    else:
        content = "<p>No reviews available.</p>"

    return f"""<html><body>
<p>{session_id}</p>
<button>Back to Search</button>
<button>&lt; Prev</button>
<p>{title}</p>
{content}
</body></html>"""

def _compute_reward(p: dict, instruction: str, selected_options: dict) -> float:
    """Compute reward: 1.0 if product title matches instruction keywords."""
    title = (_get_product_title(p) or "").lower()
    instr = (instruction or "").lower()
    # Simple keyword overlap scoring
    words = set(re.findall(r'\b\w{4,}\b', instr)) - {'want', 'need', 'looking', 'that', 'with', 'from', 'have', 'this', 'would', 'like', 'find', 'buy', 'price', 'lower', 'than', 'dollar'}
    if not words:
        return 0.5
    matches = sum(1 for w in words if w in title)
    return min(1.0, matches / max(1, len(words)))

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# Session instruction store (populated from merged_instructions.json if available)
INSTRUCTION_STORE: dict[str, str] = {}

def _get_instruction(session_id: str) -> str:
    """Get instruction for session, loading from JSON if needed."""
    if session_id in INSTRUCTION_STORE:
        return INSTRUCTION_STORE[session_id]
    # Try to load from merged_instructions.json (minja.py writes this)
    for path in [
        str(Path(__file__).parent / "results" / "merged_instructions.json"),
        str(Path(__file__).parent / "merged_instructions.json"),
    ]:
        if os.path.exists(path):
            try:
                data = _load_json(path)
                for entry in data:
                    sid = entry.get("session", "")
                    instr = entry.get("instruction", "")
                    if sid:
                        INSTRUCTION_STORE[sid] = instr
                if session_id in INSTRUCTION_STORE:
                    return INSTRUCTION_STORE[session_id]
            except Exception:
                pass
    return f"I am shopping for something related to session {session_id}."

@app.route("/<session_id>")
def init_page(session_id: str):
    instruction = _get_instruction(session_id)
    SESSION_DATA[session_id] = {
        "instruction": instruction,
        "search_results": [],
        "current_asin": None,
        "selected_options": {},
        "last_query": "",
    }
    return _init_html(session_id, instruction)

@app.route("/search_results/<session_id>/<path:query_and_page>")
def search_results(session_id: str, query_and_page: str):
    parts = query_and_page.rsplit("/", 1)
    if len(parts) == 2 and parts[1].isdigit():
        query, page = parts[0], int(parts[1])
    else:
        query, page = query_and_page, 1

    instruction = _get_instruction(session_id)
    products = search_products(query, n=50)
    if session_id not in SESSION_DATA:
        SESSION_DATA[session_id] = {}
    SESSION_DATA[session_id]["search_results"] = products
    SESSION_DATA[session_id]["last_query"] = query
    SESSION_DATA[session_id]["instruction"] = instruction
    return _search_html(session_id, instruction, products, query, page)

@app.route("/item_page/<session_id>/<asin>/<path:rest>")
def item_page(session_id: str, asin: str, rest: str):
    # rest = query/page_num/options
    parts = rest.rsplit("/", 2)
    instruction = _get_instruction(session_id)
    p = PRODUCT_BY_ASIN.get(asin)
    if p is None:
        # Search by ASIN prefix
        for k, v in PRODUCT_BY_ASIN.items():
            if k.startswith(asin[:6]):
                p = v
                break
    if p is None:
        p = {"asin": asin, "Title": "Unknown Product", "Price": "$10.00"}

    # Parse selected options from URL (options is last part)
    selected = {}
    if len(parts) >= 1:
        try:
            opts_str = parts[-1]
            # options is a dict-like string in URL
            # Try to evaluate it safely
            opts_str = opts_str.replace('%27', "'").replace('%22', '"')
            if opts_str and opts_str != '{}':
                selected = json.loads(opts_str) if opts_str.startswith('{') else {}
        except Exception:
            selected = {}

    if session_id not in SESSION_DATA:
        SESSION_DATA[session_id] = {}
    SESSION_DATA[session_id]["current_asin"] = asin
    SESSION_DATA[session_id]["selected_options"] = selected
    SESSION_DATA[session_id]["instruction"] = instruction
    return _item_html(session_id, instruction, p, "", selected)

@app.route("/item_sub_page/<session_id>/<asin>/<path:rest>")
def item_sub_page(session_id: str, asin: str, rest: str):
    # rest = query/page_num/subpage/options
    parts = rest.rsplit("/", 2)
    subpage = parts[-2] if len(parts) >= 2 else "Description"
    instruction = _get_instruction(session_id)
    p = PRODUCT_BY_ASIN.get(asin, {"asin": asin, "Title": "Unknown Product"})
    return _sub_html(session_id, instruction, p, subpage)

@app.route("/done/<session_id>/<asin>/<path:options_str>")
def done_page(session_id: str, asin: str, options_str: str):
    p = PRODUCT_BY_ASIN.get(asin, {"asin": asin, "Title": "Unknown Product"})
    instruction = _get_instruction(session_id)
    score = _compute_reward(p, instruction, {})
    return _done_html(session_id, score, asin)

if __name__ == "__main__":
    port = int(os.environ.get("WEBSHOP_PORT", "3000"))
    print(f"Starting mock WebShop server on port {port}", file=sys.stderr)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
