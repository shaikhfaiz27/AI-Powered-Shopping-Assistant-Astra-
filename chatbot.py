"""
Production-Level AI Shopping Assistant Engine
Handles: Budget, Category, PC Build, Ranking, Fallback, Intent Detection
"""

import os
import sys
import warnings
import logging

# ── Suppress noisy library warnings (16GB-RAM friendly setup) ──
# These specific "[transformers] Accessing __path__ from ..." messages
# come from transformers' lazy-module __getattr__ and are emitted via
# both the warnings system AND a dedicated logger, depending on version.
os.environ["TRANSFORMERS_VERBOSITY"]      = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"]    = "1"

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Silence every transformers-related logger
for noisy_logger in ("transformers", "transformers.utils", "transformers.utils.import_utils",
                      "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)
    logging.getLogger(noisy_logger).propagate = False

import faiss
import pickle
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from google import genai
from google.genai import types


# =====================
# GEMINI SETUP
# =====================

client = genai.Client(api_key="")  # Set your Gemini API key here


# =====================
# TRANSLATION
# =====================
# Detects user's input language and translates to English for
# internal processing, then translates Astra's reply back.
# Uses deep_translator (Google Translate, free, no API key).

try:
    from deep_translator import GoogleTranslator
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0   # deterministic detection
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False


# Languages we actively support (ISO codes -> display names)
# Note: GoogleTranslator (via deep_translator) supports these codes.
# Maithili and Magahi are NOT supported by Google Translate yet,
# so they are intentionally excluded — selecting them would silently
# fail. Bhojpuri ('bho') and Nepali ('ne') ARE supported.
SUPPORTED_LANGUAGES = {
    "en":    "English",
    "hi":    "Hindi",
    "mr":    "Marathi",
    "ta":    "Tamil",
    "te":    "Telugu",
    "kn":    "Kannada",
    "gu":    "Gujarati",
    "bn":    "Bengali",
    "ml":    "Malayalam",
    "pa":    "Punjabi",
    "ur":    "Urdu",
    "ne":    "Nepali",
    "bho":   "Bhojpuri",
    "or":    "Odia",
    "as":    "Assamese",
    "sa":    "Sanskrit",
    "sd":    "Sindhi",
    "es":    "Spanish",
    "fr":    "French",
    "de":    "German",
    "ar":    "Arabic",
    "zh-cn": "Chinese",
    "ja":    "Japanese",
    "pt":    "Portuguese",
    "ru":    "Russian",
}

# langdetect doesn't recognize 'bho' (Bhojpuri) — it usually
# detects Bhojpuri text as 'hi' (Hindi) since both share Devanagari
# script and heavy vocabulary overlap. We can't reliably auto-detect
# Bhojpuri, so it's only selectable manually from the dropdown.


def detect_language(text: str) -> str:
    """Detect the language of input text. Returns 'en' on failure or short text."""
    if not TRANSLATION_AVAILABLE:
        return "en"
    text = text.strip()
    if len(text) < 2:
        return "en"
    try:
        code = detect(text)
        if code.startswith("zh"):
            code = "zh-cn"
        if code in SUPPORTED_LANGUAGES:
            return code
        return "en"
    except Exception:
        return "en"


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate user input to English for internal processing."""
    if not TRANSLATION_AVAILABLE or source_lang == "en":
        return text
    try:
        result = GoogleTranslator(source=source_lang, target="en").translate(text)
        return result if result else text
    except Exception:
        # Try auto-detect as fallback source
        try:
            result = GoogleTranslator(source="auto", target="en").translate(text)
            return result if result else text
        except Exception:
            return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate Astra's English reply back to the user's language.

    Markdown formatting (**, |, #, etc.) is preserved by translating
    line-by-line. Lines that fail to translate are retried once with
    auto-source; if still failing, the ORIGINAL English line is kept
    rather than silently dropping content — so the user always gets
    a complete reply (mixed-language is better than missing info)."""
    if not TRANSLATION_AVAILABLE or target_lang == "en":
        return text

    try:
        translator = GoogleTranslator(source="en", target=target_lang)
    except Exception:
        return text  # invalid target language code

    lines = text.split("\n")
    out_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines, table separators, pure markdown syntax
        if not stripped or re.fullmatch(r'[\-\|\*\#\s\:_>]+', stripped):
            out_lines.append(line)
            continue

        # Preserve leading markdown markers (###, -, |, >, spaces)
        prefix_match = re.match(r'^([\#\-\*\|\s>]*)(.*)', line)
        prefix, content = prefix_match.groups()

        if not content.strip():
            out_lines.append(line)
            continue

        # ── Table rows: translate each cell separately to keep | structure ──
        if "|" in content and content.count("|") >= 2:
            cells = content.split("|")
            translated_cells = []
            for cell in cells:
                cell_stripped = cell.strip()
                if not cell_stripped or re.fullmatch(r'[\-\:\s]+', cell_stripped):
                    translated_cells.append(cell)
                    continue
                translated_cells.append(_safe_translate(translator, cell))
            out_lines.append(prefix + "|".join(translated_cells))
            continue

        # ── Normal line ──
        out_lines.append(prefix + _safe_translate(translator, content))

    return "\n".join(out_lines)


def _safe_translate(translator, text: str) -> str:
    """Translate a single chunk; on failure, return original text unchanged
    so the reply is never silently incomplete."""
    text = text.strip()
    if not text:
        return text
    try:
        result = translator.translate(text)
        return result if result else text
    except Exception:
        return text






# =====================
# LOAD DATA
# =====================

def load_data():
    import os

    # ── products.pkl ──────────────────────────────────────
    if not os.path.exists("products.pkl"):
        raise FileNotFoundError("products.pkl not found. Make sure it is in the same folder as app.py")
    products = pickle.load(open("products.pkl", "rb"))

    # ── FAISS index ───────────────────────────────────────
    if not os.path.exists("products.index"):
        raise FileNotFoundError("products.index not found. Make sure it is in the same folder as app.py")
    index = faiss.read_index("products.index")

    # ── SentenceTransformer — lightweight, CPU-only ───────
    try:
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"          # force CPU — works on any machine
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SentenceTransformer model: {e}\n"
            "Run: pip install sentence-transformers"
        )

    # ── Excel sheets ──────────────────────────────────────
    if not os.path.exists("ecommerce_data.xlsx"):
        raise FileNotFoundError("ecommerce_data.xlsx not found. Make sure it is in the same folder as app.py")

    reviews = pd.read_excel("ecommerce_data.xlsx", sheet_name="Reviews_1")
    specs   = pd.read_excel("ecommerce_data.xlsx", sheet_name="Specifications")

    compatibility = None
    try:
        compatibility = pd.read_excel("ecommerce_data.xlsx", sheet_name="Compatibility")
    except Exception:
        pass

    sales = None
    try:
        sales = pd.read_excel("ecommerce_data.xlsx", sheet_name="Sales_1")
    except Exception:
        pass

    # ── Merge sales into products ─────────────────────────
    if sales is not None and "product_id" in sales.columns:
        agg_cols = {}
        for col in ["monthly_sales", "sales_count", "popularity_score"]:
            if col in sales.columns:
                agg_cols[col] = "mean"
        if agg_cols:
            sales_agg = sales.groupby("product_id").agg(agg_cols).reset_index()
            products  = products.merge(sales_agg, on="product_id", how="left")

    return products, index, model, reviews, specs, compatibility


# =====================
# CATEGORY MAPPING
# =====================

CATEGORY_KEYWORDS = {
    "Smartphone": ["phone", "mobile", "smartphone", "iphone", "android", "5g phone"],
    "Laptop": ["laptop", "notebook", "ultrabook", "macbook", "chromebook"],
    "CPU": ["cpu", "processor", "ryzen", "intel core", "amd cpu"],
    "GPU": ["gpu", "graphics card", "rtx", "rx ", "nvidia", "amd gpu", "video card"],
    "Motherboard": ["motherboard", "mobo", "mainboard"],
    "RAM": ["ram", "memory", "ddr4", "ddr5"],
    "SSD": ["ssd", "solid state", "nvme", "m.2 drive"],
    "HDD": ["hdd", "hard drive", "hard disk"],
    "PSU": ["psu", "power supply", "smps"],
    "Monitor": ["monitor", "display", "screen", "4k monitor"],
    "Keyboard": ["keyboard", "mechanical keyboard"],
    "Mouse": ["mouse", "gaming mouse"],
    "Headset": ["headset", "headphones", "earphones", "earbuds"],
    "Router": ["router", "wifi", "networking"],
    "Tablet": ["tablet", "ipad", "android tablet"],
    "Smartwatch": ["smartwatch", "watch", "wearable"],
}

PC_COMPONENTS = ["CPU", "GPU", "Motherboard", "RAM", "SSD", "PSU"]

PC_BUILD_KEYWORDS = [
    "build", "gaming pc", "custom pc", "desktop pc", "assemble",
    "pc build", "build pc", "gaming rig", "workstation build"
]

# Default budget splits for PC build (% of total budget)
PC_BUDGET_SPLIT = {
    "CPU":         0.15,
    "GPU":         0.35,
    "Motherboard": 0.10,
    "RAM":         0.08,
    "SSD":         0.07,
    "PSU":         0.07,
}

WORKLOAD_KEYWORDS = {
    "gaming":           ["gaming", "game", "fps", "esports", "aaa game"],
    "ai_ml":            ["ai", "ml", "machine learning", "deep learning", "cuda", "tensorflow", "pytorch"],
    "programming":      ["programming", "coding", "developer", "software", "ide"],
    "office":           ["office", "excel", "word", "email", "basic use", "work from home"],
    "content_creation": ["video editing", "rendering", "blender", "premiere", "photoshop", "content creation", "creator"],
}


# =====================
# INTENT DETECTION
# =====================

def extract_budget(query: str) -> int | None:
    q = query.lower().replace(",", "")

    # Handle "lakh" / "lac"
    lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|l\b)', q)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100_000)

    # Handle "k" suffix (e.g. 50k)
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*k\b', q)
    if k_match:
        return int(float(k_match.group(1)) * 1_000)

    # Plain number
    nums = re.findall(r'\d+', q)
    if nums:
        # Pick largest number (most likely to be budget)
        return max(int(n) for n in nums)

    return None


def detect_category(query: str) -> str | None:
    q = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return category
    return None


def detect_workload(query: str) -> list[str]:
    q = query.lower()
    detected = []
    for workload, keywords in WORKLOAD_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            detected.append(workload)
    return detected


def is_pc_build_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in PC_BUILD_KEYWORDS)


def detect_intent(query: str) -> str:
    q = query.lower()

    if is_pc_build_query(q):
        return "PC_BUILD"
    elif "compare" in q or " vs " in q:
        return "COMPARE"
    elif "review" in q:
        return "REVIEW"
    elif "best" in q or "top" in q:
        return "BEST"
    elif "recommend" in q or "suggest" in q:
        return "RECOMMEND"
    elif "spec" in q or "specification" in q:
        return "SPECS"

    return "SEARCH"


def parse_query(query: str) -> dict:
    return {
        "raw":      query,
        "intent":   detect_intent(query),
        "budget":   extract_budget(query),
        "category": detect_category(query),
        "workload": detect_workload(query),
    }


# =====================
# QUERY REWRITE
# =====================

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this shopping query into concise, effective search keywords.
Query: {query}
Return only keywords, no explanation."""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip()
    except Exception:
        return query


# =====================
# SEMANTIC SEARCH
# =====================

def semantic_search(query: str, products: pd.DataFrame, index, model, k: int = 20) -> pd.DataFrame:
    vector = model.encode([query], normalize_embeddings=True)
    vector = np.asarray(vector, dtype=np.float32)
    _, ids = index.search(vector, k)
    valid_ids = [i for i in ids[0] if 0 <= i < len(products)]
    return products.iloc[valid_ids].copy()


# =====================
# FILTERING
# =====================

def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    if "category" not in df.columns:
        return df
    mask = df["category"].str.lower() == category.lower()
    filtered = df[mask]
    return filtered if len(filtered) > 0 else df


def filter_by_budget(df: pd.DataFrame, budget: int) -> pd.DataFrame:
    if "price" not in df.columns or budget is None:
        return df

    # Tier 1: 90–100% of budget (ideal sweet spot)
    t1 = df[(df["price"] >= budget * 0.90) & (df["price"] <= budget)]
    if len(t1) >= 3:
        return t1.sort_values("price", ascending=False)

    # Tier 2: 75–100% of budget
    t2 = df[(df["price"] >= budget * 0.75) & (df["price"] <= budget)]
    if len(t2) >= 3:
        return t2.sort_values("price", ascending=False)

    # Tier 3: anything under budget — sort by price descending
    # so nearest-to-budget items always come first
    under = df[df["price"] <= budget].sort_values("price", ascending=False)
    if len(under) >= 1:
        return under

    # Final fallback: sort by closeness to budget (includes slightly over)
    df = df.copy()
    df["_price_diff"] = (df["price"] - budget).abs()
    return df.sort_values("_price_diff").drop(columns=["_price_diff"])


# =====================
# ML RANKING
# =====================

RANKING_COLS = [
    "rating", "monthly_sales", "quality_score",
    "trend_score", "popularity_score", "sales_count"
]


def rank_products(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in RANKING_COLS if c in df.columns]

    if not available:
        return df

    df = df.copy()
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(df[available].fillna(0))

    # Weighted: rating & quality get more weight
    weights = {
        "rating":           2.0,
        "monthly_sales":    1.5,
        "quality_score":    1.8,
        "trend_score":      1.2,
        "popularity_score": 1.5,
        "sales_count":      1.0,
    }

    weight_arr = np.array([weights.get(c, 1.0) for c in available])
    weight_arr = weight_arr / weight_arr.sum()

    df["ml_score"] = (scores * weight_arr).sum(axis=1)
    return df.sort_values("ml_score", ascending=False)


# =====================
# REVIEW ANALYSIS
# =====================

def get_review_summary(product_id, reviews: pd.DataFrame) -> dict:
    r = reviews[reviews["product_id"] == product_id]
    if len(r) == 0:
        return {
            "avg_rating": None, "count": 0, "sentiment": "No reviews",
            "positives": [], "negatives": [], "five_star_pct": 0,
            "one_star_pct": 0, "verified_count": 0
        }

    avg   = round(r["rating"].mean(), 2)
    count = len(r)

    # Rating distribution
    five_star = len(r[r["rating"] == 5])
    one_star  = len(r[r["rating"] == 1])
    five_star_pct = round(five_star / count * 100) if count else 0
    one_star_pct  = round(one_star  / count * 100) if count else 0

    # Verified purchases
    verified_count = 0
    if "verified_purchase" in r.columns:
        verified_count = int(r["verified_purchase"].sum())

    positives, negatives, neutral = [], [], []
    if "review_text" in r.columns:
        for _, row in r.iterrows():
            text   = str(row.get("review_text", "")).strip()
            rating = row.get("rating", 3)
            if not text or text.lower() in ("nan", "none", ""):
                continue
            if rating >= 4:
                positives.append(text[:150])
            elif rating <= 2:
                negatives.append(text[:150])
            else:
                neutral.append(text[:150])

    return {
        "avg_rating":    avg,
        "count":         count,
        "five_star_pct": five_star_pct,
        "one_star_pct":  one_star_pct,
        "verified_count": verified_count,
        "positives":     positives[:3],
        "negatives":     negatives[:3],
        "neutral":       neutral[:1],
        "sentiment":     "Positive" if avg >= 4 else ("Mixed" if avg >= 3 else "Negative"),
    }


def create_context(df: pd.DataFrame, reviews: pd.DataFrame, specs: pd.DataFrame, top_n: int = 5) -> str:
    context = ""
    for rank_pos, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        pid = row.get("product_id", "")

        # ── Reviews ───────────────────────────────────────────
        rev = get_review_summary(pid, reviews)

        # Rating label
        avg_r = rev["avg_rating"]
        if avg_r is not None:
            if avg_r >= 4.5:   rating_label = "Excellent"
            elif avg_r >= 4.0: rating_label = "Very Good"
            elif avg_r >= 3.5: rating_label = "Good"
            elif avg_r >= 3.0: rating_label = "Average"
            else:              rating_label = "Below Average"
        else:
            rating_label = "No rating"

        rev_text = f"Rating: {avg_r} / 5.0 ({rating_label}) — {rev['count']} reviews"

        # Positive highlights (up to 3)
        if rev["positives"]:
            rev_text += "\n  Buyer Pros:"
            for p in rev["positives"][:3]:
                rev_text += f"\n    + {p}"

        # Negative highlights (up to 2)
        if rev["negatives"]:
            rev_text += "\n  Buyer Cons:"
            for n in rev["negatives"][:2]:
                rev_text += f"\n    - {n}"

        # ── Specs — include ALL available columns ──────────────
        spec_text = "Not available"
        if specs is not None and "product_id" in specs.columns:
            p_specs = specs[specs["product_id"] == pid]
            if len(p_specs) > 0:
                spec_items = []
                for col in p_specs.columns:
                    if col != "product_id":
                        val = p_specs.iloc[0][col]
                        if pd.notna(val) and str(val).strip() not in ("", "nan", "None"):
                            spec_items.append(f"{col}: {val}")
                spec_text = " | ".join(spec_items) if spec_items else "Not available"

        # ── ML score label ─────────────────────────────────────
        ml = row.get("ml_score", 0)
        if ml >= 0.8:   score_label = "Top Pick"
        elif ml >= 0.6: score_label = "Highly Recommended"
        elif ml >= 0.4: score_label = "Good Choice"
        else:           score_label = "Decent Option"

        # ── Extra ranking signals ──────────────────────────────
        extra_signals = []
        if row.get("trend_score", 0) >= 0.7:
            extra_signals.append("Trending 🔥")
        if row.get("monthly_sales", 0) >= 100:
            extra_signals.append(f"Sells {int(row['monthly_sales'])}+/month")
        if row.get("popularity_score", 0) >= 0.8:
            extra_signals.append("Very Popular")
        signals_text = " | ".join(extra_signals) if extra_signals else "—"

        price_val = row.get('price', 'N/A')
        try:
            price_str = f"₹{int(price_val):,}"
        except (ValueError, TypeError):
            price_str = f"₹{price_val}"

        context += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[#{rank_pos}] {row.get('product_name', 'Unknown')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Brand        : {row.get('brand', 'N/A')}
Category     : {row.get('category', 'N/A')}
Price        : {price_str}
AI Rank      : {score_label} (score: {round(ml, 3)})
Signals      : {signals_text}

DESCRIPTION  :
{str(row.get('description', 'No description available.'))}

SPECIFICATIONS:
{spec_text}

REVIEWS      :
{rev_text}
"""

    return context


# =====================
# LOCAL FALLBACK RESPONSE
# =====================

def create_local_response(df: pd.DataFrame, reviews: pd.DataFrame, query_info: dict) -> str:
    intent   = query_info.get("intent", "SEARCH")
    budget   = query_info.get("budget")
    category = query_info.get("category", "")

    lines = [f"## 🛒 Top Recommendations — {category or 'All Categories'}"]
    if budget:
        lines.append(f"💰 **Budget:** ₹{budget:,}")
    lines.append("")

    for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
        pid = row.get("product_id", "")
        rev = get_review_summary(pid, reviews)

        avg_r = rev["avg_rating"]
        rating_str = f"⭐ {avg_r}/5 ({rev['count']} reviews)" if avg_r else f"⭐ {row.get('rating', 'N/A')}"

        lines.append(f"### {i}. {row.get('product_name', 'Unknown')}")
        lines.append(
            f"**₹{int(row.get('price', 0)):,}** &nbsp;|&nbsp; "
            f"{rating_str} &nbsp;|&nbsp; "
            f"🏷️ {row.get('brand', 'N/A')} &nbsp;|&nbsp; "
            f"📦 {row.get('category', '')}"
        )

        # Description snippet
        desc = str(row.get("description", ""))[:200]
        if desc and desc.lower() not in ("nan", "none", ""):
            lines.append(f"_{desc}_")

        # Review highlights
        if rev["positives"]:
            lines.append(f"✅ **Buyers love:** {rev['positives'][0]}")
        if rev["negatives"]:
            lines.append(f"⚠️ **Watch out:** {rev['negatives'][0]}")
        if rev["five_star_pct"]:
            lines.append(f"📊 {rev['five_star_pct']}% gave 5 stars")

        lines.append("")

    if intent == "PC_BUILD":
        lines.append("---")
        lines.append("_⚠️ AI explanation unavailable right now — Gemini quota may have refreshed. Try again for detailed build notes._")

    lines.append("---")
    lines.append("💬 _Type any product name for full details, or refine your search._")

    return "\n".join(lines)


# =====================
# PC BUILD ENGINE
# =====================

def get_best_component(products: pd.DataFrame, reviews: pd.DataFrame,
                        category: str, budget: float, workload: list) -> pd.DataFrame:
    """Get best component for a PC build within a budget."""
    cat_products = products[products["category"].str.lower() == category.lower()].copy()

    if len(cat_products) == 0:
        return pd.DataFrame()

    # Filter by budget
    cat_products = cat_products[cat_products["price"] <= budget * 1.10]  # 10% flexibility

    if len(cat_products) == 0:
        cat_products = products[products["category"].str.lower() == category.lower()].copy()
        cat_products = cat_products.sort_values("price").head(3)

    cat_products = rank_products(cat_products)
    return cat_products.head(3)


def check_compatibility(components: dict, compatibility: pd.DataFrame) -> list:
    """Check basic compatibility between components."""
    issues = []
    if compatibility is None:
        return issues

    # Basic socket compatibility (CPU + Motherboard)
    # This is a simplified check — extend with your actual compatibility data
    if "CPU" in components and "Motherboard" in components:
        cpu_id = components["CPU"].iloc[0].get("product_id")
        mobo_id = components["Motherboard"].iloc[0].get("product_id")

        if cpu_id and mobo_id:
            compat_check = compatibility[
                (compatibility.get("product_id_1") == cpu_id) &
                (compatibility.get("product_id_2") == mobo_id)
            ] if "product_id_1" in compatibility.columns else pd.DataFrame()

            if len(compat_check) == 0 and len(compatibility) > 0:
                issues.append("⚠️ CPU-Motherboard compatibility not verified — double-check socket type.")

    return issues


def build_pc(products: pd.DataFrame, reviews: pd.DataFrame, compatibility,
              budget: int, workload: list) -> dict:
    """Build a complete PC within budget."""

    # Adjust splits for workload
    splits = PC_BUDGET_SPLIT.copy()

    if "gaming" in workload:
        splits["GPU"] = 0.40
        splits["CPU"] = 0.15
    elif "ai_ml" in workload or "content_creation" in workload:
        splits["GPU"] = 0.38
        splits["CPU"] = 0.18
    elif "programming" in workload or "office" in workload:
        splits["GPU"] = 0.20
        splits["CPU"] = 0.22

    # Normalize splits to sum to ≤ 0.85 (leave 15% for RAM+SSD+PSU)
    total = sum(splits.values())
    splits = {k: v / total * 0.85 for k, v in splits.items()}

    build = {}
    total_spent = 0
    breakdown = {}

    for component, pct in splits.items():
        comp_budget = budget * pct
        options = get_best_component(products, reviews, component, comp_budget, workload)

        if len(options) > 0:
            best = options.iloc[[0]]
            build[component] = best
            price = best.iloc[0].get("price", 0)
            total_spent += price
            breakdown[component] = {
                "name":    best.iloc[0].get("product_name", ""),
                "price":   price,
                "brand":   best.iloc[0].get("brand", ""),
                "options": options,
            }

    compat_issues = check_compatibility(build, compatibility) if compatibility is not None else []

    return {
        "components":   breakdown,
        "total":        total_spent,
        "budget":       budget,
        "remaining":    budget - total_spent,
        "compat_issues": compat_issues,
        "workload":     workload,
    }


def format_pc_build(build_result: dict) -> str:
    lines = ["## 🖥️ Custom PC Build\n"]
    lines.append(f"**Budget:** ₹{build_result['budget']:,}   **Total:** ₹{int(build_result['total']):,}   **Remaining:** ₹{int(build_result['remaining']):,}\n")

    if build_result.get("workload"):
        lines.append(f"**Optimized for:** {', '.join(build_result['workload']).title()}\n")

    lines.append("| Component | Product | Brand | Price |")
    lines.append("|-----------|---------|-------|-------|")

    for comp, info in build_result["components"].items():
        lines.append(f"| {comp} | {info['name']} | {info['brand']} | ₹{int(info['price']):,} |")

    if build_result.get("compat_issues"):
        lines.append("\n**Compatibility Notes:**")
        for issue in build_result["compat_issues"]:
            lines.append(f"- {issue}")

    return "\n".join(lines)


# =====================
# GEMINI PROMPTS
# =====================

def build_gemini_prompt(query_info: dict, context: str, pc_build: dict = None) -> str:
    intent   = query_info["intent"]
    budget   = query_info["budget"]
    category = query_info["category"]
    workload = query_info["workload"]
    query    = query_info["raw"]

    budget_str   = f"₹{budget:,}" if budget else "Not specified"
    workload_str = ", ".join(workload) if workload else "General"

    if intent == "PC_BUILD" and pc_build:
        pc_context = format_pc_build(pc_build)
        return f"""You are ShopGPT — an expert PC builder who explains builds clearly to both beginners and enthusiasts.

═══════════════════════════════════════
BUILD REQUEST
═══════════════════════════════════════
User Query : {query}
Budget     : {budget_str}
Use Case   : {workload_str}

═══════════════════════════════════════
SELECTED BUILD
═══════════════════════════════════════
{pc_context}

═══════════════════════════════════════
COMPONENT DETAILS
═══════════════════════════════════════
{context}

═══════════════════════════════════════
YOUR RESPONSE INSTRUCTIONS
═══════════════════════════════════════
1. START with: "Here's your ₹{budget_str} {workload_str} PC build! 🖥️"

2. FOR EACH COMPONENT explain:
   - Why THIS specific part was chosen (not just what it is)
   - What it contributes to the build's performance
   - The key spec that matters (e.g. for GPU: VRAM; for CPU: cores/clock; for RAM: speed)
   - Whether it's a bottleneck or the star of the build

3. PERFORMANCE EXPECTATIONS for {workload_str}:
   - What games can it run and at what settings/FPS (if gaming)
   - What workloads can it handle smoothly (if office/editing/AI)
   - Expected performance tier (entry/mid/high-end)

4. VALUE ANALYSIS:
   - Which component gives best value in this build
   - Where the budget was wisely spent
   - Total vs budget — how much is left over and what to do with it

5. UPGRADE PATH:
   - What to upgrade first when budget allows
   - Which components will last longer

6. COMPATIBILITY NOTE:
   - Confirm CPU + Motherboard socket match
   - Confirm PSU wattage is sufficient for GPU + CPU TDP
   - Note if any compatibility issues need checking

7. OPTIONAL ADDITIONS:
   - Suggest a cabinet/case if not in build
   - Suggest cooling if not included
   - Suggest OS if relevant

8. TONE: Enthusiastic, clear, friendly — like a friend who just built their own PC
   Use ₹ for prices. End with: "💬 Want me to swap any component or adjust the budget?"
"""

    comparison_note = (
        "Compare them side by side in a markdown table"
        if intent == "COMPARE"
        else "Give a clear winner recommendation"
    )
    value_note = (
        f"Highlight which product gives best value-for-money within the {budget_str} budget"
        if budget
        else "Highlight the best overall value pick"
    )

    return f"""You are ShopGPT — an expert AI shopping assistant for an electronics store. You give detailed, helpful, honest product advice like a knowledgeable friend who deeply understands tech.

═══════════════════════════════════════
QUERY CONTEXT
═══════════════════════════════════════
Intent        : {intent}
User Query    : {query}
Budget        : {budget_str}
Category      : {category or 'Not specified'}
Use Case      : {workload_str}

═══════════════════════════════════════
PRODUCTS TO CONSIDER
═══════════════════════════════════════
{context}

═══════════════════════════════════════
YOUR RESPONSE INSTRUCTIONS
═══════════════════════════════════════
1. GREETING: Start with a brief 1-line response to the user intent.

2. FOR EACH PRODUCT (top 3-5):
   - Product name as a bold heading with rank number
   - Price in rupees and why it is good value at that price
   - Key specs relevant to the use case (explain what they mean, not just raw numbers)
   - What this product is best for (who should buy it)
   - Pros: at least 3 specific points from specs and reviews
   - Cons: at least 1-2 honest downsides
   - Review sentiment: what real buyers are saying (use the review quotes provided)
   - Rating context: explain what the rating means

3. COMPARISON SUMMARY:
   - After listing products, add a quick section: Which one should you pick?
   - Match to use case (e.g. For office use pick X. For gaming pick Y.)
   - {comparison_note}

4. VALUE FOR MONEY:
   - {value_note}

5. TONE:
   - Conversational and warm, like a knowledgeable friend
   - Use emojis sparingly for readability
   - Use the rupee symbol for all prices
   - Be specific — mention actual spec numbers and actual review quotes
   - Do NOT recommend anything not in the product context above
   - Do NOT make up specs or reviews

6. END with: Want more details on any of these? Just ask!
"""


# =====================
# MAIN CHATBOT FUNCTION
# =====================

def ask_bot(query: str, products: pd.DataFrame, index, model,
            reviews: pd.DataFrame, specs: pd.DataFrame, compatibility) -> str:

    query_info = parse_query(query)
    intent     = query_info["intent"]
    budget     = query_info["budget"]
    category   = query_info["category"]
    workload   = query_info["workload"]

    # ── PC BUILD ──────────────────────────────────────────
    if intent == "PC_BUILD":
        if budget is None:
            return "🖥️ Please specify a budget for your PC build! E.g., *Build gaming PC under 1 lakh*"

        pc_build = build_pc(products, reviews, compatibility, budget, workload)
        pc_summary = format_pc_build(pc_build)

        # Gather context for all selected components
        all_component_rows = []
        for comp_info in pc_build["components"].values():
            all_component_rows.append(comp_info["options"].head(1))

        if all_component_rows:
            combined = pd.concat(all_component_rows)
            combined = rank_products(combined)
            context = create_context(combined, reviews, specs, top_n=len(combined))
        else:
            context = "No components found."

        prompt = build_gemini_prompt(query_info, context, pc_build)

        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return pc_summary + "\n\n---\n\n" + response.text
        except Exception as e:
            print("Gemini Error:", e)
            return pc_summary

    # ── STANDARD SEARCH PIPELINE ─────────────────────────

    # 1. Rewrite query
    try:
        search_query = rewrite_query(query)
    except Exception:
        search_query = query

    # 2. Semantic search
    results = semantic_search(search_query, products, index, model, k=30)

    # 3. Category filter (strict)
    if category:
        results = filter_by_category(results, category)

    # 4. Budget filter
    if budget:
        results = filter_by_budget(results, budget)

    # 5. ML ranking
    results = rank_products(results)

    # 6. Build context
    context = create_context(results, reviews, specs)

    # 7. Gemini explanation
    prompt = build_gemini_prompt(query_info, context)

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        print("Gemini Error:", e)
        return create_local_response(results, reviews, query_info)