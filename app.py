import streamlit as st
import pandas as pd
import re
import html as html_mod
from chatbot import (
    ask_bot, load_data, rank_products, create_context,
    detect_language, translate_to_english, translate_from_english,
    SUPPORTED_LANGUAGES, TRANSLATION_AVAILABLE
)

st.set_page_config(
    page_title="Astra — AI Shopping",
    page_icon="🛍️",
    layout="centered"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  VOICE RECOGNITION (language-aware, fixed)
#  st.markdown with unsafe_allow_html renders into the MAIN
#  document, so we use `document`/`window` directly.
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div id="voiceStatus" style="display:none;text-align:center;font-size:12px;
    color:#1d4ed8;background:#eff6ff;border:1px solid #93c5fd;
    border-radius:8px;padding:5px 12px;margin-bottom:6px;font-weight:600;">
    🎙️ Listening… speak now
</div>

<script>
(function() {
    if (window.__astraVoiceInit) return;
    window.__astraVoiceInit = true;

    window.__astraRecognition = null;
    window.__astraListening   = false;

    window.getAstraLang = function() {
        const el = document.getElementById("astra-lang-code");
        return el ? el.innerText.trim() : "en";
    };

    window.astraLangMap = {
        "en":"en-IN","hi":"hi-IN","mr":"mr-IN","ta":"ta-IN","te":"te-IN",
        "kn":"kn-IN","gu":"gu-IN","bn":"bn-IN","ml":"ml-IN","pa":"pa-IN",
        "ur":"ur-PK","ne":"ne-NP","bho":"hi-IN","or":"or-IN","as":"as-IN",
        "sa":"hi-IN","sd":"sd-IN","es":"es-ES","fr":"fr-FR","de":"de-DE",
        "ar":"ar-SA","zh-cn":"zh-CN","ja":"ja-JP","pt":"pt-PT","ru":"ru-RU"
    };

    window.initAstraRecognition = function() {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) { alert("🎙️ Voice search needs Chrome or Edge browser."); return null; }

        const r = new SR();
        r.lang            = window.astraLangMap[window.getAstraLang()] || "en-IN";
        r.continuous      = false;
        r.interimResults  = true;
        r.maxAlternatives = 1;

        r.onstart = function() {
            window.__astraListening = true;
            window.setAstraMicState(true);
        };

        r.onresult = function(e) {
            let interim = "", final = "";
            for (let i = e.resultIndex; i < e.results.length; i++) {
                const t = e.results[i][0].transcript;
                if (e.results[i].isFinal) final += t;
                else interim += t;
            }
            const text = (final || interim).trim();
            if (text) window.fillAstraInput(text);
        };

        r.onerror = function(e) {
            window.setAstraMicState(false);
            if (e.error === "not-allowed") {
                alert("🎙️ Microphone permission denied. Allow mic access in browser settings.");
            } else if (e.error === "network") {
                alert("🎙️ Voice recognition needs internet connection.");
            }
        };

        r.onend = function() {
            window.__astraListening = false;
            window.setAstraMicState(false);
        };

        return r;
    };

    window.toggleAstraVoice = function() {
        if (window.__astraListening) {
            if (window.__astraRecognition) window.__astraRecognition.stop();
            return;
        }
        window.__astraRecognition = window.initAstraRecognition();
        if (window.__astraRecognition) {
            try { window.__astraRecognition.start(); }
            catch (err) { console.warn("Recognition start error:", err); }
        }
    };

    window.setAstraMicState = function(listening) {
        const mic = document.getElementById("astraMicBtn");
        if (mic) {
            mic.innerText = listening ? "🔴" : "🎙️";
            mic.style.background = listening ? "#fee2e2" : "#eff6ff";
            mic.style.borderColor = listening ? "#f87171" : "#93c5fd";
        }
        const status = document.getElementById("voiceStatus");
        if (status) status.style.display = listening ? "block" : "none";
    };

    window.fillAstraInput = function(text) {
        const inputs = document.querySelectorAll('input[type="text"]');
        let filled = false;
        inputs.forEach(function(inp) {
            if (!filled && inp.getBoundingClientRect().width > 150) {
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                setter.call(inp, text);
                inp.dispatchEvent(new Event("input", { bubbles: true }));
                inp.dispatchEvent(new Event("change", { bubbles: true }));
                filled = true;
            }
        });
    };

    window.astraScrollToBottom = function() {
        const boxes = document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
        boxes.forEach(function(b) { b.scrollTo({ top: b.scrollHeight, behavior: "smooth" }); });
    };
})();

setTimeout(window.astraScrollToBottom, 400);
</script>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  DATA LOAD
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Loading Astra… first load takes ~15s")
def get_data():
    return load_data()

try:
    products, index, model, reviews, specs, compatibility = get_data()
except FileNotFoundError as e:
    st.error(f"📂 **Missing file:** {e}")
    st.info("Make sure `products.pkl`, `products.index`, and `ecommerce_data.xlsx` are in the same folder as `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"❌ **Startup error:** {e}")
    st.info("Run `streamlit run app.py` in your terminal to see the full traceback.")
    st.stop()

CATEGORY_COL = "category"

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

CATEGORIES = [
    "Laptop","Smartphone","GPU","CPU","Motherboard",
    "RAM","SSD","HDD","PSU","Monitor","Keyboard",
    "Mouse","Headset","Router","Tablet","Smartwatch"
]

CATEGORY_ALIASES = {
    "phone":"Smartphone","mobile":"Smartphone","iphone":"Smartphone",
    "android":"Smartphone","laptop":"Laptop","notebook":"Laptop",
    "macbook":"Laptop","chromebook":"Laptop","ultrabook":"Laptop",
    "graphics card":"GPU","video card":"GPU","rtx":"GPU","gtx":"GPU",
    "processor":"CPU","ryzen":"CPU","core i":"CPU",
    "hard disk":"HDD","hard drive":"HDD","solid state":"SSD",
    "nvme":"SSD","m.2":"SSD","power supply":"PSU","smps":"PSU",
    "display":"Monitor","screen":"Monitor","gaming monitor":"Monitor",
    "earphones":"Headset","headphones":"Headset","earbuds":"Headset",
    "wifi router":"Router","smartwatch":"Smartwatch",
    "watch":"Smartwatch","tablet":"Tablet","ipad":"Tablet",
}

BRAND_NAMES = [
    "samsung","apple","sony","lg","dell","hp","lenovo","asus",
    "acer","msi","gigabyte","asrock","amd","intel","nvidia",
    "corsair","kingston","crucial","seagate","western digital","wd",
    "logitech","razer","steelseries","hyperx","boat","oneplus",
    "realme","xiaomi","redmi","oppo","vivo","google","motorola",
    "fractal","cooler master","nzxt","thermaltake","evga","zotac",
    "powercolor","sapphire","xfx","palit","maxsun","colorful",
]

WORKLOAD_KEYWORDS = {
    "gaming":           ["gaming","game","fps","esports"],
    "office":           ["office","work","excel","word","business","wfh"],
    "content_creation": ["video editing","editing","blender","premiere","creator"],
    "ai_ml":            ["ai","ml","machine learning","deep learning"],
    "programming":      ["coding","programming","developer","software"],
    "student":          ["student","college","study","school"],
}

SMART_SUGGESTIONS = [
    "Best laptop under ₹50,000 for office",
    "Gaming phone under ₹30,000",
    "Build gaming PC under 1 lakh",
    "Best wireless headset under ₹5,000",
    "Compare Samsung vs OnePlus phones",
    "Best SSD for speed under ₹5,000",
]

# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def extract_budget(text):
    t = text.lower().replace(",","")
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|l\b)', t)
    if m: return int(float(m.group(1)) * 100_000)
    m = re.search(r'(\d+(?:\.\d+)?)\s*k\b', t)
    if m: return int(float(m.group(1)) * 1_000)
    nums = re.findall(r'\d{4,}', t)
    if nums: return max(int(n) for n in nums)
    return None

def extract_category(text):
    t = text.lower()
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in t: return cat
    for cat in CATEGORIES:
        if cat.lower() in t: return cat
    return None

def extract_brand(text):
    t = text.lower()
    for b in BRAND_NAMES:
        if re.search(r'\b' + re.escape(b) + r'\b', t): return b.title()
    return None

def extract_workload(text):
    t = text.lower()
    for wl, kws in WORKLOAD_KEYWORDS.items():
        if any(kw in t for kw in kws): return wl
    return None

def is_only_budget(text):
    t = text.strip().lower()
    cleaned = re.sub(r'\d[\d,\.]*\s*(lakh|lac|k\b)?', '', t)
    cleaned = re.sub(r'\b(under|below|around|upto|up to|within|budget|rs|inr|₹)\b', '', cleaned)
    return len(cleaned.strip()) < 5

def is_only_category(text):
    cat = extract_category(text)
    return (cat is not None and extract_budget(text) is None
            and extract_brand(text) is None and extract_workload(text) is None)

def looks_like_product_name(text, products_df):
    t = text.lower().strip()
    if len(t) < 5 or "product_name" not in products_df.columns: return False
    for name in products_df["product_name"].dropna():
        if t in name.lower() or name.lower() in t: return True
    return False

# ═══════════════════════════════════════════════════════════
#  SEARCH + FORMAT
# ═══════════════════════════════════════════════════════════

def strict_budget_search(category, budget, brand, workload, products_df, reviews_df, specs_df, top_n=5):
    df = products_df.copy()
    if category and "category" in df.columns:
        df = df[df["category"].str.lower() == category.lower()]
    if brand and "brand" in df.columns:
        df = df[df["brand"].str.lower() == brand.lower()]
    if budget and "price" in df.columns:
        df = df[df["price"] <= budget]
    if df.empty: return df
    df = rank_products(df)
    if budget and "price" in df.columns:
        df = df.copy()
        df["_ps"] = df["price"] / budget
        df["_fs"] = df.get("_ps", 0) * 0.60 + df.get("ml_score", 0) * 0.40
        df = df.sort_values("_fs", ascending=False)
        df = df.drop(columns=[c for c in ["_ps","_fs"] if c in df.columns])
    return df.head(top_n)


def format_search_results(df, reviews_df, budget, category, workload, brand, products_df=None):
    if df.empty:
        return (
            f"😕 No **{category or 'products'}** found under **₹{budget:,}**.\n\n"
            f"Try increasing your budget or removing brand filters."
        )

    header = f"Here are the best **{category or 'Products'}**"
    if budget:   header += f" under ₹{budget:,}"
    if workload: header += f" for **{workload}**"
    if brand:    header += f" from **{brand}**"
    lines = [header + ":\n"]

    for i, (_, row) in enumerate(df.iterrows(), 1):
        pid   = row.get("product_id")
        price = row.get("price", 0)
        try:    ps = f"₹{int(price):,}"
        except: ps = f"₹{price}"

        rev = reviews_df[reviews_df["product_id"] == pid] if reviews_df is not None else pd.DataFrame()
        rc  = len(rev)
        if rc:
            avg_r    = round(rev["rating"].mean(), 1)
            five_pct = round(len(rev[rev["rating"]==5]) / rc * 100)
        else:
            avg_r    = row.get("rating","N/A")
            five_pct = 0

        try:
            rv = float(avg_r)
            rl = "Excellent ✨" if rv>=4.5 else "Very Good" if rv>=4 else "Good" if rv>=3.5 else "Average"
        except: rl = ""

        sigs = []
        if row.get("trend_score",0)>=0.7:     sigs.append("🔥 Trending")
        if row.get("monthly_sales",0)>=100:    sigs.append(f"📦 {int(row['monthly_sales'])}+/mo")
        if row.get("popularity_score",0)>=0.8: sigs.append("⭐ Popular")
        ml = row.get("ml_score",0)
        if ml>=0.8:   sigs.append("🏆 Top Pick")
        elif ml>=0.6: sigs.append("👍 Recommended")

        spec_line = ""
        if specs is not None and "product_id" in specs.columns:
            ps2 = specs[specs["product_id"]==pid]
            if len(ps2):
                items = [f"**{c}:** {ps2.iloc[0][c]}" for c in ps2.columns
                         if c!="product_id" and pd.notna(ps2.iloc[0][c])
                         and str(ps2.iloc[0][c]).strip() not in ("","nan","None")]
                spec_line = " &nbsp;|&nbsp; ".join(items[:6])

        pq = [str(x)[:120] for x in rev[rev["rating"]>=4]["review_text"].dropna().head(2).tolist()] if rc and "review_text" in rev.columns else []
        nq = [str(x)[:120] for x in rev[rev["rating"]<=2]["review_text"].dropna().head(1).tolist()] if rc and "review_text" in rev.columns else []

        lines.append(f"### {i}. {row.get('product_name','Unknown')}")
        lines.append(
            f"**{ps}** &nbsp;|&nbsp; ⭐ {avg_r}/5 *({rl})* &nbsp;|&nbsp; "
            f"🏷️ {row.get('brand','N/A')} &nbsp;|&nbsp; 💬 {rc} reviews"
            + (f" &nbsp;|&nbsp; 🌟 {five_pct}% gave 5★" if five_pct else "")
        )
        if sigs: lines.append("  ".join(sigs))
        desc = str(row.get("description", "")).strip()
        if desc and desc.lower() not in ("nan","none",""): lines.append(f"\n_{desc[:220]}_")
        if spec_line: lines.append(f"\n📋 **Specs:** {spec_line}")
        if pq:
            lines.append("\n✅ **Buyers say:**")
            for q in pq: lines.append(f'- _"{q}"_')
        if nq:
            lines.append("\n⚠️ **Watch out:**")
            for q in nq: lines.append(f'- _"{q}"_')
        lines.append("")

    if len(df)>=2:
        best = df.iloc[0]
        lines.append("---")
        lines.append(f"💡 **My pick:** **{best.get('product_name','')}** — best value in this range.")
        if workload: lines.append(f"Well-suited for **{workload}** use.")

    if budget and products_df is not None and "price" in products_df.columns and category:
        nudge = products_df[
            (products_df["category"].str.lower()==category.lower()) &
            (products_df["price"]>budget) & (products_df["price"]<=budget*1.15)
        ]
        nudge = rank_products(nudge).head(2) if not nudge.empty else pd.DataFrame()
        if not nudge.empty:
            lines.append(f"\n💰 **Slightly above budget** _(up to ₹{int(budget*1.15):,})_:")
            for _, row in nudge.iterrows():
                try: p = f"₹{int(row.get('price',0)):,}"
                except: p = str(row.get('price',''))
                lines.append(f"- **{row.get('product_name','')}** — {p} ({row.get('brand','')})")

    lines.append("\n💬 _Type any product name for full details, or ask me to compare!_")
    return "\n".join(lines)


def build_product_detail_reply(text, products_df, reviews_df):
    t = text.lower().strip()
    matches = products_df[products_df["product_name"].str.lower().str.contains(t, na=False)]
    if matches.empty: return None

    product = matches.iloc[0]
    price   = product.get("price", 0)
    cat     = product.get("category","")
    pid     = product.get("product_id")
    try:    ps = f"₹{int(price):,}"
    except: ps = f"₹{price}"

    rev_rows = reviews_df[reviews_df["product_id"]==pid] if reviews_df is not None else pd.DataFrame()
    rc = len(rev_rows)
    if rc:
        avg_r    = round(rev_rows["rating"].mean(),1)
        five_pct = round(len(rev_rows[rev_rows["rating"]==5])/rc*100)
        one_pct  = round(len(rev_rows[rev_rows["rating"]==1])/rc*100)
        verified = int(rev_rows["verified_purchase"].sum()) if "verified_purchase" in rev_rows.columns else 0
    else:
        avg_r = product.get("rating","N/A"); five_pct=one_pct=verified=0

    try:
        rv=float(avg_r)
        rl="Excellent ✨" if rv>=4.5 else "Very Good" if rv>=4 else "Good" if rv>=3.5 else "Average" if rv>=3 else "Below Average"
    except: rl=""

    spec_lines=[]
    if specs is not None and "product_id" in specs.columns:
        ps2=specs[specs["product_id"]==pid]
        if len(ps2):
            for col in ps2.columns:
                if col!="product_id":
                    val=ps2.iloc[0][col]
                    if pd.notna(val) and str(val).strip() not in ("","nan","None"):
                        spec_lines.append(f"| {col} | {val} |")

    lines=[f"## {product.get('product_name','')}"]
    lines.append(f"**Brand:** {product.get('brand','N/A')} &nbsp;|&nbsp; **Category:** {cat} &nbsp;|&nbsp; **Price:** {ps}")
    lines.append(f"**Rating:** ⭐ {avg_r}/5 *({rl})* — {rc} reviews"
        +(f" &nbsp;|&nbsp; 🌟 {five_pct}% gave 5★" if five_pct else "")
        +(f" &nbsp;|&nbsp; ✅ {verified} verified" if verified else ""))
    lines.append("")
    desc=str(product.get("description","")).strip()
    if desc and desc.lower() not in ("nan","none",""): lines.append(desc); lines.append("")
    if spec_lines:
        lines.append("### 📋 Specifications")
        lines.append("| Feature | Value |"); lines.append("|---------|-------|")
        lines.extend(spec_lines); lines.append("")
    if rc:
        lines.append("### 📊 Review Breakdown")
        lines.append(f"- 🌟 5-star: **{five_pct}%**")
        lines.append(f"- 💔 1-star: **{one_pct}%**")
        if verified: lines.append(f"- ✅ Verified: **{verified}**")
        lines.append("")
    if rc and "review_text" in rev_rows.columns:
        pos=rev_rows[rev_rows["rating"]>=4]["review_text"].dropna()
        neg=rev_rows[rev_rows["rating"]<=2]["review_text"].dropna()
        if len(pos):
            lines.append("**✅ What buyers love:**")
            for r in pos.head(3): lines.append(f'- _"{str(r)[:150]}"_')
            lines.append("")
        if len(neg):
            lines.append("**⚠️ Common complaints:**")
            for r in neg.head(2): lines.append(f'- _"{str(r)[:150]}"_')
            lines.append("")

    low=price*0.95
    alts=products_df[
        (products_df["category"]==cat)&(products_df["price"]>=low)&
        (products_df["product_name"]!=product["product_name"])
    ].sort_values("price").head(4)
    if not alts.empty:
        lines.append("---\n### 🔄 How it compares")
        lines.append(f"_Same category · Price ≥ ₹{int(low):,}_\n")
        lines.append("| | Product | Brand | Price | Rating | Reviews |")
        lines.append("|--|---------|-------|-------|--------|---------|")
        lines.append(f"| ⬅ *this* | **{product.get('product_name','')}** | {product.get('brand','')} | {ps} | ⭐ {avg_r} | {rc} |")
        for _,alt in alts.iterrows():
            apid=alt.get("product_id")
            arev=reviews_df[reviews_df["product_id"]==apid] if reviews_df is not None else pd.DataFrame()
            ar=round(arev["rating"].mean(),1) if len(arev) else alt.get("rating","N/A")
            try: ap=f"₹{int(alt.get('price',0)):,}"
            except: ap=str(alt.get('price',''))
            lines.append(f"|  | {alt.get('product_name','')} | {alt.get('brand','')} | {ap} | ⭐ {ar} | {len(arev)} |")
        lines.append("\n_Type any product name above for its full details._")
    return "\n".join(lines)


def reply_ask_category(budget):
    amt  = f"₹{budget:,}" if budget else "your"
    cats = " · ".join([f"`{c}`" for c in CATEGORIES[:8]])
    return (
        f"Got it — budget of **{amt}**! 🎯\n\n"
        f"What category are you looking for?\n\n{cats}\n\n"
        f"**Or describe what you need:**\n"
        f"- *\"Gaming laptop\"* — high-performance gaming\n"
        f"- *\"Office smartphone\"* — work & productivity\n"
        f"- *\"Build a gaming PC\"* — full custom desktop\n\n"
        f"You can also add a brand: *\"Samsung phone\"*, *\"Dell laptop\"*"
    )

def reply_ask_budget(category):
    cat = category or "that"
    return (
        f"**{cat}** — great choice! 💡\n\n"
        f"What's your budget?\n\n"
        f"| Range | What you get |\n|-------|-------------|\n"
        f"| Under ₹20,000 | Entry-level |\n"
        f"| ₹20,000–₹50,000 | Mid-range |\n"
        f"| ₹50,000–₹1,00,000 | High performance |\n"
        f"| Above ₹1 lakh | Flagship / Pro |\n\n"
        f"You can also mention **brand** or **use-case** _(gaming, office, editing…)_"
    )

# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════

def _init():
    defaults = {
        "chat_history": [{"role":"ai","text":(
            "👋 Hi! I'm **Astra** — your AI shopping assistant.\n\n"
            "I give **detailed recommendations** with specs, buyer reviews, ratings & honest pros/cons.\n\n"
            "| What you want | Example |\n|---|---|\n"
            "| 🔍 Recommendations | *Best laptop under ₹50,000 for office* |\n"
            "| 🖥️ PC builds | *Build gaming PC under 1 lakh* |\n"
            "| 📊 Comparisons | *Samsung vs OnePlus phones* |\n"
            "| 🏷️ Product details | *Tell me about Dell Inspiron 15* |\n"
            "| 💰 Budget search | *50000* → I'll ask what you need |\n\n"
            "**Tip:** Tap 🎙️ for voice search · Pick your language below 🌐\n\n"
            "**What are you looking for today?**"
        )}],
        "context":          {},
        "show_suggestions": True,
        "filter_cat":       None,
        "filter_brand":     None,
        "language":         "en",
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ═══════════════════════════════════════════════════════════
#  PROCESS QUERY  (with translation pipeline)
# ═══════════════════════════════════════════════════════════

def process_query(raw_q, manual_lang=None):
    q_original = raw_q.strip()
    if not q_original: return

    # ── 1. Determine language ───────────────────────────────
    if manual_lang and manual_lang != "auto":
        user_lang = manual_lang
    else:
        user_lang = detect_language(q_original)
    st.session_state.language = user_lang

    # ── 2. Translate to English for internal processing ────
    q = translate_to_english(q_original, user_lang) if user_lang != "en" else q_original

    st.session_state.show_suggestions = False
    st.session_state.chat_history.append({"role":"user","text":q_original})

    budget   = extract_budget(q)
    category = extract_category(q)
    brand    = extract_brand(q)
    workload = extract_workload(q)

    ctx = st.session_state.context
    if budget:   ctx["budget"]   = budget
    if category: ctx["category"] = category
    if brand:    ctx["brand"]    = brand
    if workload: ctx["workload"] = workload
    st.session_state.context = ctx

    r_budget   = ctx.get("budget")
    r_category = ctx.get("category")
    r_brand    = ctx.get("brand")
    r_workload = ctx.get("workload")

    if st.session_state.filter_cat   and not r_category: r_category = st.session_state.filter_cat
    if st.session_state.filter_brand and not r_brand:    r_brand    = st.session_state.filter_brand

    reply = None  # English reply, translated at the end

    # PC BUILD
    pc_kws = ["build","assemble","gaming pc","custom pc","desktop pc"]
    if any(kw in q.lower() for kw in pc_kws):
        if not r_budget:
            reply = "What's your total budget for the PC build? e.g. *under 1 lakh*, *80000*"
        else:
            with st.spinner("🔧 Designing your PC build..."):
                enriched = f"build gaming pc under {r_budget}"
                if r_workload: enriched += f" for {r_workload}"
                reply = str(ask_bot(enriched, products, index, model, reviews, specs, compatibility))

    # ONLY BUDGET
    elif is_only_budget(q) and not r_category:
        reply = reply_ask_category(r_budget)

    # ONLY CATEGORY
    elif is_only_category(q) and not r_budget:
        reply = reply_ask_budget(r_category)

    # PRODUCT DETAIL
    elif looks_like_product_name(q, products) and not r_budget:
        reply = build_product_detail_reply(q, products, reviews)

    # FULL SEARCH
    if reply is None and r_budget and r_category:
        with st.spinner("🔍 Finding the best options for you..."):
            filtered = strict_budget_search(r_category, r_budget, r_brand, r_workload, products, reviews, specs)
            reply = format_search_results(filtered, reviews, r_budget, r_category, r_workload, r_brand, products_df=products)

    # FALLBACK
    if reply is None:
        enriched = q
        if r_budget   and str(r_budget) not in q:            enriched += f" under {r_budget}"
        if r_category and r_category.lower() not in q.lower(): enriched += f" {r_category}"
        with st.spinner("🔍 Searching products..."):
            reply = str(ask_bot(enriched, products, index, model, reviews, specs, compatibility))

    # ── 3. Translate reply back to user's language ──────────
    if user_lang != "en":
        with st.spinner(f"🌐 Translating to {SUPPORTED_LANGUAGES.get(user_lang, user_lang)}..."):
            reply = translate_from_english(reply, user_lang)

    st.session_state.chat_history.append({"role":"ai","text":reply})


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

# ── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="app-title">🛍️ Astra</div>
<div style="text-align:center;font-size:12px;color:#64748b;margin-top:-8px;margin-bottom:10px;letter-spacing:.5px;">
    AI Shopping Assistant &nbsp;·&nbsp; 🎙️ Voice &nbsp;·&nbsp; 🌐 Multilingual
</div>
""", unsafe_allow_html=True)

# Hidden element so voice JS can read the selected language code
st.markdown(f"<span id='astra-lang-code' style='display:none'>{st.session_state.language}</span>", unsafe_allow_html=True)

# ── LANGUAGE SELECTOR ─────────────────────────────────────
st.markdown("<div class='section-title' style='margin-top:6px'>🌐 Language</div>", unsafe_allow_html=True)

lang_options = ["🌐 Auto-detect"] + [f"{name}" for name in SUPPORTED_LANGUAGES.values()]
lang_codes   = ["auto"] + list(SUPPORTED_LANGUAGES.keys())
current_idx  = lang_codes.index(st.session_state.language) if st.session_state.language in lang_codes else 0

selected_lang_name = st.selectbox(
    "Language", lang_options, index=current_idx,
    label_visibility="collapsed", key="lang_select"
)
selected_lang_code = lang_codes[lang_options.index(selected_lang_name)]

if not TRANSLATION_AVAILABLE:
    st.warning("⚠️ Translation not available. Run: `pip install deep-translator langdetect`")

# ── ACTIVE FILTER CHIPS ───────────────────────────────────
active_filters = []
if st.session_state.filter_cat:              active_filters.append((f"📦 {st.session_state.filter_cat}","cat"))
if st.session_state.filter_brand:            active_filters.append((f"🏷️ {st.session_state.filter_brand}","brand"))
if st.session_state.context.get("budget"):   active_filters.append((f"💰 ₹{st.session_state.context['budget']:,}","budget"))
if st.session_state.context.get("workload"): active_filters.append((f"🎮 {st.session_state.context['workload']}","workload"))

if active_filters:
    st.markdown("<div class='section-title'>Active Filters <span style='font-weight:400;color:#64748b'>(click ✕ to remove)</span></div>", unsafe_allow_html=True)
    fc = st.columns(len(active_filters)+1, gap="small")
    for i,(label,ftype) in enumerate(active_filters):
        with fc[i]:
            if st.button(f"{label} ✕", key=f"chip_{ftype}", use_container_width=True):
                if ftype=="cat":      st.session_state.filter_cat = None
                elif ftype=="brand":  st.session_state.filter_brand = None
                elif ftype=="budget": st.session_state.context.pop("budget",None)
                elif ftype=="workload": st.session_state.context.pop("workload",None)
                st.rerun()

# ── QUICK ACTIONS ─────────────────────────────────────────
st.markdown("<div class='section-title'>Quick Actions</div>", unsafe_allow_html=True)

cats_list = products[CATEGORY_COL].dropna().unique().tolist()[:6] if CATEGORY_COL in products.columns else []
if cats_list:
    cc = st.columns(len(cats_list), gap="small")
    for i,cat in enumerate(cats_list):
        with cc[i]:
            if st.button(cat, key=f"cat_{i}", use_container_width=True):
                process_query(f"Recommend best {cat}", manual_lang=selected_lang_code); st.rerun()

b1,b2,b3,b4 = st.columns(4, gap="small")
with b1:
    if st.button("🔥 Trending",  use_container_width=True, key="trending"):
        process_query("Trending products this month", manual_lang=selected_lang_code); st.rerun()
with b2:
    if st.button("💰 Budget",    use_container_width=True, key="budget_btn"):
        process_query("Best budget products under 20000", manual_lang=selected_lang_code); st.rerun()
with b3:
    if st.button("⭐ Top Rated", use_container_width=True, key="toprated"):
        process_query("Top rated products", manual_lang=selected_lang_code); st.rerun()
with b4:
    if st.button("🖥️ Build PC", use_container_width=True, key="buildpc"):
        process_query("Build gaming PC under 100000", manual_lang=selected_lang_code); st.rerun()

# ── CHAT BOX ──────────────────────────────────────────────
with st.container(border=True):
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            safe = html_mod.escape(msg["text"])
            st.markdown(
                f"<div class='user-row'><div class='user-bubble'>🧑 {safe}</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("<div class='ai-label'>🤖 Astra</div>", unsafe_allow_html=True)
            st.markdown(msg["text"])
            st.markdown("<div class='msg-gap'></div>", unsafe_allow_html=True)

# ── SMART SUGGESTIONS ─────────────────────────────────────
if st.session_state.show_suggestions:
    st.markdown("<div class='section-title'>✨ Try asking</div>", unsafe_allow_html=True)
    sc = st.columns(2, gap="small")
    for i,sug in enumerate(SMART_SUGGESTIONS):
        with sc[i%2]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                process_query(sug, manual_lang=selected_lang_code); st.rerun()

# ── CATEGORY FILTER CHIPS ────────────────────────────────
st.markdown("<div class='section-title' style='margin-top:6px'>🔎 Filter by Category</div>", unsafe_allow_html=True)
fcc = st.columns(8, gap="small")
filter_cats = ["Laptop","Smartphone","GPU","Monitor","SSD","RAM","Headset","Tablet"]
for i,fc_name in enumerate(filter_cats):
    with fcc[i]:
        active = st.session_state.filter_cat == fc_name
        if st.button(f"✓{fc_name}" if active else fc_name, key=f"fcat_{i}", use_container_width=True):
            st.session_state.filter_cat = None if active else fc_name
            st.rerun()

# ── INPUT ROW (voice button is plain HTML — no rerun on click) ──
st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns([5,1,1,1], gap="small")

with col1:
    query = st.text_input(
        "Search", placeholder="Ask anything… or tap 🎙️ to speak (any language)",
        label_visibility="collapsed", key="user_input"
    )

with col2:
    st.markdown("""
    <button id="astraMicBtn" onclick="toggleAstraVoice()" title="Voice search"
        style="width:100%;height:36px;border-radius:999px;font-size:15px;cursor:pointer;
               background:#eff6ff;border:1.5px solid #93c5fd;transition:.2s;
               display:flex;align-items:center;justify-content:center;">🎙️</button>
    """, unsafe_allow_html=True)

with col3:
    send = st.button("🚀", use_container_width=True, key="send_btn", help="Send")

with col4:
    if st.button("🗑️", use_container_width=True, key="clear_btn", help="Clear chat"):
        st.session_state.chat_history = [{"role":"ai","text":"👋 Chat cleared! What are you looking for?"}]
        st.session_state.context          = {}
        st.session_state.show_suggestions = True
        st.session_state.filter_cat       = None
        st.session_state.filter_brand     = None
        st.rerun()

if send and query.strip():
    process_query(query, manual_lang=selected_lang_code)
    st.rerun()