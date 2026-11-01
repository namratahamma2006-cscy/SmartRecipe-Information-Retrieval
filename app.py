"""SmartRecipe - Recipe Information Retrieval System (Streamlit UI).

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from smartrecipe import knowledge as kb
from smartrecipe.engine import DEFAULT_WEIGHTS, SmartRecipeEngine, load_recipes
from smartrecipe.evaluation import compare_configs, load_eval_queries
from smartrecipe.preprocessing import apply_phrases, preprocess, tokenize

st.set_page_config(page_title="SmartRecipe", page_icon="🍳", layout="wide",
                   initial_sidebar_state="expanded")

DIET_ICONS = {"Vegetarian": "🥕", "Vegan": "🌱", "Gluten-Free": "🌾",
              "Dairy-Free": "🥛", "Nut-Free": "🥜"}
COURSE_ICONS = {"Main": "🍽️", "Breakfast": "🥞", "Soup": "🍲", "Salad": "🥗",
                "Dessert": "🍰", "Snack": "🥨", "Drink": "🥤", "Side": "🥔"}
EXAMPLES = [
    "quick vegetarian breakfast",
    "spicy indian curry",
    "garbanzo bean dishes",
    "prawn pasta",
    "gluten free chicken dinner",
    "cheesy comfort food without beef",
    "choclate desert",
    "vegan soup under 45 minutes",
]

# --------------------------------------------------------------------------- style
st.markdown("""
<style>
:root { --sr-accent:#ff6b35; --sr-accent2:#f7c548; --sr-green:#2a9d8f; --sr-amber:#e76f51; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
.sr-hero {
  background: linear-gradient(120deg, #ff6b35 0%, #f7931e 45%, #f7c548 100%);
  border-radius: 22px; padding: 1.6rem 2rem; color: #fff; margin-bottom: 1.1rem;
  box-shadow: 0 10px 30px rgba(255,107,53,.25);
}
.sr-hero h1 { color:#fff; font-size: 2.3rem; margin: 0 0 .2rem 0; padding:0; letter-spacing:-.5px; }
.sr-hero p  { margin: 0; font-size: 1.04rem; opacity: .95; }
.sr-card {
  border: 1px solid rgba(128,128,128,.22); border-radius: 18px; padding: 1.05rem 1.2rem .6rem;
  margin-bottom: .2rem; background: rgba(255,255,255,.02);
}
.sr-card h3 { margin: 0 0 .2rem 0; padding: 0; font-size: 1.28rem; }
.sr-rank { display:inline-block; background: var(--sr-accent); color:#fff; border-radius: 10px;
  padding: 0 .5rem; margin-right: .45rem; font-weight: 700; font-size: .95rem; }
.sr-meta { opacity:.8; font-size:.9rem; margin-bottom:.45rem; }
.sr-desc { margin: .25rem 0 .55rem 0; }
.sr-chip { display:inline-block; border-radius: 999px; padding: .1rem .6rem; margin: 0 .3rem .35rem 0;
  font-size: .82rem; border:1px solid rgba(128,128,128,.25); }
.sr-have { background: rgba(42,157,143,.15); border-color: rgba(42,157,143,.5); }
.sr-miss { background: rgba(231,111,81,.12); border-color: rgba(231,111,81,.45); }
.sr-diet { background: rgba(247,197,72,.18); border-color: rgba(247,197,72,.55); }
.sr-intent { background: rgba(255,107,53,.12); border-color: rgba(255,107,53,.45); }
.sr-scorebox { text-align:right; }
.sr-score { font-size: 1.9rem; font-weight: 800; color: var(--sr-accent); line-height: 1; }
.sr-bar { height: 8px; border-radius: 6px; background: rgba(128,128,128,.18); overflow:hidden; margin:.35rem 0 .3rem; }
.sr-bar > div { height:100%; background: linear-gradient(90deg,#ff6b35,#f7c548); }
.sr-sub { font-size:.8rem; opacity:.8; }
.sr-label { font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; opacity:.65; margin:.2rem 0; }
@media (max-width: 640px) {
  .sr-hero { padding: 1.1rem 1.1rem; border-radius: 16px; }
  .sr-hero h1 { font-size: 1.6rem; }
  .sr-scorebox { text-align:left; }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Building TF-IDF index…")
def get_engine() -> SmartRecipeEngine:
    return SmartRecipeEngine(load_recipes())


@st.cache_data(show_spinner="Evaluating retrieval performance…")
def run_evaluation(k: int):
    results = compare_configs(load_recipes(), load_eval_queries(), k)
    summary = pd.DataFrame({name: s for name, (s, _) in results.items()}).T
    per_query = {
        name: pd.DataFrame([{
            "ID": r.qid,
            "Query": r.query or "—",
            "Pantry": ", ".join(r.ingredients) or "—",
            "Tests": r.tests,
            "Relevant": ", ".join(r.relevant),
            "Retrieved": ", ".join(r.retrieved),
            "Precision": r.precision, "Recall": r.recall, "F1": r.f1, "AP": r.ap,
        } for r in rows])
        for name, (_, rows) in results.items()
    }
    return summary, per_query


engine = get_engine()
recipes = engine.recipes
ALL_CUISINES = sorted({r["cuisine"] for r in recipes})
ALL_COURSES = sorted({r["course"] for r in recipes})
INGREDIENT_OPTIONS = sorted({i["name"] for r in recipes for i in r["ingredients"]})

esc = html.escape


def chips(items, css="", icon=""):
    return "".join(f'<span class="sr-chip {css}">{icon}{esc(str(i))}</span>' for i in items)


def set_query(q: str):
    st.session_state["query"] = q


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🧺 My Pantry")
    pantry = st.multiselect(
        "Ingredients you have",
        INGREDIENT_OPTIONS,
        key="pantry",
        accept_new_options=True,
        placeholder="Type or pick… e.g. garbanzo beans",
        help="Pick from the list or type your own. Synonyms (capsicum, prawns, "
             "aubergine, curd…) and small typos are recognised automatically.",
    )
    assume_staples = st.toggle("I have salt, pepper, oil & water", value=True)

    st.markdown("## 🥗 Dietary filters")
    diets = st.pills("Diet", list(kb.DIET_RULES), selection_mode="multi",
                     format_func=lambda d: f"{DIET_ICONS[d]} {d}", label_visibility="collapsed")

    st.markdown("## 🎛️ Refine")
    max_time = st.slider("Max cooking time (min)", 5, 90, 90, step=5)
    sel_cuisines = st.multiselect("Cuisine", ALL_CUISINES)
    sel_courses = st.multiselect("Course", ALL_COURSES,
                                 format_func=lambda c: f"{COURSE_ICONS.get(c, '')} {c}")
    exclude_text = st.text_input("Exclude ingredients", placeholder="e.g. onion, peanut")

    with st.expander("⚖️ Ranking weights & settings"):
        w_text = st.slider("Text similarity (TF-IDF)", 0.0, 1.0, DEFAULT_WEIGHTS["text"], 0.05)
        w_cov = st.slider("Ingredient coverage", 0.0, 1.0, DEFAULT_WEIGHTS["coverage"], 0.05)
        w_use = st.slider("Pantry usage", 0.0, 1.0, DEFAULT_WEIGHTS["usage"], 0.05)
        top_k = st.slider("Max results", 3, 30, 10)
    st.caption("SmartRecipe · TF-IDF + cosine similarity · 50 curated recipes")

# --------------------------------------------------------------------------- header
st.markdown("""
<div class="sr-hero">
  <h1>🍳 SmartRecipe</h1>
  <p>Find the perfect recipe from what you <b>crave</b> and what's in your <b>kitchen</b> —
  powered by TF-IDF, cosine similarity and ingredient intelligence.</p>
</div>
""", unsafe_allow_html=True)

tab_search, tab_eval, tab_data, tab_how = st.tabs(
    ["🔍 Discover", "📊 Evaluation", "📚 Recipe Dataset", "⚙️ How it works"])

# =========================================================================== SEARCH
with tab_search:
    with st.form("search", border=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        query = c1.text_input(
            "What would you like to cook?", key="query",
            placeholder="Try: quick spicy vegan dinner without onion under 30 minutes")
        c2.form_submit_button("Search", type="primary", width="stretch", key="search_btn")

    st.markdown('<div class="sr-label">Try an example</div>', unsafe_allow_html=True)
    ex_cols = st.columns(4)
    for i, ex in enumerate(EXAMPLES):
        ex_cols[i % 4].button(ex, key=f"ex{i}", on_click=set_query, args=(ex,),
                              width="stretch")

    if not query.strip() and not pantry:
        st.info("👆 Type a request in plain English, and/or add the ingredients you have in "
                "the **sidebar**. SmartRecipe understands diets (*vegan*, *gluten free*), time "
                "(*quick*, *under 20 minutes*), exclusions (*without onion*, *no dairy*), "
                "cuisines, courses and ingredient synonyms.", icon="💡")
        st.markdown("#### 🌟 Featured recipes")
        featured = [engine.get(x) for x in ("r05", "r27", "r46", "r33")]
        fcols = st.columns(4)
        for col, r in zip(fcols, featured):
            with col.container(border=True):
                st.markdown(f"**{COURSE_ICONS.get(r['course'], '')} {r['name']}**")
                st.caption(f"{r['cuisine']} · {r['time_minutes']} min · {r['difficulty']}")
                st.markdown(chips([f"{DIET_ICONS[d]} {d}" for d in r["diets"][:3]], "sr-diet"),
                            unsafe_allow_html=True)
    else:
        resp = engine.search(
            query, pantry, diets=diets, max_time=max_time if max_time < 90 else None,
            cuisines=sel_cuisines, courses=sel_courses,
            exclude=[e for e in exclude_text.split(",") if e.strip()],
            top_k=top_k, weights={"text": w_text, "coverage": w_cov, "usage": w_use},
            assume_staples=assume_staples,
        )

        understood = resp.parsed.summary()
        recognised = [f"{k} → {v.replace('_', ' ')}" for k, v in resp.pantry.items()
                      if k.lower().strip() != v.replace("_", " ")]
        if understood or recognised:
            html_bits = chips(understood, "sr-intent", "🧠 ")
            html_bits += chips(recognised, "sr-have", "🔁 ")
            st.markdown(f'<div class="sr-label">SmartRecipe understood</div>{html_bits}',
                        unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Results", len(resp.results))
        m2.metric("Recipes after filters", resp.candidates)
        m3.metric("Pantry items recognised", len(resp.pantry))
        best = resp.results[0].score if resp.results else 0
        m4.metric("Best match", f"{best:.0%}")

        if resp.shopping:
            items = " · ".join(f"**{name}** (+{n})" for name, n in resp.shopping)
            st.success(f"🛒 **Smart shopping tip** — buying one of these completes the most "
                       f"recipes: {items}", icon="🛍️")

        if not resp.results:
            st.warning("No recipes matched. Try removing a filter, widening the time limit "
                       "or using a more general word (e.g. *curry*, *soup*, *pasta*).", icon="🤔")

        for rank, res in enumerate(resp.results, start=1):
            r = res.recipe
            left, right = st.columns([4, 1.1])
            with left:
                diet_html = chips([f"{DIET_ICONS[d]} {d}" for d in r["diets"]], "sr-diet")
                st.markdown(f"""
<div class="sr-card">
  <h3><span class="sr-rank">#{rank}</span>{esc(r['name'])}</h3>
  <div class="sr-meta">{COURSE_ICONS.get(r['course'], '')} {esc(r['course'])} · {esc(r['cuisine'])}
   · ⏱️ {r['time_minutes']} min · 👥 {r['servings']} · 📶 {esc(r['difficulty'])}</div>
  <div class="sr-desc">{esc(r['description'])}</div>
  {diet_html}
</div>""", unsafe_allow_html=True)
            with right:
                sub = []
                if res.text_score is not None:
                    sub.append(f"Text similarity: <b>{res.text_score:.2f}</b>")
                if res.coverage is not None:
                    sub.append(f"Ingredient coverage: <b>{res.coverage:.0%}</b>")
                if res.usage is not None:
                    sub.append(f"Pantry used: <b>{res.usage:.0%}</b>")
                if res.boost:
                    sub.append(f"Intent boost: <b>+{res.boost:.2f}</b>")
                st.markdown(f"""
<div class="sr-scorebox">
  <div class="sr-label">Relevance</div>
  <div class="sr-score">{res.score:.0%}</div>
  <div class="sr-bar"><div style="width:{res.score * 100:.0f}%"></div></div>
  <div class="sr-sub">{'<br>'.join(sub)}</div>
</div>""", unsafe_allow_html=True)

            if pantry:
                ing_html = ""
                if res.matched:
                    ing_html += chips(res.matched, "sr-have", "✅ ")
                if res.missing:
                    ing_html += chips(res.missing, "sr-miss", "🛒 ")
                st.markdown(ing_html, unsafe_allow_html=True)

            with st.expander(f"📖 View recipe — {r['name']}"):
                c_ing, c_steps = st.columns([1, 1.6])
                with c_ing:
                    st.markdown("**Ingredients**")
                    have_set = set(res.matched)
                    for ing in r["ingredients"]:
                        mark = "✅" if ing["name"] in have_set else ("🛒" if pantry and ing["name"] in res.missing else "•")
                        st.markdown(f"{mark} **{ing['name']}** — {ing['quantity']}")
                with c_steps:
                    st.markdown("**Cooking instructions**")
                    for n, step in enumerate(r["instructions"], start=1):
                        st.markdown(f"{n}. {step}")
                if res.substitutions:
                    st.markdown("**🔄 Missing-ingredient substitutions**")
                    for miss, opts in res.substitutions.items():
                        rendered = ", ".join(f"**{o}** ✅ (you have it)" if has else o
                                             for o, has in opts)
                        st.markdown(f"- No **{miss}**? Use {rendered}")
                if res.explanation:
                    st.markdown("**🔎 Why this result** — top TF-IDF terms contributing to "
                                "the cosine score:")
                    st.markdown(chips([f"{t} {s:.3f}" for t, s in res.explanation], "sr-intent"),
                                unsafe_allow_html=True)

# =========================================================================== EVALUATION
with tab_eval:
    st.markdown("### 📊 Retrieval performance")
    st.markdown(
        "SmartRecipe is evaluated on **25 labelled test queries** (`data/eval_queries.json`) "
        "covering synonyms, plurals, spelling mistakes, dietary and time constraints and "
        "pantry-based search. For each query, the top-*k* results are compared with the "
        "manually judged relevant recipes. Three configurations are compared (ablation):")
    st.markdown(
        "- **Baseline TF-IDF** – raw tokens, no lemmatisation/synonyms, cosine similarity only\n"
        "- **+ Preprocessing & Synonyms** – lemmatisation, multi-word phrases, synonym "
        "mapping, field weighting\n"
        "- **SmartRecipe (full)** – plus query understanding (diet/time/exclusion intents, "
        "concept expansion, spelling correction, intent boosts)")
    k = st.select_slider("Cut-off k (top-k results evaluated)", options=[1, 3, 5, 10], value=5)
    summary, per_query = run_evaluation(k)

    best_name = "SmartRecipe (full)"
    b = summary.loc[best_name]
    base = summary.loc["Baseline TF-IDF"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Precision@{k}", f"{b['Precision']:.3f}", f"{b['Precision'] - base['Precision']:+.3f} vs baseline")
    c2.metric(f"Recall@{k}", f"{b['Recall']:.3f}", f"{b['Recall'] - base['Recall']:+.3f} vs baseline")
    c3.metric(f"F1@{k}", f"{b['F1']:.3f}", f"{b['F1'] - base['F1']:+.3f} vs baseline")
    c4.metric("MAP", f"{b['MAP']:.3f}", f"{b['MAP'] - base['MAP']:+.3f} vs baseline")

    cl, cr = st.columns([1.2, 1])
    with cl:
        chart_df = summary[["Precision", "Recall", "F1", "MAP"]].reset_index(names="System") \
            .melt(id_vars="System", var_name="Metric", value_name="Score")
        st.bar_chart(chart_df, x="Metric", y="Score", color="System", stack=False, height=330)
    with cr:
        st.dataframe(summary.style.format("{:.3f}").highlight_max(axis=0, color="#ffd9c7"),
                     width="stretch")
        st.latex(r"P@k=\frac{|Rel\cap Ret_k|}{|Ret_k|}\quad R@k=\frac{|Rel\cap Ret_k|}{|Rel|}")
        st.latex(r"F_1=\frac{2PR}{P+R}")

    st.markdown("#### Per-query results")
    cfg = st.radio("Configuration", list(per_query), index=2, horizontal=True)
    metric_col = {m: st.column_config.ProgressColumn(m, format="%.2f", min_value=0, max_value=1)
                  for m in ("Precision", "Recall", "F1", "AP")}
    st.dataframe(per_query[cfg], column_config=metric_col, width="stretch", hide_index=True)
    st.caption("Note: when a query has fewer than k relevant recipes, Precision@k is capped "
               "below 1 by definition — Recall and MAP are the better indicators there.")

# =========================================================================== DATASET
with tab_data:
    st.markdown("### 📚 Structured recipe dataset")
    df = pd.DataFrame([{
        "ID": r["id"], "Name": r["name"], "Cuisine": r["cuisine"], "Course": r["course"],
        "Time (min)": r["time_minutes"], "Servings": r["servings"], "Difficulty": r["difficulty"],
        "Diets": ", ".join(r["diets"]), "Ingredients": ", ".join(i["name"] for i in r["ingredients"]),
        "# Ingredients": len(r["ingredients"]), "Tags": ", ".join(r["tags"]),
    } for r in recipes])
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Recipes", len(df))
    d2.metric("Cuisines", df["Cuisine"].nunique())
    d3.metric("Unique ingredients", len(engine.known_ingredients))
    d4.metric("Avg. time", f"{df['Time (min)'].mean():.0f} min")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Recipes per cuisine**")
        st.bar_chart(df["Cuisine"].value_counts(), height=260, color="#ff6b35")
    with g2:
        st.markdown("**Recipes per dietary label**")
        st.bar_chart(pd.Series({d: sum(d in r["diets"] for r in recipes) for d in kb.DIET_RULES}),
                     height=260, color="#2a9d8f")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("⬇️ Download dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "smartrecipe_dataset.csv", "text/csv")

# =========================================================================== HOW IT WORKS
with tab_how:
    st.markdown("### ⚙️ How SmartRecipe works")
    st.markdown("""
1. **Text preprocessing** – lowercasing, accent & punctuation removal, tokenisation,
   rule-based lemmatisation (*tomatoes → tomato*), multi-word phrase detection
   (*bell pepper → bell_pepper*), **synonym recognition** (*garbanzo → chickpea*,
   *aubergine → eggplant*, *prawn → shrimp*, *curd → yogurt*) and stop-word removal.
2. **Indexing** – each recipe becomes a weighted document (name ×3, ingredients ×3, tags,
   description, cuisine & course ×2, instructions ×1) and is embedded with a from-scratch
   **TF-IDF** model (sublinear TF, smoothed IDF, L2 normalisation).
3. **Query understanding** – diets, time limits, exclusions (*without onion*, *no dairy*,
   *egg-free*), courses and cuisines are detected; descriptive words are expanded
   (*spicy → chili, garam masala…*); misspellings are corrected against the vocabulary.
4. **Matching** – **cosine similarity** between the query vector and every recipe vector.
5. **Ingredient intelligence** – pantry items are normalised the same way and matched
   exactly or through ingredient families (*pasta ⊃ spaghetti*, *cheese ⊃ cheddar*).
6. **Intelligent ranking** – `score = Σ wᵢ·signalᵢ / Σ wᵢ + intent boosts` over the
   available signals: text similarity, ingredient coverage and pantry usage.
7. **Suggestions** – missing ingredients, substitutions (✅ if you already have the
   substitute) and a shopping tip for the ingredient that completes the most recipes.
""")
    st.markdown("#### 🧪 Try the preprocessing pipeline")
    demo = st.text_input("Enter any text", "Two ripe Aubergines, garbanzo beans & chopped red capsicums")
    s1, s2, s3 = st.columns(3)
    toks = tokenize(demo)
    s1.markdown("**1 · Tokens + lemmas**")
    s1.code(" ".join(toks) or "—", language=None)
    s2.markdown("**2 · Phrases & synonyms**")
    s2.code(" ".join(apply_phrases(toks)) or "—", language=None)
    s3.markdown("**3 · After stop-words**")
    s3.code(" ".join(preprocess(demo)) or "—", language=None)
    with st.expander("📖 Synonym dictionary"):
        st.dataframe(pd.DataFrame(sorted(kb.SYNONYMS.items()), columns=["Variant", "Canonical"]),
                     width="stretch", hide_index=True)
