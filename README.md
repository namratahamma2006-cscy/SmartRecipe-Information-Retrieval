# 🍳 SmartRecipe – Recipe Information Retrieval System

SmartRecipe finds recipes that fit a **plain-English request** and the **ingredients you have**.
It uses a TF-IDF vector space model with cosine similarity, written from scratch, plus
ingredient knowledge: synonyms, ingredient families, dietary rules and substitutions.

> **Live demo:** `https://<your-app-name>.streamlit.app` *(see [Deploy](#-deploy-to-streamlit-community-cloud))*

## ✨ Features

| Feature | What it does |
|---|---|
| 🔍 **Natural-language search** | *"quick spicy vegan curry without onion under 25 minutes"* |
| 🧺 **Pantry search** | Ranks recipes by how much of each one you can make with what you have |
| 🔁 **Synonym recognition** | garbanzo→chickpea, aubergine→eggplant, prawn→shrimp, capsicum→bell pepper, curd→yogurt, coriander→cilantro, … |
| ✏️ **Spelling correction** | *"choclate desert"* → chocolate dessert; *"tomatos"* → tomato |
| 🧠 **Query understanding** | Detects diets, time limits, exclusions (*without*, *no*, *-free*), courses and cuisines; expands words like *spicy*, *creamy* and *cheesy* |
| 🥗 **Dietary filters** | Vegetarian, Vegan, Gluten-Free, Dairy-Free, Nut-Free (worked out from the ingredients) |
| 🛒 **Missing ingredients** | Shows what you have ✅ and what you're missing 🛒, with substitutes (and a mark when you already own the substitute) |
| 🛍️ **Smart shopping tip** | Names the single ingredient that would complete the most recipes |
| 📊 **Evaluation dashboard** | Precision@k, Recall@k, F1@k and MAP for three systems, compared side by side |
| 🔎 **Explainability** | Score breakdown and the TF-IDF terms behind each match |

## 🏗️ Architecture

```
app.py                      Streamlit UI (Discover · Evaluation · Dataset · How it works)
smartrecipe/
  preprocessing.py          tokenise → lemmatise → phrases & synonyms → stop-words
  knowledge.py              synonyms, families, diet categories, substitutes, concepts
  tfidf.py                  TF-IDF + cosine similarity in NumPy
  query.py                  natural-language query parser (intents, expansion, spelling)
  engine.py                 filtering, ingredient matching, ranking, suggestions
  evaluation.py             Precision / Recall / F1 / MAP, comparison of three systems
data/recipes.json           50 structured recipes, 11 cuisines
data/eval_queries.json      25 labelled test queries with relevance judgements
tests/                      67 pytest tests (unit + headless Streamlit UI tests)
```

### Ranking model

* **TF-IDF:** `tf = 1 + log(count)`, `idf = log((1+N)/(1+df)) + 1`, then L2-normalised.
  Fields are weighted: name ×3, ingredients ×3, tags/description/cuisine/course ×2, instructions ×1.
* **Signals** (each between 0 and 1): text cosine similarity, *ingredient coverage* (share of the
  recipe's ingredients you have) and *pantry usage* (share of your ingredients the recipe uses).
* **Final score:** `Σ wᵢ·signalᵢ / Σ wᵢ` over the signals that are present, plus small boosts when
  the course, cuisine or "easy" intent matches. You can change the weights in the sidebar.

## 📊 Evaluation results

There are 25 test queries. Each one checks something specific: synonyms, plurals, typos, diets, time
limits, cuisines or pantry matching. Relevance was judged by hand from the recipe content.

| System (k = 5) | Precision@5 | Recall@5 | F1@5 | MAP |
|---|---|---|---|---|
| Baseline TF-IDF (raw tokens) | 0.638 | 0.843 | 0.676 | 0.791 |
| + Preprocessing & Synonyms | 0.708 | 0.943 | 0.768 | 0.919 |
| **SmartRecipe (full)** | **0.752** | **0.973** | **0.813** | **0.973** |

Precision@5 can't reach 1.0 for queries with fewer than 5 relevant recipes, so Recall and MAP
show ranking quality better. To reproduce the numbers: `python -m smartrecipe.evaluation`.

## 🚀 Run locally

```bash
git clone https://github.com/namratahamma2006-cscy/SmartRecipe-Information-Retrieval.git
cd SmartRecipe-Information-Retrieval
pip install -r requirements.txt
streamlit run app.py
```

## ✅ Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

GitHub Actions runs the tests on every push (`.github/workflows/tests.yml`).

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (it must be **public**).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Fill in: Repository `namratahamma2006-cscy/SmartRecipe-Information-Retrieval`, Branch `main`,
   Main file path `app.py`. Under *Advanced settings*, pick Python **3.12**.
5. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a public
   `*.streamlit.app` URL. Paste that URL into the *Live demo* line above.

## 💬 Example queries

* `quick vegetarian breakfast`
* `garbanzo bean dishes`
* `spicy indian curry`
* `gluten free chicken dinner`
* `cheesy comfort food without beef`
* `vegan soup under 45 minutes`
* Pantry only: *eggs, bread, avocado* → Avocado Toast with Egg
