"""
DecodeLabs - AI Project 3: Tech Stack Recommender
Content-based filtering with TF-IDF + Cosine Similarity (pure Python, no ML libs).

Pipeline:  1. Ingestion -> 2. Scoring -> 3. Sorting -> 4. Filtering (Top-N)
Usage:
    python recommender.py                       # interactive
    python recommender.py python cloud automation   # command-line skills
"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

DATA_FILE = Path(__file__).with_name("raw_skills.csv")
TOP_N = 3
MIN_INPUTS = 3

# Shared vocabulary guard: map user wording onto the dataset's exact vocabulary
ALIASES = {
    "cloud computing": "cloud", "ml": "machine learning", "ai": "machine learning",
    "dl": "deep learning", "js": "javascript", "node": "node.js", "nodejs": "node.js",
    "k8s": "kubernetes", "cicd": "ci/cd", "ci cd": "ci/cd", "frontend": "html",
    "frontend development": "web design", "ui": "ui/ux", "ux": "ui/ux",
    "scikit learn": "scikit-learn", "sklearn": "scikit-learn", "powerbi": "power bi",
    "api": "apis", "rest": "apis", "pen testing": "penetration testing",
    "ds": "data structures", "dsa": "algorithms", "tf": "tensorflow",
}

# Cold-start fallback ("Trending"): shown when the user vector is all zeros
TRENDING = ["Software Engineer", "Data Scientist", "DevOps Engineer"]


# ---------- Data loading ----------
def load_roles(path=DATA_FILE):
    roles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roles[row["role"]] = [s.strip().lower() for s in row["skills"].split(";") if s.strip()]
    return roles


# ---------- TF-IDF ----------
def build_vocab(roles):
    return sorted({s for skills in roles.values() for s in skills})


def compute_idf(roles, vocab):
    """Smoothed IDF: log((1+N)/(1+df)) + 1  -> never zero, still penalises common skills."""
    n = len(roles)
    df = Counter(s for skills in roles.values() for s in set(skills))
    return {t: math.log((1 + n) / (1 + df[t])) + 1 for t in vocab}


def tfidf_vector(terms, vocab, idf):
    """TF = count(t)/total terms ; weight = TF * IDF ; returned in vocab order."""
    counts, total = Counter(terms), len(terms)
    if total == 0:
        return [0.0] * len(vocab)
    return [(counts[t] / total) * idf[t] if t in counts else 0.0 for t in vocab]


# ---------- Cosine similarity ----------
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


# ---------- Pipeline ----------
def normalise(raw_inputs, vocab):
    """Step 1 (Ingestion): clean text, resolve aliases, split known/unknown skills."""
    known, unknown = [], []
    for s in raw_inputs:
        s = s.strip().lower()
        s = ALIASES.get(s, s)
        if not s:
            continue
        (known if s in vocab else unknown).append(s)
    return list(dict.fromkeys(known)), unknown  # de-duplicate, keep order


def recommend(user_skills, roles, top_n=TOP_N):
    vocab = build_vocab(roles)
    idf = compute_idf(roles, vocab)
    known, unknown = normalise(user_skills, set(vocab))

    user_vec = tfidf_vector(known, vocab, idf)

    # Step 2: Scoring
    scored = [(role, cosine_similarity(user_vec, tfidf_vector(skills, vocab, idf)))
              for role, skills in roles.items()]
    # Step 3: Sorting (descending)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Cold start: no recognised skills -> trending fallback
    if not known or scored[0][1] == 0:
        return [(r, 0.0) for r in TRENDING[:top_n]], known, unknown, True

    # Step 4: Filtering (Top-N)
    return scored[:top_n], known, unknown, False


def display(results, known, unknown, fallback, roles):
    print("\n" + "=" * 52)
    print("  TECH STACK RECOMMENDER - TOP MATCHES")
    print("=" * 52)
    if unknown:
        print(f"  (Skipped unknown skills: {', '.join(unknown)})")
    if fallback:
        print("  Cold start: no matching skills found. Showing trending roles.\n")
    for i, (role, score) in enumerate(results, 1):
        pct = f"{score * 100:5.1f}% match" if not fallback else "trending"
        print(f"  {i}. {role:<28} {pct}")
        matched = [s for s in roles[role] if s in known]
        missing = [s for s in roles[role] if s not in known]
        if matched:
            print(f"     You have : {', '.join(matched)}")
        print(f"     Learn next: {', '.join(missing[:4])}")
    print("=" * 52)


def get_user_input():
    print("Enter at least 3 skills (comma-separated), e.g. Python, Cloud Computing, Automation")
    while True:
        raw = input("Your skills > ").split(",")
        if len([s for s in raw if s.strip()]) >= MIN_INPUTS:
            return raw
        print(f"Please enter at least {MIN_INPUTS} skills.")


def main():
    roles = load_roles()
    if len(sys.argv) > 1:
        user_skills = [a.replace("_", " ") for a in sys.argv[1:]]
        if len(user_skills) < MIN_INPUTS:
            sys.exit(f"Provide at least {MIN_INPUTS} skills.")
    else:
        print("Available vocabulary:", ", ".join(build_vocab(roles)), "\n")
        user_skills = get_user_input()
    display(*recommend(user_skills, roles), roles)


if __name__ == "__main__":
    main()
