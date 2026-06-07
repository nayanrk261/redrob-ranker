import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---- Job Description ----
JD_TEXT = """
Senior AI Engineer at Redrob AI.
5 to 9 years of experience required.
Must have skills: RAG systems, vector embeddings, semantic search, LLMs, Python.
Experience with vector databases like Pinecone, Weaviate, Qdrant, FAISS.
Product company experience preferred. Consulting or services background not preferred.
Location: Pune or Noida. Willing to relocate preferred.
Notice period under 30 days preferred.
Open to work candidates preferred.
"""

# ---- Honeypot detection ----
def is_honeypot(c):
    profile = c.get('profile', {})
    yoe = profile.get('years_of_experience', 0)

    for job in c.get('career_history', []):
        start = job.get('start_date', '')
        if start:
            start_year = int(start[:4])
            if yoe > 0 and (2025 - start_year) > yoe + 10:
                return True

    expert_zero = sum(
        1 for s in c.get('skills', [])
        if s.get('proficiency') == 'expert' and s.get('duration_months', 1) == 0
    )
    if expert_zero >= 5:
        return True

    return False

# ---- Wrong domain check ----
WRONG_DOMAINS = [
    'civil engineer', 'graphic designer', 'hr manager',
    'accountant', 'mechanical engineer', 'marketing manager',
    'sales manager', 'content writer', 'teacher', 'doctor',
    'project manager', 'business analyst', 'financial analyst',
    'operations manager', 'scrum master', 'product manager',
    'customer support', 'customer success', 'support engineer'
]

def is_wrong_domain(c):
    title = c.get('profile', {}).get('current_title', '').lower()
    return any(d in title for d in WRONG_DOMAINS)

# ---- Experience range check ----
def wrong_experience(c):
    yoe = c.get('profile', {}).get('years_of_experience', 0)
    if yoe < 4 or yoe > 11:
        return True
    return False

# ---- Career quality score ----
CONSULTING_KEYWORDS = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'hcl', 'tech mahindra']

def career_score(c):
    score = 0
    history = c.get('career_history', [])

    for job in history:
        company = job.get('company', '').lower()
        is_consulting = any(k in company for k in CONSULTING_KEYWORDS)
        if not is_consulting:
            score += 1
        else:
            score -= 0.5

    if len(history) == 0:
        return 0
    return max(0, min(1, (score / len(history) + 1) / 2))

# ---- Availability score ----
def availability_score(c):
    signals = c.get('redrob_signals', {})
    score = 0

    if signals.get('open_to_work_flag', False):
        score += 0.4

    notice = signals.get('notice_period_days', 90)
    if notice <= 30:
        score += 0.3
    elif notice <= 60:
        score += 0.1

    response_rate = signals.get('recruiter_response_rate', 0)
    score += response_rate * 0.2

    if signals.get('willing_to_relocate', False):
        score += 0.1

    return min(1.0, score)

# ---- Skills match score ----
MUST_HAVE_SKILLS = ['rag', 'embeddings', 'vector', 'llm', 'python', 'semantic search', 'faiss', 'pinecone', 'weaviate']

def skills_score(c):
    candidate_skills = [s.get('name', '').lower() for s in c.get('skills', [])]
    all_text = " ".join(candidate_skills)
    matches = sum(1 for skill in MUST_HAVE_SKILLS if skill in all_text)
    return min(1.0, matches / 5)

# ---- Reasoning generator ----
def generate_reasoning(c, final_score):
    profile = c.get('profile', {})
    signals = c.get('redrob_signals', {})

    parts = []
    parts.append(f"{profile.get('years_of_experience', 0)} yrs exp")
    parts.append(profile.get('current_title', ''))

    if signals.get('open_to_work_flag'):
        parts.append("actively looking")

    notice = signals.get('notice_period_days', 90)
    parts.append(f"{notice}d notice")

    top_skills = [s.get('name') for s in c.get('skills', [])
                  if s.get('proficiency') in ['expert', 'advanced']][:3]
    if top_skills:
        parts.append("skills: " + ", ".join(top_skills))

    return " | ".join([p for p in parts if p])

# ========== MAIN ==========

print("Loading model and embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.load('embeddings.npy')
with open('candidate_ids.json') as f:
    candidate_ids = json.load(f)

print("Loading candidates...")
candidates = []
with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

cand_map = {c['candidate_id']: c for c in candidates}

print("Embedding job description...")
jd_embedding = model.encode([JD_TEXT], convert_to_numpy=True)[0]

print("Calculating scores...")
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normed = embeddings / (norms + 1e-9)
jd_norm = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-9)
semantic_scores = normed @ jd_norm

results = []
for i, cid in enumerate(candidate_ids):
    c = cand_map.get(cid)
    if c is None:
        continue
    if is_honeypot(c):
        continue
    if is_wrong_domain(c):
        continue
    if wrong_experience(c):
        continue

    sem = float(semantic_scores[i])
    car = career_score(c)
    avl = availability_score(c)
    skl = skills_score(c)

    final = (sem * 0.45) + (car * 0.20) + (avl * 0.10) + (skl * 0.25)

    results.append({
        'candidate_id': cid,
        'score': round(final, 4),
        'reasoning': generate_reasoning(c, final)
    })

print(f"Scored {len(results)} candidates (after filtering)")

results.sort(key=lambda x: x['score'], reverse=True)
top100 = results[:100]

for i, r in enumerate(top100):
    r['rank'] = i + 1

df = pd.DataFrame(top100)[['candidate_id', 'rank', 'score', 'reasoning']]
df.to_csv('submission.csv', index=False)

print("\nDone! submission.csv ban gayi.")
print("\nTop 5 candidates:")
for r in top100[:5]:
    print(f"  Rank {r['rank']}: {r['candidate_id']} | Score: {r['score']} | {r['reasoning']}")