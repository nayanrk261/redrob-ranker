import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Step 1 — model load karo
print("Model load ho raha hai...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model ready!")

# Step 2 — candidates.jsonl padho
print("Candidates padh raha hai...")
candidates = []
with open('candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

print(f"Total candidates: {len(candidates)}")

# Step 3 — har candidate ka text banao
def candidate_to_text(c):
    parts = []
    
    # basic profile
    profile = c.get('profile', {})
    parts.append(profile.get('headline', ''))
    parts.append(profile.get('summary', ''))
    parts.append(f"Current title: {profile.get('current_title', '')}")
    parts.append(f"Industry: {profile.get('current_industry', '')}")
    parts.append(f"Experience: {profile.get('years_of_experience', 0)} years")
    
    # career history
    for job in c.get('career_history', []):
        parts.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
    
    # skills
    skill_names = [s.get('name', '') for s in c.get('skills', [])]
    parts.append("Skills: " + ", ".join(skill_names))
    
    return " | ".join([p for p in parts if p])

# Step 4 — sab candidates ka text banao
print("Text bana raha hai...")
texts = []
candidate_ids = []

for c in tqdm(candidates):
    texts.append(candidate_to_text(c))
    candidate_ids.append(c['candidate_id'])

# Step 5 — embeddings banao
print("Embeddings bana raha hai... (yeh 10-15 min lagega, chal jaane do)")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Step 6 — save karo
print("Save kar raha hai...")
np.save('embeddings.npy', embeddings)

with open('candidate_ids.json', 'w') as f:
    json.dump(candidate_ids, f)

print("Done! embeddings.npy aur candidate_ids.json ban gaye.")
print(f"Embeddings shape: {embeddings.shape}")