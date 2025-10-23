import os
import pandas as pd
from ollama import Client
from tqdm import tqdm
import time

client = Client()

criteria = [
    "Overall", "RecommendHiring", "Colleague", "Engaged", "Excited",
    "EyeContact", "Smiled", "SpeakingRate", "NoFillers", "Friendly",
    "Paused", "EngagingTone", "StructuredAnswers", "Calm", "NotStressed",
    "Focused", "Authentic", "NotAwkward"
]

features = ["transcript", "prosodic", "facial", "smile_pre", "smile_post"]

# Folder for saving results
RESULTS_DIR = "ablation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------
# Load data
# ------------------------
def load_data():
    transcripts = pd.read_csv("interview_transcripts_by_turkers.csv")
    prosodic = pd.read_csv("prosodic_features.csv")
    return transcripts, prosodic

# ------------------------
# Helper: read file contents
# ------------------------
def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

# ------------------------
# Prepare features for one candidate
# ------------------------
def get_candidate_features(idx, transcripts, prosodic):
    transcript = transcripts.iloc[idx]["Transcript"] if "Transcript" in transcripts.columns else ""
    prosodic_rows = prosodic.iloc[idx*5:(idx+1)*5]
    prosodic_summary = prosodic_rows.to_string(index=False)

    facial_path = f"Facial_Features/candidate{idx+1}.csv"
    facial = read_file(facial_path)

    smile_pre = read_file(f"SmileData/pre/candidate{idx+1}.txt")
    smile_post = read_file(f"SmileData/post/candidate{idx+1}.txt")

    return {
        "transcript": transcript,
        "prosodic": prosodic_summary,
        "facial": facial,
        "smile_pre": smile_pre,
        "smile_post": smile_post
    }

# ------------------------
# Build prompt for model
# ------------------------
def build_prompt(criteria_name, included_features, candidate_data):
    included_text = "\n\n".join(
        f"{feat.capitalize()}:\n{candidate_data[feat]}" for feat in included_features
    )
    return f"""You are an expert interviewer evaluator.

Rate this candidate for the criterion: **{criteria_name}** on a scale from 1 to 7.
Justify briefly in ONE LINE why you gave that score.

Input data (features):
{included_text}

Output format:
<score (1-7)>, <one-line justification>
Example: 6, Spoke clearly and confidently with good engagement.
"""

# ------------------------
# Query the model
# ------------------------
def query_phi4(prompt):
    try:
        response = client.generate(model="phi4", prompt=prompt, options={"temperature": 0.2})
        text = response.response.strip() if hasattr(response, "response") else str(response).strip()
        score_part = ''.join(c for c in text if c.isdigit())
        score = int(score_part[0]) if score_part else 1
        justification = text.split(",", 1)[-1].strip() if "," in text else "No justification"
        return score, justification
    except Exception as e:
        print(f"Error during model query: {e}")
        return 1, "Error"

# ------------------------
# Evaluate candidates
# ------------------------
def evaluate_candidates(start_idx, end_idx):
    transcripts, prosodic = load_data()

    # Define evaluation phases
    phases = ["all_features"] + [f"ablation_{f}" for f in features]

    for phase in phases:
        results = []

        if phase == "all_features":
            included = features
            output_path = os.path.join(RESULTS_DIR, "All_features.csv")
        else:
            ablate_feat = phase.replace("ablation_", "")
            included = [f for f in features if f != ablate_feat]
            output_path = os.path.join(RESULTS_DIR, f"{phase}.csv")

        print(f"\n🔹 Starting phase: {phase} -> Saving to {output_path}")

        for idx in tqdm(range(start_idx - 1, end_idx)):
            candidate_data = get_candidate_features(idx, transcripts, prosodic)
            candidate_id = idx + 1

            for crit in criteria:
                prompt = build_prompt(crit, included, candidate_data)
                score, justification = query_phi4(prompt)

                results.append({
                    "Candidate": candidate_id,
                    "Criterion": crit,
                    "Score": score,
                    "Justification": justification
                })
                time.sleep(0.5)  # avoid rate limits

        # Save results for this phase
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"✅ Saved results to {output_path}")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print("Enter candidate range (e.g., 1 50):")
    start, end = map(int, input().split())
    evaluate_candidates(start, end)
