import pandas as pd
from ollama import Client
import os
import time
from tqdm import tqdm

client = Client()

criteria = [
    "Overall", "RecommendHiring", "Colleague", "Engaged", "Excited",
    "EyeContact", "Smiled", "SpeakingRate", "NoFillers", "Friendly",
    "Paused", "EngagingTone", "StructuredAnswers", "Calm", "NotStressed",
    "Focused", "Authentic", "NotAwkward"
]

def evaluate_single(transcript, criterion):
    prompt = f"""You are a job interview judge. 
    Rate ONLY the following criterion on a 1-7 scale: {criterion}.
    PLEASE IGNORE THE INTERVIEWER WORDS. FOCUS ON INTERVIEWEE.
    Return EXACTLY ONE number (1-7). No explanations. No extra text.

    Transcript:
    {transcript}

    Score:"""

    try:
        response = client.generate(model='phi4', prompt=prompt, options={'temperature': 0.1})
        if hasattr(response, 'response'):
            first_line = response.response.strip().split('\n')[0]
            clean = ''.join(c for c in first_line if c.isdigit())
            if clean.isdigit():
                return int(clean)
        print(f"Invalid response for {criterion}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

def main():
    input_csv = 'interview_transcripts_by_turkers.csv'
    output_csv = 'interview_scores_by_metric.csv'

    df = pd.read_csv(input_csv, header=None, names=['ID', 'Transcript'])

    if os.path.exists(output_csv):
        results = pd.read_csv(output_csv)
    else:
        results = df.copy()
        for c in criteria:
            results[c] = None

    print(f"Total interviews: {len(df)}")

    for criterion in criteria:
        pending = results[results[criterion].isna()]
        if pending.empty:
            print(f"{criterion} already done")
            continue

        print(f"Evaluating {criterion} ({len(pending)} left)")
        for idx, row in tqdm(pending.iterrows(), total=len(pending), desc=criterion):
            score = evaluate_single(row['Transcript'], criterion)
            results.at[idx, criterion] = score
            results.to_csv(output_csv, index=False)
            time.sleep(0.5)

    print(f"Finished. Results saved in {output_csv}")

if __name__ == "__main__":
    main()
