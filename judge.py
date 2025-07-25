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

def evaluate_transcript(transcript):
    prompt = f"""You are a job interview judge. Rate this interview transcript on 1-7 scales for:
    {", ".join(criteria)}
    PLEASE IGNORE THE INTERVIEWER WORDS. FOCUS ON INTERVIEWEE.
    Return EXACTLY 18 comma-separated numbers (1-7) and NOTHING ELSE. No explanations. No additional text.
    Example: 5,4,6,3,5,4,5,3,6,5,4,5,6,4,5,5,4,5
    
    Transcript:
    {transcript}
    
    Scores:"""
    
    try:
        response = client.generate(model='phi4', prompt=prompt, options={'temperature': 0.1})
        if hasattr(response, 'response'):
            # Extract first line only and remove any non-numeric characters except commas
            first_line = response.response.split('\n')[0]
            clean_response = ''.join(c for c in first_line if c.isdigit() or c == ',')
            scores = [int(x) for x in clean_response.split(',')[:18] if x.isdigit()]
            if len(scores) == 18:
                return scores
        print("Invalid response format - Using fallback scores")
        return [1]*18
    except Exception as e:
        print(f"Error: {e} - Using fallback scores")
        return [1]*18

def main():
    input_csv = 'interview_transcripts_by_turkers.csv'
    output_csv = 'interview_scores_ignore_itver.csv'
    
    # Load all transcripts
    df = pd.read_csv(input_csv, header=None, names=['ID', 'Transcript'])
    all_data = df[['ID', 'Transcript']].values.tolist()
    
    # Load existing results if available
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        completed_ids = set(existing['ID'].tolist())
        results = existing.values.tolist()
    else:
        completed_ids = set()
        results = []
    
    print(f"\nTotal interviews: {len(all_data)}")
    print(f"Already evaluated: {len(completed_ids)}")
    
    while True:
        user_input = input("\nEvaluate which IDs? (e.g. '1-50' or '51-100') or 'q' to quit: ").strip()
        if user_input.lower() == 'q':
            break
        
        try:
            start, end = map(int, user_input.split('-'))
            batch = all_data[start-1:end]  # Convert to 0-based index
            
            for id, transcript in tqdm(batch, desc=f"Processing {start}-{end}"):
                if id in completed_ids:
                    continue
                
                scores = evaluate_transcript(transcript)
                results.append([id, transcript] + scores)
                
                # Save after each interview
                pd.DataFrame(results, columns=['ID', 'Transcript'] + criteria).to_csv(output_csv, index=False)
                time.sleep(0.5)  # Rate limit
            
            print(f"✅ Saved {len(batch)} evaluations to {output_csv}")
        except Exception as e:
            print(f"🚨 Error: {e} - Use format like '1-50'")

if __name__ == "__main__":
    main()