import os
import pandas as pd
from sklearn.metrics import mean_squared_error

# ------------------------
# Configuration
# ------------------------
RESULTS_DIR = "ablation_results"
CONTROL_FILE = os.path.join(RESULTS_DIR, "control_var.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "squared_error_summary.csv")

# ------------------------
# Compute Squared Errors
# ------------------------
def compute_squared_errors():
    if not os.path.exists(CONTROL_FILE):
        print(f"❌ Control file not found: {CONTROL_FILE}")
        return

    control_df = pd.read_csv(CONTROL_FILE)
    if "Candidate" not in control_df.columns or "Criterion" not in control_df.columns or "Score" not in control_df.columns:
        print("❌ control_var.csv missing required columns: Candidate, Criterion, Score")
        return

    results = []
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv") and f != "control_var.csv" and f != "squared_error_summary.csv"]

    for f in files:
        file_path = os.path.join(RESULTS_DIR, f)
        try:
            df = pd.read_csv(file_path)
            merged = pd.merge(control_df, df, on=["Candidate", "Criterion"], suffixes=("_control", "_test"))

            mse = mean_squared_error(merged["Score_control"], merged["Score_test"])
            results.append({"File": f, "Mean_Squared_Error": mse, "N": len(merged)})

            print(f"✅ Compared {f}: MSE = {mse:.4f} ({len(merged)} matching rows)")
        except Exception as e:
            print(f"⚠️ Error comparing {f}: {e}")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n📊 Results saved to {OUTPUT_FILE}")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    compute_squared_errors()
