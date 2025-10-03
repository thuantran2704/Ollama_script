import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import cross_val_score, KFold

def evaluate_r2(X, y):
    """Compute mean R² using 5-fold CV."""
    model = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return np.mean(scores)

def analyze_file(file_path, target_col=None, alpha=None):
    """Run LASSO + ablation on a single CSV file."""
    df = pd.read_csv(file_path)

    # Try to guess target if not provided
    if target_col is None:
        target_col = df.columns[-1]  # last column
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode non-numeric if needed
    X = pd.get_dummies(X, drop_first=True)

    # --- Baseline R² ---
    baseline_r2 = evaluate_r2(X, y)

    # --- LASSO Feature Selection ---
    lasso = LassoCV(cv=5, random_state=42, alphas=None if alpha is None else [alpha])
    lasso.fit(X, y)
    selected_features = list(X.columns[lasso.coef_ != 0])

    # --- Ablation Study ---
    ablation_results = []
    for feature in X.columns:
        X_ablate = X.drop(columns=[feature])
        r2_ablate = evaluate_r2(X_ablate, y)
        ablation_results.append({
            "Feature_Removed": feature,
            "R2_After_Removal": r2_ablate,
            "R2_Change": r2_ablate - baseline_r2
        })

    ablation_df = pd.DataFrame(ablation_results)

    return {
        "File": os.path.basename(file_path),
        "Target": target_col,
        "Baseline_R2": baseline_r2,
        "Selected_Features(LASSO)": selected_features,
        "Ablation": ablation_df
    }

def run_analysis_on_directory(directory="./", target_col=None):
    """Run analysis on all CSV files in a directory."""
    all_results = []

    for file_path in glob.glob(os.path.join(directory, "*.csv")):
        print(f"Processing {file_path} ...")
        result = analyze_file(file_path, target_col=target_col)

        # Save ablation results per file
        out_ablation = f"{os.path.splitext(file_path)[0]}_ablation.csv"
        result["Ablation"].to_csv(out_ablation, index=False)

        all_results.append({
            "File": result["File"],
            "Target": result["Target"],
            "Baseline_R2": result["Baseline_R2"],
            "Selected_Features(LASSO)": "; ".join(result["Selected_Features(LASSO)"])
        })

    # Save overall summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("feature_selection_summary.csv", index=False)
    print("\nSaved overall summary to feature_selection_summary.csv")

if __name__ == "__main__":
    run_analysis_on_directory("./")
