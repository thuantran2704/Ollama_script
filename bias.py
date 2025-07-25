import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def load_and_prepare_data():
    """Load and prepare the LLM and Turker score datasets"""
    # Load datasets
    llm_scores = pd.read_csv('interview_scores_ignore_itver.csv')
    turker_scores = pd.read_csv('turker_scores_full_interview.csv')
    
    # Drop unnecessary columns
    llm_scores = llm_scores.drop(columns=['Transcript'], errors='ignore')
    turker_scores = turker_scores.drop(columns=['Worker'], errors='ignore')
    
    # Merge datasets
    merged_scores = pd.merge(llm_scores, turker_scores, 
                           on='Participant', 
                           suffixes=('_llm', '_turker'),
                           how='inner')
    return merged_scores

def calculate_comparison_metrics(merged_scores):
    """Calculate comparison metrics between LLM and Turker scores"""
    metrics = [col.replace('_llm', '').replace('_turker', '') 
               for col in merged_scores.columns[1:]]
    metrics = list(set(metrics))  # Get unique metrics
    
    results = []
    for metric in metrics:
        llm_col = f'{metric}_llm'
        turker_col = f'{metric}_turker'
        
        if llm_col in merged_scores.columns and turker_col in merged_scores.columns:
            # Calculate differences
            differences = merged_scores[turker_col] - merged_scores[llm_col]
            mean_diff = differences.mean()
            abs_mean_diff = differences.abs().mean()
            
            # Correlation
            corr, p_value_corr = stats.pearsonr(merged_scores[llm_col], merged_scores[turker_col])
            
            # Paired t-test
            t_stat, p_val_ttest = stats.ttest_rel(merged_scores[llm_col], merged_scores[turker_col])
            
            # Effect size
            n = len(differences)
            d = mean_diff / (differences.std() / np.sqrt(n))
            
            # Agreement metrics
            within_1 = (np.abs(differences) <= 1).mean() * 100
            
            results.append({
                'Metric': metric,
                'Mean_Difference': mean_diff,
                'Absolute_Mean_Difference': abs_mean_diff,
                'Cohen_d': d,
                'Correlation': corr,
                'P_Value_Correlation': p_value_corr,
                'T_Statistic': t_stat,
                'P_Value_TTest': p_val_ttest,
                'Agreement_Within_1_Point(%)': within_1,
                'LLM_Mean': merged_scores[llm_col].mean(),
                'Turker_Mean': merged_scores[turker_col].mean(),
                'LLM_Std': merged_scores[llm_col].std(),
                'Turker_Std': merged_scores[turker_col].std()
            })
    
    results_df = pd.DataFrame(results)
    results_df['Significant_Difference'] = results_df['P_Value_TTest'] < 0.05
    results_df['Significant_Correlation'] = results_df['P_Value_Correlation'] < 0.05
    
    return results_df

def visualize_results(results_df):
    """Create visualization plots for the comparison results"""
    plt.figure(figsize=(22, 16))
    plt.suptitle("LLM vs. Turker Interview Score Comparisons", fontsize=18, y=1.02)
    
    # Plot 1: Mean differences
    ax1 = plt.subplot(2, 2, 1)
    sns.barplot(data=results_df, x='Metric', y='Mean_Difference',
                hue='Significant_Difference', dodge=False,
                palette={True: 'coral', False: 'lightblue'})
    plt.xticks(rotation=90, fontsize=10)
    ax1.set_title('A. Mean Score Differences (Turker - LLM)\nRed bars indicate statistically significant differences', 
                 pad=20, fontsize=12)
    plt.axhline(0, color='black', linestyle='--')
    ax1.set_ylabel('Mean Difference', fontsize=11)
    ax1.get_legend().remove()
    
    # Plot 2: Correlation plot
    ax2 = plt.subplot(2, 2, 2)
    scatter = sns.scatterplot(data=results_df, x='Correlation', y='Metric',
                             hue='Significant_Correlation', size='Cohen_d',
                             sizes=(50, 250), palette={True: 'limegreen', False: 'gray'})
    ax2.set_title('B. Correlation Between LLM and Turker Ratings\nPoint size = Effect Size (Cohen\'s d)', 
                 pad=20, fontsize=12)
    plt.axvline(0, color='gray', linestyle='--')
    plt.legend(bbox_to_anchor=(1.35, 1), loc='upper right', title='Significant Correlation')
    ax2.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
    
    # Plot 3: Means comparison
    ax3 = plt.subplot(2, 2, 3)
    results_melted = results_df.melt(id_vars=['Metric'], 
                                   value_vars=['LLM_Mean', 'Turker_Mean'],
                                   var_name='Source', value_name='Mean_Score')
    sns.barplot(data=results_melted, x='Metric', y='Mean_Score', hue='Source',
                palette={'LLM_Mean': 'navy', 'Turker_Mean': 'darkorange'})
    plt.xticks(rotation=90, fontsize=10)
    ax3.set_title('C. Comparison of Mean Scores by Rating Source', pad=20, fontsize=12)
    ax3.set_ylabel('Mean Score', fontsize=11)
    plt.legend(title='Rating Source', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Plot 4: Agreement within 1 point
    ax4 = plt.subplot(2, 2, 4)
    sns.barplot(data=results_df, x='Metric', y='Agreement_Within_1_Point(%)',
                color='mediumseagreen')
    plt.xticks(rotation=90, fontsize=10)
    ax4.set_title('D. Percentage of Ratings Within 1 Point Difference', pad=20, fontsize=12)
    ax4.set_ylabel('Agreement (%)', fontsize=11)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=3.0)
    plt.savefig('llm_turker_comparison_results_ignore_interviewer.png', bbox_inches='tight', dpi=300)
    plt.show()

def main():
    """Main function to run the analysis"""
    print("Loading and preparing data...")
    merged_scores = load_and_prepare_data()
    
    print("Calculating comparison metrics...")
    results_df = calculate_comparison_metrics(merged_scores)
    
    # Display key results
    print("\nKey Comparison Results:")
    display_cols = ['Metric', 'Mean_Difference', 'Cohen_d', 'Correlation',
                   'P_Value_TTest', 'Significant_Difference',
                   'Agreement_Within_1_Point(%)']
    print(results_df[display_cols].sort_values('P_Value_TTest'))
    
    # Save full results
    results_df.to_csv('llm_turker_comparison_results_ignore_interviewer.csv', index=False)
    print("\nSaved full results to 'llm_turker_comparison_results_ignore_interviewer.csv'")
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_results(results_df)
    print("Saved visualization to 'llm_turker_comparison_results_ignore_interviewer.png'")

if __name__ == "__main__":
    main()