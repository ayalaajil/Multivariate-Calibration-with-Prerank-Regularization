import os
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
PRERANKS = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']
# Add more lambdas here if you want a finer search
LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] 

BASELINE_PATH = "csv-files/metrics-from-model-without-reg.csv"
OUTPUT_DIR = "csv-files/tuning_results"
SUMMARY_OUTPUT = "best_lambdas_found.csv"

# ==========================================

def modify_script_for_env_vars():
    """
    Ensures multiple-preranks.py is set up to read from environment variables.
    This allows us to pass parameters without rewriting the file for every run.
    """
    script_path = "multiple-preranks.py"
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    # We replace the hardcoded assignments with os.getenv lookups if they aren't already there
    modified = False
    
    if "train_prerank =" in content and "os.getenv" not in content:
        content = content.replace(
            "train_prerank = 'density'", 
            "train_prerank = os.getenv('SEARCH_PRERANK', 'density')"
        ).replace(
            "train_prerank = 'marginal'",
            "train_prerank = os.getenv('SEARCH_PRERANK', 'marginal')"
        )
        modified = True

    if "'lambda':" in content and "os.getenv" not in content:
        # Looking for the lambda assignment in hparams
        import re
        content = re.sub(
            r"'lambda':\s*[\d\.]+,", 
            "'lambda': float(os.getenv('SEARCH_LAMBDA', 10)),", 
            content
        )
        modified = True

    if "results_path =" in content and "tuning_results" not in content:
        content = re.sub(
            r"results_path = f\"csv-files/.*\.csv\"",
            "results_path = f\"csv-files/tuning_results/metrics_{train_prerank}_lambda={os.getenv('SEARCH_LAMBDA', 10)}.csv\"",
            content
        )
        modified = True

    if modified:
        with open(script_path, "w") as f:
            f.write(content)
        print(f"Modified {script_path} to support environment variables.")
    
    return True

def run_experiment(prerank, lam):
    """Runs a single experiment for a specific prerank and lambda."""
    env = os.environ.copy()
    env["SEARCH_PRERANK"] = prerank
    env["SEARCH_LAMBDA"] = str(lam)
    # We use -u to get unbuffered output so we can see progress in logs
    result = subprocess.run(["python", "-u", "multiple-preranks.py"], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {prerank} with lambda {lam}:")
        print(result.stderr)
    return result.returncode == 0

def analyze_results():
    """
    Analyzes all generated CSVs and finds the best lambda for each prerank.
    The 'best' is the one with lowest target PCE that doesn't increase Energy Score by > 10%.
    """
    if not os.path.exists(BASELINE_PATH):
        print(f"Warning: Baseline file {BASELINE_PATH} not found. Cannot check 10% threshold.")
        return

    baseline_df = pd.read_csv(BASELINE_PATH)
    # Average baseline energy across seeds per dataset
    baseline_agg = baseline_df.groupby('data_name')['energy'].mean().reset_index()
    baseline_agg.rename(columns={'energy': 'energy_baseline'}, inplace=True)

    final_results = []

    for prerank in PRERANKS:
        prerank_results = []
        for lam in LAMBDAS:
            path = os.path.join(OUTPUT_DIR, f"metrics_{prerank}_lambda={float(lam)}.csv")
            if not os.path.exists(path):
                continue
            
            df = pd.read_csv(path)
            # We want the PCE specific to the training prerank
            target_col = f"pce_{prerank}"
            if target_col not in df.columns:
                continue

            # Average across seeds per dataset
            agg = df.groupby('data_name').agg({
                'energy': 'mean',
                target_col: 'mean'
            }).reset_index()

            # Merge with baseline
            merged = pd.merge(agg, baseline_agg, on='data_name')
            merged['es_increase_pct'] = (merged['energy'] - merged['energy_baseline']) / merged['energy_baseline'] * 100
            
            # Global average for this lambda
            avg_es_increase = merged['es_increase_pct'].mean()
            avg_target_pce = merged[target_col].mean()

            prerank_results.append({
                'lambda': lam,
                'avg_target_pce': avg_target_pce,
                'avg_es_increase_pct': avg_es_increase,
                'is_valid': avg_es_increase <= 10.0
            })

        if not prerank_results:
            print(f"No results found for {prerank}")
            continue

        # Filter for valid ones (ES increase <= 10%)
        valid_results = [r for r in prerank_results if r['is_valid']]
        
        if valid_results:
            # Sort by PCE (lowest first)
            best = min(valid_results, key=lambda x: x['avg_target_pce'])
            final_results.append({
                'prerank': prerank,
                'best_lambda': best['lambda'],
                'pce': best['avg_target_pce'],
                'es_increase_%': best['avg_es_increase_pct']
            })
        else:
            # If none are below 10%, maybe take the one with smallest increase?
            best_attempt = min(prerank_results, key=lambda x: x['avg_es_increase_pct'])
            print(f"Warning: No lambda for {prerank} satisfied the <10% ES increase. " 
                  f"Best attempt was lambda={best_attempt['lambda']} with {best_attempt['avg_es_increase_pct']:.2f}% increase.")

    if final_results:
        summary_df = pd.DataFrame(final_results)
        print("\n" + "="*50)
        print("SUMMARY: BEST LAMBDA PER PRERANK")
        print("Constraint: Energy Score increase <= 10%")
        print("="*50)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(SUMMARY_OUTPUT, index=False)
        print(f"\nSummary saved to {SUMMARY_OUTPUT}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if modify_script_for_env_vars():
        print(f"Starting grid search over {len(PRERANKS)} preranks and {len(LAMBDAS)} lambdas...")
        
        # Nested loop for grid search
        total_runs = len(PRERANKS) * len(LAMBDAS)
        pbar = tqdm(total=total_runs, desc="Global Progress")
        
        for prerank in PRERANKS:
            for lam in LAMBDAS:
                # print(f"Processing {prerank} with lambda {lam}...")
                run_experiment(prerank, lam)
                pbar.update(1)
        
        pbar.close()
        
        # Analyze results
        analyze_results()
    else:
        print("Failed to initialize. Check multiple-preranks.py.")
