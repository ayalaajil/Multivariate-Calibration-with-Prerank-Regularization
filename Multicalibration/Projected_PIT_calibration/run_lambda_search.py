import os
import subprocess
import pandas as pd
import numpy as np

PRERANKS = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']
LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
BASELINE_PATH = "csv-files/metrics-from-model-without-reg.csv"
OUTPUT_DIR = "csv-files/tuning_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_experiment(prerank, lam):
    """
    Runs the multiple-preranks.py script with a specific prerank and lambda.
    We'll pass them as environment variables to avoid modifying the original file too much,
    or we can modify the file to accept sys.argv.
    """
    print(f"\n>>> Running: Prerank={prerank}, Lambda={lam}")
    # We modify the environment so the script can pick it up
    env = os.environ.copy()
    env["SEARCH_PRERANK"] = prerank
    env["SEARCH_LAMBDA"] = str(lam)

    # We use a temporary script or just run multiple-preranks.py if we modify it to check env vars
    subprocess.run(["python", "multiple-preranks.py"], env=env)

def modify_script_for_env_vars():
    """
    Modifies multiple-preranks.py to use environment variables if present.
    """
    with open("multiple-preranks.py", "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "train_prerank = 'density'" in line:
            new_lines.append("train_prerank = os.getenv('SEARCH_PRERANK', 'density')\n")
        elif "'lambda': 10," in line:
            new_lines.append("            'lambda': float(os.getenv('SEARCH_LAMBDA', 10)),\n")
        elif "results_path = f\"csv-files/TEST-metrics-from-model-trained-on-{name}-lambda=10.csv\"" in line:
            new_lines.append("results_path = f\"csv-files/tuning_results/metrics_{train_prerank}_lambda={os.getenv('SEARCH_LAMBDA', 10)}.csv\"\n")
        else:
            new_lines.append(line)

    with open("multiple-preranks.py", "w") as f:
        f.writelines(new_lines)

def analyze_best_lambda():
    """
    Reads the results and finds the best lambda for each prerank.
    """
    if not os.path.exists(BASELINE_PATH):
        print("Baseline results not found! Run lambda=0 first.")
        return

    baseline_df = pd.read_csv(BASELINE_PATH)
    # Average across seeds for baseline
    baseline_agg = baseline_df.groupby(['data_name']).agg({'energy': 'mean'}).reset_index()
    baseline_agg.rename(columns={'energy': 'energy_baseline'}, inplace=True)

    summary_rows = []

    for prerank in PRERANKS:
        for lam in LAMBDAS:
            path = f"{OUTPUT_DIR}/metrics_{prerank}_lambda={lam}.csv"
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            # Focus on the PCE metric corresponding to the training prerank
            target_pce_col = f"pce_{prerank}"

            # Aggregate across seeds
            agg = df.groupby(['data_name']).agg({
                'energy': 'mean',
                target_pce_col: 'mean'
            }).reset_index()

            # Merge with baseline to compare energy
            merged = pd.merge(agg, baseline_agg, on='data_name')

            # Calculate ES degradation
            merged['es_ratio'] = merged['energy'] / merged['energy_baseline']

            # Filter rows where ES degradation is <= 10%
            valid = merged[merged['es_ratio'] <= 1.1]

            if not valid.empty:
                # Score is the target PCE (lower is better)
                avg_pce = valid[target_pce_col].mean()
                avg_es_ratio = valid['es_ratio'].mean()
                summary_rows.append({
                    'prerank': prerank,
                    'lambda': lam,
                    'avg_target_pce': avg_pce,
                    'avg_es_ratio': avg_es_ratio
                })

    if not summary_rows:
        print("No valid results found to analyze.")
        return

    summary_df = pd.DataFrame(summary_rows)
    # For each prerank, find the lambda that minimizes avg_target_pce
    best_lambdas = summary_df.loc[summary_df.groupby('prerank')['avg_target_pce'].idxmin()]

    print("\n" + "="*30)
    print("BEST LAMBDA PER PRERANK")
    print("Criteria: Maximize calibration (min PCE) while ES degradation <= 10%")
    print("="*30)
    print(best_lambdas.to_string(index=False))

    best_lambdas.to_csv("best_lambdas_found.csv", index=False)

if __name__ == "__main__":
    # 1. Prepare script
    modify_script_for_env_vars()

    # 2. Run grid search
    for prerank in PRERANKS:
        for lam in LAMBDAS:
            run_experiment(prerank, lam)

    # 3. Analyze
    analyze_best_lambda()
