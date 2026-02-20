import os
import subprocess

# List of all pre-ranks to train on
#PRERANKS = ['marginal', 'mean', 'variance', 'dependency', 'pca', 'density', 'cdf']

PRERANKS = ['density', 'cdf']
FIXED_LAMBDA = 10

def run_training():
    print(f"Starting sequential training for {len(PRERANKS)} pre-ranks with lambda={FIXED_LAMBDA}...")

    for prerank in PRERANKS:
        print(f"\n" + "="*50)
        print(f"TRAINING PRERANK: {prerank}")
        print("="*50)

        env = os.environ.copy()
        env["SEARCH_PRERANK"] = prerank
        env["SEARCH_LAMBDA"] = str(FIXED_LAMBDA)

        try:
            subprocess.run(["python", "-u", "multiple-preranks.py"], env=env, check=True)
            print(f"Successfully finished training for {prerank}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while training {prerank}: {e}")
            # Continue to next prerank even if one fails
            continue

    print("\nAll pre-ranks have finished training.")

if __name__ == "__main__":
    run_training()
