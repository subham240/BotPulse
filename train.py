"""
Main Model Training CLI for BotPulse.

Executes the combined model training pipeline:
1. Verifies dataset presence
2. Extracts M4D features from Apache logs if needed
3. Harmonizes features and creates leakage-free splits
4. Fits preprocessor
5. Trains Random Forest and XGBoost classifiers
6. Benchmarks performance and source-wise accuracy
7. Persists champion model and evaluation reports
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_training import run_full_training_pipeline


def main():
    try:
        report = run_full_training_pipeline(force_rebuild=False)
        print("\n>>> BotPulse training workflow completed successfully! <<<")
        print(f"Champion Model: {report['champion_model']}")
        print(f"Overall Accuracy: {report['models'][report['champion_model'].lower().replace(' ', '_')]['overall']['accuracy']:.4f}")
        print(f"Overall Macro-F1: {report['models'][report['champion_model'].lower().replace(' ', '_')]['overall']['macro_f1']:.4f}")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
