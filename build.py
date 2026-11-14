"""
SIH26006 - FreightAI
Build script: Trains models during Render deployment
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def models_exist():
    base = os.path.dirname(os.path.abspath(__file__))
    return (
        os.path.exists(os.path.join(base, "models", "linear_regression.pkl")) and
        os.path.exists(os.path.join(base, "models", "xgboost_model.pkl"))
    )

if __name__ == "__main__":
    if not models_exist():
        print("[BUILD] Models not found. Training now...")
        from src.train import main as train_main
        train_main()
        print("[BUILD] Training complete!")
    else:
        print("[BUILD] Models already exist. Skipping training.")
