"""
SIH26006 – FreightAI
Entry point: Trains models (if needed) and starts Flask dashboard
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def models_exist() -> bool:
    return (
        os.path.exists(os.path.join(BASE_DIR, "models", "linear_regression.pkl")) and
        os.path.exists(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
    )


if __name__ == "__main__":
    # Auto-train if models don't exist
    if not models_exist():
        print("🚂 Models not found. Running training pipeline first...")
        from src.train import main as train_main
        train_main()
        print("\n" + "=" * 60)

    print("\n🌐 Starting FreightAI Web Dashboard...")
    print("   Open → http://localhost:5000")
    print("   Press Ctrl+C to stop\n")

    from app.app import app
    app.run(debug=False, host="0.0.0.0", port=5000)
