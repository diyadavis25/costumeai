"""One command to (re)build data + compatibility model + report plots:  python build_all.py"""
import subprocess
import sys

for mod, args in (("training.generate_outfit_pairs", []), ("training.train_compatibility", []),
                  ("training.evaluate_model", [])):
    print(f"\n=== {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod, *args])
print("\nDone. Launch the app with:  streamlit run app/app.py")
