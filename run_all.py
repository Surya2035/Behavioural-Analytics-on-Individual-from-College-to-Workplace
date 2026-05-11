import subprocess
import sys

print("\nStarting Behavioral Analytics Pipeline\n")

scripts = [

"analysis/preprocess.py",
"analysis/clustering.py",
"analysis/correlation.py",
"analysis/feature_importance.py",
"analysis/train_model.py",
"analysis/sem_paths.py",
"visualization/sem_diagram.py"

]

for script in scripts:

    print(f"\nRunning {script}\n")

    subprocess.run([sys.executable, script])


print("\nPipeline completed successfully\n")

print("\nLaunching Dashboard...\n")

subprocess.Popen([
sys.executable,
"-m",
"streamlit",
"run",
"dashboard/dashboard.py"
])