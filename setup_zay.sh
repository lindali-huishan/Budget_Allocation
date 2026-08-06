#!/usr/bin/env bash
#
# setup_zay.sh — bootstrap the Python environment for this project, then
# commit and push the current state of the repo.
#
# Usage:
#   ./setup_zay.sh ["commit message"]
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"
COMMIT_MSG="${1:-Set up Python virtual environment}"

# --- 1. Create the virtual environment (if it doesn't already exist) ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment $VENV_DIR already exists, skipping creation."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- 2. Install dependencies ---
echo "Installing packages..."
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib jupyter

# Freeze exact versions for reproducibility
pip freeze > requirements.txt

# --- 3. Make sure the venv and other junk never get committed ---
touch .gitignore
for entry in ".venv/" "__pycache__/" "*.pyc" ".ipynb_checkpoints/" ".DS_Store"; do
    grep -qxF "$entry" .gitignore || echo "$entry" >> .gitignore
done

# --- 4. Stage, commit, and push ---
git add -A

if git diff --cached --quiet; then
    echo "Nothing to commit — working tree already matches the last commit."
else
    git commit -m "$COMMIT_MSG"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Pushing branch '$CURRENT_BRANCH' to origin..."
git push -u origin "$CURRENT_BRANCH"

echo "Done. Activate the environment later with: source $VENV_DIR/bin/activate"
