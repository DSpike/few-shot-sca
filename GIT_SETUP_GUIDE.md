# Push to GitHub - Step by Step Guide

## Prerequisites

1. **GitHub account** - Create one at https://github.com if you don't have
2. **Git installed** - Check with: `git --version`
3. **GitHub token** (for authentication) - We'll create this

## Step 1: Create GitHub Repository

### Option A: Via GitHub Website (Easier)

1. Go to https://github.com/new
2. Repository name: `few-shot-sca` (or your preferred name)
3. Description: "Few-Shot Meta-Learning for Side-Channel Analysis of Wearable IoT Devices"
4. **Make it Private** (until paper is published!)
5. **Do NOT initialize** with README, .gitignore, or license (we have our own files)
6. Click "Create repository"
7. **Copy the repository URL** (looks like: `https://github.com/YOUR_USERNAME/few-shot-sca.git`)

### Option B: Via GitHub CLI (Alternative)

```bash
# Install GitHub CLI first: https://cli.github.com/
gh auth login
gh repo create few-shot-sca --private --source=. --remote=origin
```

## Step 2: Create .gitignore File

**Important**: Don't commit large data files, results, or sensitive info!

Run this to create `.gitignore`:

```bash
cd c:\Users\Dspike\Documents\sca
```

Then create the file with the content below (I'll create it for you in the next step).

## Step 3: Initialize Git Repository

```bash
cd c:\Users\Dspike\Documents\sca

# Initialize git repo
git init

# Configure your identity (use your real info!)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add remote
git remote add origin https://github.com/DSpike/few-shot-sca.git

# Check remote is set
git remote -v
```

## Step 4: Add and Commit Files

```bash
# Add all Python scripts and documentation
git add *.py *.md

# Check what will be committed
git status

# Commit
git commit -m "Initial commit: Few-shot meta-learning for SCA

- Implemented MAML, Prototypical Networks, Siamese Networks
- Stratified random sampling for 256-way classification
- Baseline standard CNN for comparison
- Reproducible experiments with fixed seeds (42-51)
- Documentation and guides for reproducibility"
```

## Step 5: Push to GitHub

### First Time Push (set upstream)

```bash
git push -u origin main
```

**If this fails with authentication error**, you need a Personal Access Token:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "SCA Research Project"
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. When git asks for password, paste the token

### Subsequent Pushes

```bash
git push
```

## Step 6: Add Results Later (After Experiments)

**After you run experiments**, add and push results:

```bash
# Add result files
git add experiment_results/*.csv
git add baseline_results/*.csv
git add figures/*.png
git add figures/*.pdf

# Commit results
git commit -m "Add experimental results

- 10 runs of few-shot experiments (seeds 42-51)
- 10 runs of baseline experiments (seeds 42-51)
- Aggregated statistics with Mean ± Std
- Publication-quality figures"

# Push
git push
```

## What to Commit (✅) and What to Ignore (❌)

### ✅ **DO Commit**:
- All `.py` files (your code)
- All `.md` files (documentation)
- Small result CSVs (< 10 MB)
- Figures (PNG, PDF)
- Configuration files
- Requirements.txt (Python dependencies)

### ❌ **DON'T Commit**:
- **ASCAD dataset** (`ASCAD.h5` - it's huge! ~1-2 GB)
- `.venv/` folder (Python virtual environment)
- `__pycache__/` folders (Python cache)
- `.ipynb_checkpoints/` (Jupyter cache)
- Temporary files (`.pyc`, `.tmp`)
- Very large result files (> 100 MB)

## Recommended .gitignore Contents

The `.gitignore` file I'll create includes:
```
# Data files (too large)
*.h5
*.hdf5
ASCAD_data/

# Python cache
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Temporary
*.tmp
*.log
```

## Sharing Your Repository

### Private Repository (Recommended until publication)

**Keep it private** until your paper is accepted/published.

To share with collaborators:
1. Go to your GitHub repo
2. Settings → Collaborators
3. Add their GitHub usernames

### Public Repository (After publication)

When paper is published, make it public:
1. Go to repo Settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Make public"

## Adding a README to GitHub

Your repository should have a nice README. Here's what to include:

```markdown
# Few-Shot Meta-Learning for Side-Channel Analysis

Official implementation of "Few-Shot Meta-Learning for Side-Channel Analysis of Wearable IoT Devices"

## Abstract
[Paste your paper abstract here]

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU
- ASCAD dataset

## Installation
\`\`\`bash
pip install torch numpy pandas h5py scipy matplotlib seaborn
\`\`\`

## Usage

### Run Few-Shot Experiments
\`\`\`bash
# Single run
python comprehensive_few_shot_study.py --seed 42

# Multiple runs (10 runs, seeds 42-51)
python run_multiple_experiments.py
\`\`\`

### Run Baseline
\`\`\`bash
# Multiple runs matching few-shot experiments
python run_baseline_multiple.py
\`\`\`

### Generate Figures
\`\`\`bash
python generate_plots.py
\`\`\`

## Reproducibility
All experiments use fixed seeds (42-51) for reproducibility.

## Citation
\`\`\`
[Your citation when paper is published]
\`\`\`

## License
[Add license - e.g., MIT, Apache 2.0]
```

## Useful Git Commands

```bash
# Check status
git status

# See what changed
git diff

# View commit history
git log --oneline

# Create a new branch (for experiments)
git checkout -b experiment-variations

# Switch back to main
git checkout main

# Pull latest changes
git pull

# Undo last commit (keep changes)
git reset --soft HEAD~1
```

## Best Practices

### Commit Messages

**Good** ✅:
```
git commit -m "Add early stopping to baseline training

- Patience = 20 epochs
- Saves best model state
- Reduces overfitting"
```

**Bad** ❌:
```
git commit -m "fix"
git commit -m "updates"
```

### Commit Often

- Commit after each significant change
- Don't wait to commit everything at once
- Use meaningful commit messages

### Branch Strategy (Optional)

```bash
# Main branch: stable, working code
main

# Experiment branches
git checkout -b try-different-learning-rates
git checkout -b add-new-baseline

# Merge when done
git checkout main
git merge try-different-learning-rates
```

## Troubleshooting

### "Large files" error

If git complains about large files:
```bash
# Remove from staging
git reset HEAD path/to/large/file

# Add to .gitignore
echo "path/to/large/file" >> .gitignore
```

### Authentication Failed

Use Personal Access Token instead of password:
1. Generate token at: https://github.com/settings/tokens
2. Use token as password when prompted

### Already exists on remote

```bash
# If you need to force push (CAREFUL!)
git push -f origin main
```

## Summary

**Quick start**:
```bash
# 1. Create .gitignore (I'll do this next)
# 2. Initialize repo
git init
git config user.name "Your Name"
git config user.email "your@email.com"

# 3. Add remote
git remote add origin https://github.com/DSpike/few-shot-sca.git

# 4. Add, commit, push
git add *.py *.md
git commit -m "Initial commit: Few-shot SCA implementation"
git push -u origin main
```

**Ready to push your code to GitHub!** 🚀

Let me know if you want me to help create the .gitignore file and README.md!
