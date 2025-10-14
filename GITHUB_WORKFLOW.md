# GitHub Workflow Guide

## Repository Setup

This repository is a fork of the upstream project. We maintain our own changes on the `neo` branch.

### Remotes Configuration

```bash
origin  https://github.com/Haoming02/sd-webui-forge-classic.git  (upstream - READ ONLY)
myfork  https://github.com/SillySilk/sd-webui-forge-classic.git  (your fork - PUSH HERE)
```

## ⚠️ IMPORTANT: Always Push to YOUR Fork

**NEVER push to `origin`** - You don't have write access to the upstream repository.

**ALWAYS push to `myfork`** - This is your personal fork where you have full control.

## Common Git Commands

### Check Current Status
```bash
git status
```

### Stage Changes
```bash
# Stage specific files
git add path/to/file1.py path/to/file2.js

# Stage all modified files
git add -u

# Stage everything (including new files)
git add .
```

### Create a Commit
```bash
git commit -m "Your commit message here"
```

### Push Changes (ALWAYS to myfork!)
```bash
# Push to YOUR fork on the neo branch
git push myfork neo

# If you get conflicts, force push (use with caution!)
git push myfork neo --force
```

### Pull Latest Changes from Upstream
```bash
# Fetch updates from the original repository
git fetch origin

# Merge upstream changes into your branch
git merge origin/main
```

## Workflow Steps

### 1. Make Changes
Edit your files as needed.

### 2. Check What Changed
```bash
git status
git diff
```

### 3. Stage Your Changes
```bash
git add extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
git add javascript/controlnet_defaults.js
# ... add other files as needed
```

### 4. Commit Your Changes
```bash
git commit -m "Brief description of changes

Detailed explanation of what was changed and why.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 5. Push to YOUR Fork
```bash
git push myfork neo
```

## Branch Information

- **Main Branch**: `neo` - This is where all your custom work lives
- **Upstream Main**: `origin/main` - The original repository's main branch

## Quick Reference

| Action | Command |
|--------|---------|
| Check status | `git status` |
| View changes | `git diff` |
| Stage files | `git add <file>` |
| Commit | `git commit -m "message"` |
| **Push to fork** | `git push myfork neo` |
| Pull from upstream | `git fetch origin && git merge origin/main` |
| View remotes | `git remote -v` |
| View commit history | `git log --oneline -10` |

## Emergency Commands

### Undo Last Commit (Keep Changes)
```bash
git reset --soft HEAD~1
```

### Discard All Local Changes
```bash
# ⚠️ WARNING: This will delete all uncommitted changes!
git restore .
```

### View What Would Be Pushed
```bash
git log origin/neo..neo --oneline
```

## Remember

✅ **DO**: Push to `myfork`
❌ **DON'T**: Push to `origin` (you can't anyway, and shouldn't try)

Your fork URL: https://github.com/SillySilk/sd-webui-forge-classic
Your neo branch: https://github.com/SillySilk/sd-webui-forge-classic/tree/neo

---

*Keep this file as a reference for git operations in this repository.*
