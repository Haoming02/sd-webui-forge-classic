# Custom Modifications to Forge Neo

This file tracks custom modifications made to this installation to prevent them from being overwritten during updates.

## ⚠️ IMPORTANT: Modified Files

When updating Forge Neo, the following files contain custom modifications and should NOT be overwritten:

---

## 1. ControlNet Batch Processing Fix

**Date Modified:** 2025-09-29
**Status:** ✅ Submitted as PR to upstream (https://github.com/SillySilk/sd-webui-forge-classic)
**PR Status:** Pending review

### Modified File:
```
extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
```

### Backup Location:
```
extensions-builtin/sd_forge_controlnet/scripts/controlnet.py.backup
```

### What Was Fixed:
- Fixed batch folder processing (previously only processed first image)
- Fixed batch upload processing (previously only processed first image)
- Added memory-optimized sub-batching for large batches
- Added error handling for model loading failures

### Memory Configuration:
**Line 375** in `controlnet.py`:
```python
optimal_batch_size = 2  # Configured for 8GB VRAM
```

**Adjust for your VRAM:**
- 6-8 GB VRAM: `1-2` (current setting)
- 10-12 GB VRAM: `2-4`
- 16+ GB VRAM: `4-6`

### Testing Status:
- ✅ Tested with 10 images (Batch Upload & Folder)
- ✅ Tested with 25 images (Batch Folder)
- ✅ Works with various preprocessors
- ✅ Memory optimization functioning correctly

### Documentation:
See `CONTROLNET_BATCH_FIX.md` for complete details

---

## Update Procedure

### Before Updating Forge:

1. **Check PR Status:**
   - Visit: https://github.com/SillySilk/sd-webui-forge-classic/pulls
   - If PR merged: Update will include the fix ✅
   - If PR not merged: Need to preserve custom files ⚠️

2. **Backup Modified Files:**
   ```bash
   cp extensions-builtin/sd_forge_controlnet/scripts/controlnet.py extensions-builtin/sd_forge_controlnet/scripts/controlnet.py.custom
   ```

3. **Perform Update:**
   ```bash
   git fetch origin
   git status  # Check what will be overwritten
   ```

4. **If controlnet.py will be overwritten:**
   ```bash
   # Option A: Stash your changes before update
   git stash push -m "ControlNet batch fix" extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
   git pull origin neo
   git stash pop  # Re-apply your changes

   # Option B: Keep your version
   git pull origin neo
   git checkout HEAD extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
   cp extensions-builtin/sd_forge_controlnet/scripts/controlnet.py.custom extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
   ```

### After Updating:

1. **Verify Fix Still Present:**
   - Test batch folder with 10+ images
   - Check line 375 for `optimal_batch_size` setting
   - Verify log shows: "Batch mode: Processing X images as Y iteration(s) of Z (memory optimized)"

2. **If Fix Was Overwritten:**
   ```bash
   # Restore from backup
   cp extensions-builtin/sd_forge_controlnet/scripts/controlnet.py.backup extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
   ```

3. **If PR Was Merged:**
   - Fix is now part of official build! ✅
   - May need to re-adjust `optimal_batch_size` for your VRAM
   - Can delete backup files

---

## 2. CivitAI Browser+ Extension - REMOVED

**Date Removed:** 2025-10-04
**Status:** ❌ Removed due to incompatibility issues
**Reason:** Both original and Forge fork versions caused startup errors

### What Happened:
- Attempted to use original version: `BlafKing/sd-civitai-browser-plus` → Failed
- Attempted to use Forge fork: `thebob0042/sd-civitai-browser-plus-forge` → Still failed
- Error persisted in `civitai_gui.py` line 1029 despite fixes

### Error:
```python
Traceback (most recent call last):
  File "civitai_gui.py", line 1029, in <lambda>
    return lambda: {"choices": sub
```

**Root Cause:** Extension incompatible with Forge Neo's module structure and settings system

**Solution:** Extension removed entirely. Forge works without it.

### Alternative:
If CivitAI browsing functionality is needed, consider:
- Using CivitAI website directly in browser
- Manual model downloads
- Waiting for official Forge-compatible version

### Note:
Do NOT reinstall this extension unless a verified Forge Neo-compatible version becomes available.

---

## 3. sd-forge-couple Extension - Rolled Back

**Date Modified:** 2025-10-04
**Status:** ⚠️ Pinned to specific commit (not tracking latest)
**Repository:** https://github.com/Haoming02/sd-forge-couple

### Extension Location:
```
extensions/sd-forge-couple
```

### What Was Changed:
- **Rolled back from latest commit** to previous working version
- Latest commit (`4710ba6 fix #112`) broke functionality
- Pinned to commit `97dc96c` (October 2, 2025)

### Current Commit:
```
97dc96c - Merge pull request #111
```

### Why:
- Extension worked correctly before October 3 update
- Commit `4710ba6` introduced breaking changes
- Rolled back to last known working version

### Update Procedure:

**This extension will NOT auto-update:**
```bash
cd extensions/sd-forge-couple
git status  # Shows "HEAD detached" or on specific commit
```

**To update in the future:**
1. Check GitHub for new releases/commits
2. Test new version in separate directory first
3. Only update if confirmed working:
   ```bash
   cd extensions/sd-forge-couple
   git fetch origin
   git log --oneline origin/main -10  # Check what's new
   git checkout <new-commit-hash>  # Only if verified working
   ```

**To restore this working version:**
```bash
cd extensions/sd-forge-couple
git checkout main
git reset --hard 97dc96c
```

### Important Notes:
- **Do NOT run `git pull`** - it will break functionality again
- Extension is locked to working version
- Monitor repository for fixes to issue #112
- Test any future updates before committing

---

## 4. Custom Parameter Defaults

**Date Modified:** 2025-10-10
**Status:** ✅ Active (automatically preserved via git rebase)

### Modified Files:
- `modules/api/models.py`
- `modules/processing.py`
- `modules/processing_scripts/sampler.py`
- `modules/ui.py`

### What Was Changed:

**Denoising Strength:** 0.75 → 0.60
- Applied to img2img processing
- Applied to txt2img hires fix
- More conservative defaults for better quality

**Sampling Steps:** 20 → 30
- Better quality output with more steps
- Still reasonable generation time

### Why:
- Default 0.75 denoising was too aggressive for most use cases
- 20 steps often insufficient for high quality output
- New defaults provide better out-of-box experience

### Update Procedure:
These changes are tracked in git commits, so they will be automatically preserved during updates via `git rebase`.

**Verify after update:**
```bash
grep "denoising_strength: float = 0.60" modules/processing.py
grep "value=30" modules/processing_scripts/sampler.py
```

---

## 5. UI Prompt Styles Rename Fix

**Date Modified:** 2025-10-10
**Status:** ✅ Active
**File:** `modules/ui_prompt_styles.py`

### What Was Fixed:
- Added ability to rename prompt styles in UI
- Properly handles style name changes
- Prevents duplicate entries when renaming

### Technical Details:
- Added `original_name` hidden field to track rename operations
- Modified `save_style()` to detect and handle renames
- Removes old style entry when name changes

### Update Procedure:
Tracked in git, automatically preserved during updates.

---

## Other Custom Modifications

### (None currently)

Add any additional custom modifications here with:
- Date modified
- File path
- Backup location
- What was changed
- Why it was changed
- How to restore after update

---

## Quick Reference

**Check if ControlNet fix is active:**
```bash
grep -n "optimal_batch_size = 2" extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
```
Should return: `375:            optimal_batch_size = 2`

**Restore ControlNet fix from backup:**
```bash
cp extensions-builtin/sd_forge_controlnet/scripts/controlnet.py.backup extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
```

**Compare current vs backup:**
```bash
diff extensions-builtin/sd_forge_controlnet/scripts/controlnet.py.backup extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
```

---

## Related Files

- `CONTROLNET_BATCH_FIX.md` - Complete documentation of the fix
- `controlnet_batch_fix.patch` - Git diff patch file
- `COMMIT_MESSAGE.txt` - PR commit message
- `PR_SUBMISSION_GUIDE.md` - Instructions for PR submission

---

**Last Updated:** 2025-10-10
**Forge Neo Branch:** neo (updated to commit 510f167a)
**ControlNet Fix Commit:** 5ffcc83d (rebased)
**Total Custom Modifications:** 4 (active)
**Removed Extensions:** 1
**Pinned Extensions:** 1

## Update History

### 2025-10-10 - Updated to Latest Neo Branch
- **Action:** Rebased onto origin/neo (510f167a)
- **Commits Behind:** 30 commits updated
- **Strategy:** Git rebase to replay custom commits on top of latest code
- **Backup Branch:** neo-backup-20251010
- **Result:** ✅ All custom fixes preserved successfully
- **Verified:**
  - ControlNet batch fix intact (line 375)
  - Custom parameter defaults intact
  - UI prompt styles fix intact