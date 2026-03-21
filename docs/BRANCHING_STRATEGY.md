# PTM-RL Branching Strategy

**Version:** 1.0  
**Updated:** 2026-03-17

---

## Overview

PTM-RL uses a simplified Git Flow branching model for version management and parallel development.

---

## Branch Structure

```
main                        # Stable releases (v1.0, v1.1, ...)
  ↓
develop                     # Development integration
  ↓
feature/*                   # Feature branches (created as needed)
```

---

## Branch Roles

### `main`
**Purpose:** Production-ready releases only

**Rules:**
- ✅ Only accepts merges from `develop`
- ✅ Every merge = new version tag (v1.0.0, v1.1.0, ...)
- ✅ Always stable and deployable
- ❌ No direct commits

**Commits:**
- Release merges
- Hotfix merges (critical bugs)

**Tags:**
- `v1.0.0` - Initial release (2026-03-17)
- `v1.1.0` - Toroidal geometry (planned)
- `v1.2.0` - Resistive MHD (planned)

---

### `develop`
**Purpose:** Development integration branch

**Rules:**
- ✅ Accepts merges from `feature/*` branches
- ✅ Integration testing happens here
- ✅ Merges to `main` when release-ready
- ⚠️ May be unstable during active development

**Workflow:**
1. Feature branches merge here
2. Run full test suite
3. Fix integration issues
4. When stable → merge to `main` + tag

---

### `feature/*`
**Purpose:** Individual feature development

**Naming:**
- `feature/toroidal-geometry` - Toroidal coordinate transformation
- `feature/rl-adaptation` - RL framework adaptation for toroidal MHD
- `feature/<descriptive-name>` - General pattern

**Rules:**
- ✅ Created from `develop`
- ✅ Merged back to `develop` when complete
- ✅ Delete after merge (keep remote for history)
- ⚠️ Keep focused on single feature

**Lifecycle:**
```bash
# Create
git checkout develop
git checkout -b feature/my-feature

# Develop
git commit -m "..."
git push origin feature/my-feature

# Merge
git checkout develop
git merge feature/my-feature
git branch -d feature/my-feature  # local
git push origin --delete feature/my-feature  # remote (optional)
```

---

## v1.1 Development Workflow

### Initial Setup (Done ✅)

```bash
# ∞ created:
main → develop

# Feature branches will be created after YZ decides v1.1 scope
```

### When v1.1 Scope is Decided

**Create feature branches based on actual tasks:**
```bash
git checkout develop
git checkout -b feature/<descriptive-name>
git push -u origin feature/<descriptive-name>
```

### Parallel Development (Example)

**Developer 1:**
```bash
git checkout feature/<task-1>
# Implement feature
git commit -m "..."
git push origin feature/<task-1>
```

**Developer 2:**
```bash
git checkout feature/<task-2>
# Implement feature
git commit -m "..."
git push origin feature/<task-2>
```

### Integration

**When 小P completes:**
```bash
git checkout develop
git merge feature/toroidal-geometry
git push origin develop
# Run tests, fix any issues
```

**When 小A completes:**
```bash
git checkout develop
git merge feature/rl-adaptation
git push origin develop
# Integration test: toroidal MHD + RL
```

### Release v1.1

**When develop is stable:**
```bash
git checkout main
git merge develop
git tag v1.1.0 -m "Release v1.1.0: Toroidal Geometry"
git push origin main --tags
# Update CHANGELOG.md
```

---

## Quick Reference

### Check current branch
```bash
git branch        # Local branches
git branch -a     # All branches (local + remote)
```

### Switch branches
```bash
git checkout develop
git checkout feature/toroidal-geometry
git checkout main
```

### Sync with remote
```bash
git pull origin develop
git push origin feature/my-feature
```

### Merge feature to develop
```bash
git checkout develop
git merge feature/my-feature
git push origin develop
```

### Create new feature branch
```bash
git checkout develop
git checkout -b feature/new-feature
git push -u origin feature/new-feature
```

---

## Conflict Resolution

**If merge conflicts occur:**

```bash
git checkout develop
git merge feature/my-feature
# Conflicts! 

# Resolve in editor, then:
git add <resolved-files>
git commit -m "Merge feature/my-feature, resolved conflicts"
git push origin develop
```

**Prevention:**
- Merge `develop` into your feature branch regularly
- Communicate with team about overlapping changes
- Small, focused features reduce conflicts

---

## Protection Rules (GitHub)

**Recommended settings:**

**`main` branch:**
- ✅ Require pull request reviews (1 reviewer)
- ✅ Require status checks (GitHub Actions CI)
- ✅ Require branches to be up to date
- ✅ Do not allow force push

**`develop` branch:**
- ✅ Require status checks (optional)
- ⚠️ Allow force push (for rebasing, use carefully)

**`feature/*` branches:**
- ⚠️ No protection (developers manage)

---

## Best Practices

### Commits
- ✅ Clear, descriptive messages
- ✅ Atomic commits (one logical change)
- ✅ Test before committing

### Feature Branches
- ✅ Keep focused (single feature/fix)
- ✅ Merge frequently to `develop` (don't hoard)
- ✅ Delete after merge (reduce clutter)

### Code Review
- ✅ Review before merging to `develop`
- ✅ Physics changes: 小P reviews
- ✅ RL changes: 小A reviews
- ✅ Architecture: ∞ reviews

### Testing
- ✅ Run tests locally before pushing
- ✅ Wait for CI before merging to `develop`
- ✅ Full test suite before merging to `main`

---

## Troubleshooting

### "Your branch is behind"
```bash
git pull origin <branch-name>
# Or rebase (advanced):
git pull --rebase origin <branch-name>
```

### "Merge conflict"
```bash
# Option 1: Merge
git merge <branch>
# Resolve conflicts, then commit

# Option 2: Rebase (cleaner history)
git rebase <branch>
# Resolve conflicts, then:
git rebase --continue
```

### "Accidentally committed to wrong branch"
```bash
# If not pushed yet:
git reset --soft HEAD~1
git stash
git checkout correct-branch
git stash pop
git commit -m "..."

# If already pushed: contact ∞
```

---

## Version History

**v1.0 (2026-03-17):**
- Initial branching strategy document
- Created `develop`, `feature/toroidal-geometry`, `feature/rl-adaptation`
- Documented v1.1 workflow

---

**Maintained by:** ∞ (PM)  
**Questions:** Ask in #项目讨论
