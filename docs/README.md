# PTM-RL Documentation

This directory contains detailed technical documentation for the PTM-RL project.

---

## Directory Structure

```
docs/
├── phase-reports/      # Detailed completion reports for each development phase
├── architecture/       # Project architecture and planning documents
├── layer1-pytokeq/     # PyTokEq equilibrium solver documentation
└── README.md          # This file
```

---

## Phase Reports (`phase-reports/`)

Chronological development documentation:

### Phase 1: PyTokEq Integration
- `PHASE1_COMPLETION_REPORT.md` - Equilibrium solver integration

### Phase 2: Equilibrium Cache
- `PHASE2_COMPLETION_REPORT.md` - Fast equilibrium loading system
- `PHASE2_HANDOFF.md` - Handoff to Phase 3

### Phase 3: MHD Diagnostics
- `PHASE3_COMPLETION_REPORT.md` - Island width, growth rate diagnostics

### Phase 4: RMP Control
- `PHASE4_COMPLETION_REPORT.md` - RMP field generation and coupling
- `PHASE4_HANDOFF.md` - Handoff to Phase 5

### Phase 5: RL Training Framework
- `PHASE5_STEP1_COMPLETION_REPORT.md` - RL environment initial implementation
- `PHASE5_STEP1_FINAL_REPORT.md` - Environment production-ready validation
- `PHASE5_STEP2_NUMERICAL_STABILITY_FIX.md` - Numerical stability improvements
- `PHASE5_STEP2.5_GYMNASIUM_MIGRATION.md` - Gym to Gymnasium migration
- `PHASE5_STEP3_PYTOKEQ_INTEGRATION.md` - Solovev equilibrium integration

### Cross-Phase Documentation
- `HANDOFF_TO_XIAOA.md` - Project handoff protocols
- `SYSTEMATIC_DIAGNOSIS.md` - Debugging methodologies
- `LARGE_GRID_VERIFICATION.md` - Large-scale grid validation

---

## Architecture (`architecture/`)

High-level project design:

- `PROJECT_PTM_RL.md` - Complete project specification
- `STATUS.md` - Development status tracking

---

## Layer Documentation

### PyTokEq (`layer1-pytokeq/`)

Equilibrium solver documentation (see subdirectory for details).

---

## Quick Navigation

**For new users:**
1. Start with [`../README.md`](../README.md) - Project overview and quick start
2. Read [`architecture/PROJECT_PTM_RL.md`](architecture/PROJECT_PTM_RL.md) - Full specification
3. Check [`../CHANGELOG.md`](../CHANGELOG.md) - Version history

**For developers:**
1. Review phase reports in chronological order
2. Check latest status in [`architecture/STATUS.md`](architecture/STATUS.md)
3. Follow handoff documents for context

**For researchers:**
1. Physics validation: See Phase 3 & 4 reports
2. RL performance: See Phase 5 reports
3. Benchmarks: Check [`../README.md`](../README.md) Performance section

---

## Document Conventions

### Report Structure

Each phase report typically includes:
1. **Objectives** - What was planned
2. **Implementation** - What was built
3. **Testing** - Validation results
4. **Performance** - Benchmarks and metrics
5. **Known Issues** - Limitations and future work
6. **Handoff** - Next phase preparation

### Status Indicators

- ✅ Complete and validated
- 🔄 In progress
- ⏸️ Paused/deferred
- ❌ Failed/deprecated
- ⚠️ Needs attention

---

## Maintenance

**Document Owners:**
- Phase reports: Respective phase leads (小A, 小P)
- Architecture docs: ∞ (PM)
- PyTokEq docs: 小P (Physics lead)

**Update Policy:**
- Phase reports: Finalized at phase completion, minimal updates
- Architecture docs: Updated as project evolves
- README files: Keep current with latest changes

---

**Last Updated:** 2026-03-17  
**Maintained by:** PTM-RL Team
