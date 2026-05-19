# Reproduction Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the SmartSAT reproduction code with the paper-facing claims while keeping the Kaggle notebook runnable locally and excluding the notebook from the pushed commit.

**Architecture:** Keep the benchmark scoped to `uf20-91` because the PPO action and observation spaces are fixed. Make paper reward a single canonical behavior, require full SATfeatPy probing by default in code, remove paper time scaling from evaluation, and share CDCL decision/conflict helpers between training and evaluation paths.

**Tech Stack:** Python, unittest, Stable-Baselines3, sb3-contrib, SATfeatPy.

---

### Task 1: Regression Tests

**Files:**
- Create: `tests/test_reproduction_alignment.py`

- [ ] **Step 1: Add tests for paper reward, UF20 validation, no time scaling, and SATfeatPy defaults.**
- [ ] **Step 2: Run `python -m unittest tests.test_reproduction_alignment -v` and confirm failures reflect missing behavior.**

### Task 2: SmartSAT Environment and CDCL Helpers

**Files:**
- Modify: `src/smartsat_env.py`
- Modify: `src/cdcl_baseline.py`
- Modify: `src/evaluate.py`

- [ ] **Step 1: Enforce `uf20-91` in SmartSAT training and evaluation before building observations.**
- [ ] **Step 2: Change paper reward to a clause-score delta so cumulative episode reward is bounded by the final clause score.**
- [ ] **Step 3: Move decision enqueue and conflict learning/backtracking into CDCL helper methods used by both the env and solver loop.**
- [ ] **Step 4: Remove evaluation time scaling and always report raw total time unless explicitly using search-time diagnostics.**

### Task 3: SATfeatPy Strictness

**Files:**
- Modify: `src/satfeat_adapter.py`
- Modify: `README.md`

- [ ] **Step 1: Default full local-search probing to enabled in production code.**
- [ ] **Step 2: Raise a clear error if SATfeatPy cannot provide the full requested feature set in strict mode.**
- [ ] **Step 3: Document that Kaggle may override local-search probing only for runnable diagnostics, not strict paper claims.**

### Task 4: Kaggle Notebook Local-Only

**Files:**
- Modify but do not stage: `notebooks/LangSAT_Kaggle_Reproduce.ipynb`

- [ ] **Step 1: Remove time-scale environment setup.**
- [ ] **Step 2: Keep notebook runnable on Kaggle by detecting local-search availability and printing whether the run is strict or partial-feature diagnostic.**
- [ ] **Step 3: Verify notebook is modified locally but excluded from git staging.**

### Task 5: Verification and Push

**Files:**
- Stage code/docs/tests only, not notebook.

- [ ] **Step 1: Run `python -m unittest tests.test_reproduction_alignment -v`.**
- [ ] **Step 2: Run `python -m compileall src`.**
- [ ] **Step 3: Confirm `git diff --cached --name-only` excludes the notebook.**
- [ ] **Step 4: Commit and push branch `codex/repro-alignment-fixes`.**
