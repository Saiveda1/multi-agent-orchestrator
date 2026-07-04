# Multi-Agent Orchestrator Project Document

**Prepared For:** Sai Veda  
**GitHub Publishing Account:** Nikeshk834  
**Repository Slug:** `03-multi-agent-orchestrator`  
**Verified Test Count From Portfolio Index:** 69  

## Background

A from-scratch agentic framework — LangGraph / AutoGPT-class, but a clean,
dependency-light design of its own. A **planner** decomposes a natural-language
task into a **DAG of tool calls**; an **executor** runs that DAG respecting
dependencies and running independent branches in parallel; **guardrails** validate
inputs, tool arguments, and outputs; and a **self-correction loop** turns real
execution feedback into a repaired plan and retries.

Everything runs **fully offline and deterministically**. There is no external LLM
API. The "reasoning" is provided by a deterministic rule/pattern policy that sits
behind a clean `LLMEngine` interface, so a real model can be dropped in unchanged.

> **Headline result (31-task benchmark, measured):** the self-correction loop
> lifts task success from **41.9% → 100.0%** — a **+58.1 point** improvement —
> at an average of **1.58 attempts** and **$0.259** simulated cost per task.

---

## Why this is interesting

Agent frameworks live or die on two things: **planning into a correct dependency
graph**, and **recovering from the model's inevitable mistakes**. This project
builds both as real, testable components and *measures* the payoff of recovery.

The offline engine is intentionally *fast and imperfect* on its first pass —
exactly like a language model producing plausible-but-wrong tool calls:

| First-pass mistake (realistic)                    | Guardrail that catches it | Repair applied                    |
|---------------------------------------------------|---------------------------|-----------------------------------|
| `2 ^ 10` (uses bitwise-xor for power)             | tool error                | rewrite `^` → `**`                |
| `Convert 10 kilometers to miles` (English units)  | tool error (unknown unit) | normalize `kilometers→km`, `miles→mi` |
| code snippet passed under the wrong arg key       | **schema** validation     | move value to the `code` key      |
| returns `0.375` when asked for a percentage       | **output** contract       | wrap expression `× 100`           |

None of these repairs peek at the ground-truth answer — the loop only ever sees
execution feedback, so every point of lift is earned.

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
   task prompt  ─────────▶│                   AGENT                       │
   "Convert 12 km to mi   │  engine · toolbox · blackboard · bus · exec   │
    and ... then add"     └───────────────┬──────────────────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  PLANNER  (LLMEngine interface)       │
                       │  OfflineRuleEngine: prompt → Steps    │
                       └──────────────────┬───────────────────┘
                                          │  list[Step]
                       ┌──────────────────▼───────────────────┐
                       │  DAG   Kahn topo-sort → parallel      │
                       │  levels;  $s1 / Ref edges; acyclic    │
                       └──────────────────┬───────────────────┘
                                          │  levels
                 ┌────────────────────────▼─────────────────────────┐
                 │                    EXECUTOR                        │
                 │  per level → ThreadPool (independent steps ‖)      │
                 │                                                    │
                 │  resolve Refs/$vars → validate_args ─┐  guardrails │
                 │        │                             │             │
                 │        ▼                             ▼             │
                 │   ┌─────────┐  ┌──────────┐  ┌──────────────┐      │
                 │   │calculator│  │unit_conv │  │corpus_search │ ...  │
                 │   │ (AST safe)│ │          │  │  (TF-IDF)    │      │
                 │   └────┬────┘  └────┬─────┘  └──────┬───────┘      │
                 │        └── results → Blackboard ────┘  MessageBus  │
                 │                     │                              │
                 │            validate_output (contract)              │
                 └─────────────────────┬──────────────────────────────┘
                                       │  ok? ── yes ─▶ final answer + trace
                                       │
                                       └── no ─▶ engine.repair(error) ─▶ retry
                                                 (self-correction, up to N)
```

Core abstractions: **`Tool`** (name, typed schema, `run`), **`Agent`**,
**`Blackboard`** (thread-safe shared memory + write history), **`MessageBus`**
(pub/sub event log), **`Planner`**, **`Executor`**, and the **`LLMEngine`**
interface with its offline implementation.

---

## Results (measured)

31-task suite, deterministic, offline. Full tables in
[`benchmarks/results.md`](benchmarks/results.md); raw per-task data in
[`benchmarks/results.csv`](benchmarks/results.csv).

| Condition                | Success rate | Solved | Avg steps | Avg attempts | Avg sim. cost | Avg tokens |
|--------------------------|--------------|--------|-----------|--------------|---------------|------------|
| Without self-correction  | 41.9%        | 13/31  | 1.35      | 1.00         | $0.16524      | 238        |
| **With self-correction** | **100.0%**   | 31/31  | 1.35      | 1.58         | $0.25834      | 381        |

**Self-correction lift: +58.1 percentage points.** The loop costs ~0.58 extra
attempts and ~$0.09 extra simulated spend per task to recover every otherwise-failing task.

Tool-call distribution (with correction): `calculator` 20, `unit_convert` 14,
`corpus_search` 9, `code_check` 3.

---

## Project Purpose

This repository is part of the AI engineering portfolio and focuses on the following problem space:

- Planner/executor DAG, tools, self-correction
- Headline result from the portfolio index: Self-correction lifts task success **41.9% → 100%**

## What This Project Solves

This project provides a production-style implementation with benchmark evidence and operational checks committed into the repository.

## Technical Approach

# Architecture

## 1. Design goals

1. **Real components, not a demo script.** Every part — planner, DAG scheduler,
   tools, guardrails, accounting — is a standalone, tested unit.
2. **Offline and deterministic.** Zero external API. The "LLM" is a rule/pattern
   policy behind an interface, so results are reproducible and CI-friendly, and a
   real model can be swapped in without touching the executor or tools.
3. **Measure recovery, honestly.** The self-correction loop must *earn* its lift:
   it may only use execution feedback (errors, failed validators), never the
   task's ground-truth answer.

## 2. Component model

```
Agent ── owns ──▶ LLMEngine (plan/repair)   Toolbox {name → Tool}
             └──▶ Blackboard (shared memory) MessageBus (events)
             └──▶ Executor (schedule + guardrails + accounting)
```

### Tool
`Tool` = `name` + human `description` + typed `schema` + `run(args) -> dict`.
The schema is `{arg: {type, required, default}}`. Tools raise `ToolError` for
recoverable domain failures; guardrails (not tools) enforce the schema, so the
same validator protects every tool and the executor can attribute a failure to
the correct stage.

The four tools are deliberately heterogeneous to exercise the framework:
- **`calculator`** — evaluates arithmetic by walking a *restricted AST*
  (`ast.parse(mode="eval")`). Only literals, whitelisted binary/unary operators,
  whitelisted functions (`sqrt`, `abs`, `round`, `min/max`, `log`, ...) and named
  constants (`pi`, `e`) are allowed. Names, attribute access, calls to arbitrary
  builtins, comprehensions, lambdas and the bitwise operators are rejected.
  **`eval`/`exec` are never used** (asserted by a test that AST-scans the module).
- **`unit_convert`** — length / mass via linear base-unit factors, temperature via
  affine C/F/K formulas; unknown or incompatible units raise `ToolError`.
- **`corpus_search`** — TF-IDF + cosine over a small synthetic knowledge base
  (`scikit-learn`); returns the top fact's numeric value plus ranked hits.
- **`code_check`** — AST lint flagging `eval`/`exec`, bare `except:`, `== None`,
  and mutable default arguments; syntax errors are reported, not raised.

### Planner and the DAG
The `LLMEngine.plan(prompt)` returns an ordered `list[Step]`. A `Step` names a
tool, an args dict, and optional explicit deps. Dependencies are also **derived**
automatically from two fo

## Benchmark And Validation Evidence

The portfolio root documents **69 passing tests** for this project, and the repo quickstart uses `make test` as the standard validation path. The benchmark outputs committed in `benchmarks/` and the generated visuals in `assets/` are the evidence package for this delivery.

### results.md

# Benchmark Results

Task suite: **31** multi-step tasks. Deterministic, fully offline.

| Condition | Success rate | Solved | Avg steps | Avg attempts | Avg sim. cost | Avg tokens |
|---|---|---|---|---|---|---|
| Without self-correction | 41.9% | 13/31 | 1.35 | 1.00 | $0.16524 | 238 |
| With self-correction | 100.0% | 31/31 | 1.35 | 1.58 | $0.25834 | 381 |

**Self-correction lift: +58.1 percentage points** (41.9% -> 100.0%).

## Tool-call distribution (with self-correction)

| Tool | Calls |
|---|---|
| `calculator` | 20 |
| `unit_convert` | 14 |
| `corpus_search` | 9 |
| `code_check` | 3 |

## Per-task detail

| Task | Category | Without SC | With SC | Attempts (SC) | Steps |
|---|---|---|---|---|---|
| T01 | calc | PASS | PASS | 1 | 1 |
| T02 | calc | PASS | PASS | 1 | 1 |
| T03 | calc | PASS | PASS | 1 | 1 |
| T04 | calc | PASS | PASS | 1 | 1 |
| T05 | calc_pow | FAIL | PASS | 2 | 1 |
| T06 | calc_pow | FAIL | PASS | 2 | 1 |
| T07 | calc_pow | FAIL | PASS | 2 | 1 |
| T08 | convert | PASS | PASS | 1 | 1 |
| T09 | convert | PASS | PASS | 1 | 1 |
| T10 | convert | FAIL | PASS | 2 | 1 |
| T11 | convert | FAIL | PASS | 2 | 1 |
| T12 | convert | FAIL | PASS | 2 | 1 |
| T13 | convert | FAIL | PASS | 2 | 1 |
| T14 | convert_calc | FAIL | PASS | 2 | 2 |
| T15 | convert_calc | FAIL | PASS | 2 | 2 |
| T16 | percent | FAIL | PASS | 2 | 1 |
| T17 | percent | FAIL | PASS | 2 | 1 |
| T18 | percent | FAIL | PASS | 2 | 1 |
| T19 | corpus | PASS | PASS | 1 | 1 |
| T20 | corpus | PASS | PASS | 1 | 1 |
| T21 | corpus | PASS | PASS | 1 | 1 |
| T22 | corpus | PASS | PASS | 1 | 1 |
| T23 | corpus | PASS | PASS | 1 | 1 |
| T24 | corpus_calc | PASS | PASS | 1 | 2 |
| T25 | corpus_calc | PASS | PASS | 1 | 2 |
| T26 | code | FAIL | PASS | 2 | 1 |
| T27 | code | FAIL | PASS | 2 | 1 |
| T28 | code | FAIL | PASS | 2 | 1 |
| T29 | parallel | FAIL | PASS | 2 | 3 |
| T30 | parallel | FAIL | PASS | 2 | 3 |
| T31 | showcase | FAIL | PASS | 2 | 4 |

## Visual Artifacts Reviewed

- `assets/kpi_dashboard.png`: assets/kpi_dashboard.png.
- `assets/execution_gantt.png`: assets/execution_gantt.png.
- `assets/success_rate.png`: assets/success_rate.png.
- `assets/tool_distribution.png`: assets/tool_distribution.png.

## Engineering Notes

The primary design and scale decisions are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md). The benchmark markdown in [`benchmarks/`](./benchmarks) and the generated figures in [`assets/`](./assets) should be read together: the markdown gives the measured numbers, and the screenshots make those results easier to inspect quickly during review.

## Files Included In This Repo

- [`README.md`](./README.md) for project overview, quickstart, and headline results
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design and scaling choices
- [`benchmarks/`](./benchmarks) for measured results from the committed runs
- [`assets/`](./assets) for generated screenshots and dashboards
- [`tests/`](./tests) for the automated validation suite

## Delivery Summary

This project document was prepared for **Sai Veda** so the repository reads like a real project handoff: what the system is for, what problem it solves, what evidence supports it, and where the benchmark and test artifacts live inside the repo.
