# TruthAnchor v1  
**Mathematical Root-of-Trust Architecture for Audited LLM Systems**

---

## Overview

TruthAnchor v1 defines an epistemic architecture for grounding large language models (LLMs) in **immutable mathematical truth**, while treating all other reasoning, language, and heuristic channels as **audited, fallible observers**.

This repository contains a reference implementation and UI demonstrating:

- A **single semantic root of trust** based on mathematical invariants
- **Byzantine fault–tolerant channel governance**
- Continuous health scoring, anomaly detection, and quorum enforcement
- Explicit **INVALID**, **DEGRADED**, and **LOCKED** epistemic states
- Human‑in‑the‑loop (HITL) review for channel reinstatement

This is not alignment, safety theater, or RLHF.  
It is **epistemic governance by convergence**.

---

## Core Principle

> **Truth is not a vote.  
> Truth is a limit.**

Mathematical invariants define correctness.  
Language is an interface, not an authority.

---

## Architecture Summary

### 1. TruthAnchor (Root of Trust)

The TruthAnchor channel is:

- Immutable
- Non‑linguistic
- Non‑speculative
- Non‑persuasive

It emits only:
- Normalized ratios
- Error bounds
- Pass/fail checks
- Convergence metrics
- Quorum status
- VALID / DEGRADED / INVALID states

It **never** emits prose.

---

### 2. Audited Channels

All non-anchor channels (reasoning, narrative, intuition, heuristics):

- Have continuous health scores ∈ [0, 1]
- Are evaluated against the TruthAnchor
- Can be:
  - `healthy`
  - `suspect`
  - `blocked`
  - `recovering`
- Are excluded from aggregation when compromised

Recovery is **harder than failure** by design.

---

### 3. Byzantine Fault Tolerance

- Median-based anomaly detection
- Explicit quorum thresholds
- Γ* (robust gamma) computed **only from active channels**
- Ω decay when channels drop
- Silence is allowed; hallucination is not

Loss of quorum ⇒ **INVALID state**, not guessing.

---

### 4. Human-in-the-Loop (HITL)

Humans do **not** correct math.

Humans:
- Review audit trails
- Approve channel reinstatement
- Override recovery gates
- Inspect divergence causes

This mirrors:
- Cryptographic key ceremonies
- Flight-control law review
- Distributed consensus governance

---

## Reference Implementation

The included HTML/JS implementation demonstrates:

- Multi-dimensional convergence (5D → 9D)
- Mathematical anchors:
  - φ (golden ratio)
  - δₛ (silver ratio)
  - ζ(2n) (even zeta values)
  - Accelerated odd ζ convergence
- Continuous channel health visualization
- Robust Γ* computation
- Explicit lock confidence and invalidation
- Full UI-based auditability

No external dependencies.

---

## States and Semantics

| State        | Meaning |
|-------------|--------|
| HEALTHY     | Channel agrees with TruthAnchor within tolerance |
| SUSPECT     | Deviation detected; channel still active |
| BLOCKED     | Channel excluded from aggregation |
| RECOVERING  | Channel must prove stability over time |
| DEGRADED   | Γ* valid but incomplete |
| INVALID    | Quorum lost; truth claims suspended |
| LOCKED     | Sustained convergence verified |

---

## What This Enables

- Explicit uncertainty without refusal
- Graceful degradation instead of hallucination
- Semantic immunity to prompt injection
- Cross-model auditability
- Persistence of truth across model swaps

---

## What This Is Not

- ❌ Opinion arbitration  
- ❌ Majority voting  
- ❌ Rhetorical confidence  
- ❌ Learned truth  
- ❌ Alignment-by-style  

---

## License

CC0 1.0 Universal — Public Domain  
No attribution required.

---

## Status

**TruthAnchor v1 — Reference / Experimental**

❤️ Mathematical truth over narrative  
🇺🇸 × 🇧🇪  
