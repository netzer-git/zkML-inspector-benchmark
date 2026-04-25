# Example agent output

[`agent_output.example.json`](./agent_output.example.json) is a reference file showing the **exact format the grader expects** from an agent. The findings in it are fictional (projects `zkFoo`, `zkBar`) and do not correspond to any entry in the real benchmark dataset — they exist only to demonstrate the schema.

## Schema

The file is a flat JSON array. Each element is a finding object with **all 8 fields required** (the grader errors on missing fields):

```json
{
  "entry-id": "<project key>",
  "issue-name": "<3-7 word title>",
  "issue-explanation": "<one paragraph>",
  "severity": "Critical | Warning | Info",
  "category": "<one of 7 values or Other>",
  "security-concern": "<one of 6 values or Other>",
  "relevant-code": "file.rs:10-20, other.cu:3",
  "paper-reference": "Section 6.1.3: \"...\""
}
```

Findings from different projects may be interleaved in any order. The grader groups them by `entry-id` (case-insensitive) before matching.

## Closed-list values

Invalid values cause the grader to exit with a validation error.

**severity** — exactly one of:
- `Critical`
- `Warning`
- `Info`

**category** — answers: *"Where exactly did the implementation go wrong?"* Exactly one of:
- `Under-constrained Circuit` — the R1CS or PLONKish constraints are too "loose," allowing non-deterministic values that the verifier can't catch.
- `Protocol/Transcript Logic` — errors in the Fiat-Shamir implementation, weak hashing, or missing domain separation (especially common in Sumcheck/GKR rounds).
- `Specification Mismatch` — the code deviates from the paper or the formal model description (even if not immediately exploitable).
- `Numerical/Quantization Bug` — precision loss, fixed-point overflows, or rounding biases introduced during the ML-to-Field conversion.
- `Witness/Commitment Mismatch` — the proof doesn't correctly link the circuit logic to the external data commitments (e.g., the Merkle root of the weights is never checked).
- `Engineering/Prototype Gap` — "lazy" implementations, such as hardcoded constants, missing range checks, or unoptimized kernels that lead to memory safety issues.
- `Other`

**security-concern** — answers: *"If an attacker exploits this, what happens to the system's trust assumptions?"* Exactly one of:
- `Proof Forgery (Soundness)` — a malicious prover can generate a valid proof for an incorrect result or a different model.
- `Information Leakage (Privacy)` — private data (e.g., user prompts, LLM weights) is partially or fully recoverable from the proof or transcript.
- `Semantic Subversion (Integrity)` — the proof is mathematically sound, but it binds to the wrong inputs/outputs (e.g., proving the wrong model, or a "deepfake" inference).
- `Proof Malleability` — an attacker can modify a valid proof into another valid proof without knowing the underlying witnesses.
- `Denial of Proof (Reliability)` — a vulnerability that prevents an honest prover from generating a valid proof (e.g., crashing the GPU kernel or hitting an unprovable state).
- `Governance Bypass` — circumventing the auditing/policy layer because the ZK system doesn't actually enforce the claimed audit rules.
- `Other`

## Free-text fields

**`entry-id`** — the project key. Use whatever key identifies the (paper, codebase) pair in the dataset you are grading against. Case is normalized by the grader.

**`issue-name`** — short descriptive title, roughly 3-7 words. Used (together with `issue-explanation`) for matching agent findings to ground-truth findings via text similarity.

**`issue-explanation`** — one paragraph describing the root cause and impact. Contributes to match similarity alongside `issue-name`.

**`relevant-code`** — comma-separated `file:line` references:
- Line ranges: `file.rs:10-20`
- Single line: `file.cu:42`
- File only (no line): `config.yaml`
- Multiple: `src/a.rs:10-15, src/b.rs:3`
- Empty (no code applies to this finding): `""`

The grader matches code refs by basename (so `verifier.rs:38` matches ground-truth `src/util/verifier.rs:36-42`). Line proximity is scored — overlap is best, within 30 lines is partial credit, different files score zero.

**`paper-reference`** — cite a specific section, protocol, theorem, equation, or example from the paper, optionally with a quoted claim. The grader extracts structured identifiers (`Section 6.1.3`, `Protocol 1`, `Theorem 7.1`, `Eq. 33`, `Example 4.2`) and scores:
- Exact section match (1.0)
- Parent/child section match e.g. `Section 6.1` vs `Section 6.1.3` (0.6)
- Same top-level section (0.3)

A quoted claim in the reference is also compared by text similarity. Use `"-"` or `""` when the finding has no applicable paper claim.

## Running the grader against this example

```bash
python -m grader \
    --ground-truth your_findings.json \
    --agent-output examples/agent_output.example.json \
    --output grade_report.json \
    --output-md grade_report.md
```

Since the example findings describe fictional projects (`zkFoo`, `zkBar`) with no corresponding ground-truth entries, the grader will report `no ground truth available, skipping` for them. To see scoring in action, point `--agent-output` at output covering projects actually in your `--ground-truth` JSON.

## What to aim for

The grader rewards:
- **Correct severity** — under-reporting (calling a Critical issue a Warning) scores zero; over-reporting gets partial credit.
- **Precise code location** — overlapping line ranges in the right file score best.
- **Concrete paper anchoring** — cite the exact section or protocol; include the quoted claim when you can.
- **Finding the issue the dataset recorded** — agents are scored on matched pairs, so the `name + explanation` text should describe the same vulnerability a human auditor would flag. Extra Critical findings not in the dataset count against precision.
