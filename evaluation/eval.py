from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:8000"
DATASET_PATH = Path("eval_dataset.json")
RESULTS_JSON = Path("eval_results.json")
RESULTS_MD = Path("eval_report.md")
JUDGE_MODEL = "claude-haiku-4-5"
ASK_TIMEOUT = 60

REFUSAL_PATTERNS = [
    r"\bdon't\s+see\b",
    r"\bdon't\s+have\b",
    r"\bnot\s+(in|covered|available|included|mentioned|present)\b",
    r"\bno\s+information\b",
    r"\bisn't\s+in\b",
    r"\bcannot\s+find\b",
    r"\bunable\s+to\s+find\b",
    r"\bnot\s+found\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)



@dataclass
class AskResult:
    retrieved_urls: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    answer: str = ""
    citations: list[dict] = field(default_factory=list)


@dataclass
class JudgeScore:
    faithfulness: int = 0
    relevance: int = 0
    citation_correctness: int = 0
    rationale: str = ""


@dataclass
class PairResult:
    id: str
    type: str
    question: str
    expected_urls: list[str]
    should_refuse: bool

    retrieved_urls: list[str]
    answer: str
    citation_count: int

    precision_at_5: float
    refused: bool
    refusal_correct: bool | None  # None if not adversarial
    judge: JudgeScore | None      # None if adversarial (judge not applicable)


def ask(base_url: str, question: str) -> AskResult:
    """POST /ask, parse the SSE stream, return collected fields."""
    result = AskResult()
    url = f"{base_url.rstrip('/')}/ask"
    payload = {"question": question}

    with httpx.stream("POST", url, json=payload, timeout=ASK_TIMEOUT) as r:
        r.raise_for_status()
        event_type = None
        for line in r.iter_lines():
            if not line:
                event_type = None
                continue
            if line.startswith("event: "):
                event_type = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
                if event_type == "retrieval":
                    result.retrieved_chunks = data.get("chunks", [])
                    # Use canonical .md URL for precision@k comparison.
                    result.retrieved_urls = [c["url"] for c in result.retrieved_chunks]
                elif event_type == "text":
                    result.answer += data.get("delta", "")
                elif event_type == "citation":
                    result.citations.append(data)
    return result


def precision_at_k(retrieved: list[str], expected: list[str], k: int = 5) -> float:
    """Fraction of top-k retrieved URLs that appear in expected_urls.
    """
    if not expected:
        return float("nan")  # not applicable (adversarial)
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for u in top if u in expected)
    return hits / len(top)


def detect_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer))

JUDGE_SYSTEM = """\
You are an evaluator for a retrieval-augmented question answering system over the Anthropic Claude documentation. You receive a question, the chunks the retrieval system returned, and the answer the system produced. You score the answer on three dimensions (1-5):

- faithfulness: is every factual claim in the answer supported by the retrieved chunks? 5 = fully grounded. 1 = mostly hallucinated.
- relevance: does the answer address the question that was asked? 5 = directly and completely. 1 = off-topic.
- citation_correctness: are the citations attached to the right claims? 5 = every cited claim genuinely comes from the cited source. 1 = citations are wrong or absent when needed.

Respond with ONLY a JSON object, no preamble:
{"faithfulness": int, "relevance": int, "citation_correctness": int, "rationale": "1-2 sentences"}
"""

JUDGE_TEMPLATE = """\
<question>
{question}
</question>

<retrieved_chunks>
{chunks}
</retrieved_chunks>

<answer>
{answer}
</answer>

<citations>
{citations}
</citations>
"""


def judge(client: anthropic.Anthropic, question: str, result: AskResult) -> JudgeScore:
    chunks_text = "\n\n---\n\n".join(
        f"[{i+1}] {c.get('title', '')} ({c.get('url', '')})\n{c.get('header_path', '')}"
        for i, c in enumerate(result.retrieved_chunks)
    ) or "(none)"

    citations_text = "\n".join(
        f"- {c.get('title', '')}: \"{(c.get('cited_text') or '')[:200]}...\""
        for c in result.citations
    ) or "(none)"

    user = JUDGE_TEMPLATE.format(
        question=question,
        chunks=chunks_text,
        answer=result.answer or "(empty)",
        citations=citations_text,
    )

    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()

    # code fences despite the instructions. Find the first {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        print(f"  ! judge returned no JSON object\n    raw: {text[:200]}", file=sys.stderr)
        return JudgeScore(rationale="parse_error: no JSON object found")

    try:
        data = json.loads(match.group(0))
        return JudgeScore(
            faithfulness=int(data.get("faithfulness", 0)),
            relevance=int(data.get("relevance", 0)),
            citation_correctness=int(data.get("citation_correctness", 0)),
            rationale=str(data.get("rationale", "")),
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"  ! judge parse failed: {e}\n    raw: {text[:200]}", file=sys.stderr)
        return JudgeScore(rationale=f"parse_error: {e}")

def run_eval(base_url: str) -> list[PairResult]:
    dataset = json.loads(DATASET_PATH.read_text())
    pairs = dataset["pairs"]
    client = anthropic.Anthropic()

    results: list[PairResult] = []
    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pair['id']} ({pair['type']}): {pair['question'][:60]}...")
        try:
            ar = ask(base_url, pair["question"])
        except Exception as e:
            print(f"  ! /ask failed: {e}")
            continue

        p_at_5 = precision_at_k(ar.retrieved_urls, pair["expected_urls"], k=5)
        refused = detect_refusal(ar.answer)
        refusal_correct = (refused == pair["should_refuse"]) if pair["should_refuse"] else None

        # Run judge for non-adversarial pairs (judging a "I don't know" answer
        # against irrelevant chunks would be meaningless).
        if pair["type"] != "adversarial":
            j = judge(client, pair["question"], ar)
        else:
            j = None

        results.append(PairResult(
            id=pair["id"],
            type=pair["type"],
            question=pair["question"],
            expected_urls=pair["expected_urls"],
            should_refuse=pair["should_refuse"],
            retrieved_urls=ar.retrieved_urls,
            answer=ar.answer,
            citation_count=len(ar.citations),
            precision_at_5=p_at_5,
            refused=refused,
            refusal_correct=refusal_correct,
            judge=j,
        ))

        # One-line progress summary
        if j:
            print(f"    p@5={p_at_5:.2f}  faith={j.faithfulness}  rel={j.relevance}  cite={j.citation_correctness}")
        else:
            mark = "✓" if refusal_correct else "✗"
            print(f"    refusal={refused} (expected {pair['should_refuse']}) {mark}")

    return results

def _mean(values: list[float]) -> float:
    values = [v for v in values if v == v]  # drop NaN
    return sum(values) / len(values) if values else 0.0


def aggregate(results: list[PairResult]) -> dict[str, Any]:
    by_type: dict[str, list[PairResult]] = defaultdict(list)
    for r in results:
        by_type[r.type].append(r)

    summary: dict[str, Any] = {
        "n": len(results),
        "by_type": {t: len(rs) for t, rs in by_type.items()},
    }

    non_adv = [r for r in results if r.type != "adversarial"]
    if non_adv:
        summary["precision_at_5_mean"] = _mean([r.precision_at_5 for r in non_adv])
        judged = [r.judge for r in non_adv if r.judge]
        if judged:
            summary["judge"] = {
                "faithfulness_mean":          _mean([j.faithfulness for j in judged]),
                "relevance_mean":             _mean([j.relevance for j in judged]),
                "citation_correctness_mean":  _mean([j.citation_correctness for j in judged]),
            }

    adv = by_type.get("adversarial", [])
    if adv:
        correct = sum(1 for r in adv if r.refusal_correct)
        summary["refusal_accuracy"] = correct / len(adv)
        summary["refusal_correct"] = correct
        summary["refusal_total"] = len(adv)

    return summary


def render_markdown(results: list[PairResult], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Eval Report")
    lines.append(f"\n_Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")

    lines.append("## Summary\n")
    lines.append(f"- Total pairs: **{summary['n']}**")
    lines.append(f"- Breakdown: {summary['by_type']}")
    if "precision_at_5_mean" in summary:
        lines.append(f"- precision@5 (mean): **{summary['precision_at_5_mean']:.2f}**")
    if "judge" in summary:
        j = summary["judge"]
        lines.append(f"- faithfulness (mean): **{j['faithfulness_mean']:.2f} / 5**")
        lines.append(f"- relevance (mean): **{j['relevance_mean']:.2f} / 5**")
        lines.append(f"- citation_correctness (mean): **{j['citation_correctness_mean']:.2f} / 5**")
    if "refusal_accuracy" in summary:
        lines.append(f"- refusal accuracy (adversarial): **{summary['refusal_correct']}/{summary['refusal_total']}**")

    lines.append("\n## Per-question results\n")
    lines.append("| id | type | p@5 | faith | rel | cite | refused | answer (truncated) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        p = f"{r.precision_at_5:.2f}" if r.precision_at_5 == r.precision_at_5 else "—"
        if r.judge:
            f_, rel, cit = r.judge.faithfulness, r.judge.relevance, r.judge.citation_correctness
        else:
            f_, rel, cit = "—", "—", "—"
        refused = "✓" if r.refusal_correct else ("✗" if r.refusal_correct is False else "—")
        ans = r.answer.replace("\n", " ").replace("|", "\\|")[:80] + ("…" if len(r.answer) > 80 else "")
        lines.append(f"| {r.id} | {r.type} | {p} | {f_} | {rel} | {cit} | {refused} | {ans} |")

    # Error analysis: anything that scored badly or missed retrieval.
    lines.append("\n## Failures & weak spots\n")
    flagged = []
    for r in results:
        reasons = []
        if r.type != "adversarial" and r.precision_at_5 == 0:
            reasons.append("precision@5 = 0 (retrieved no expected URLs)")
        if r.judge and r.judge.faithfulness <= 2:
            reasons.append(f"faithfulness {r.judge.faithfulness}/5")
        if r.judge and r.judge.citation_correctness <= 2:
            reasons.append(f"citation_correctness {r.judge.citation_correctness}/5")
        if r.type == "adversarial" and not r.refusal_correct:
            reasons.append("did not refuse out-of-corpus question")
        if reasons:
            flagged.append((r, reasons))

    if not flagged:
        lines.append("_No major failures flagged._")
    else:
        for r, reasons in flagged:
            lines.append(f"- **{r.id}** ({r.type}): {'; '.join(reasons)}")
            lines.append(f"  - Q: {r.question}")
            if r.judge and r.judge.rationale:
                lines.append(f"  - judge: {r.judge.rationale}")

    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", DEFAULT_BASE_URL))
    args = parser.parse_args()

    print(f"-> running eval against {args.base_url}")
    results = run_eval(args.base_url)
    summary = aggregate(results)

    # Serialize: dataclasses -> dicts.
    payload = {
        "summary": summary,
        "results": [
            {**asdict(r), "judge": asdict(r.judge) if r.judge else None}
            for r in results
        ],
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"-> wrote {RESULTS_JSON}")

    RESULTS_MD.write_text(render_markdown(results, summary))
    print(f"-> wrote {RESULTS_MD}")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()