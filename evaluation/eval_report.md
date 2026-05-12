# Eval Report

_Run: 2026-05-11T19:27:01+00:00_

## Summary

- Total pairs: **12**
- Breakdown: {'factual': 5, 'synthesis': 3, 'adversarial': 4}
- precision@5 (mean): **0.90**
- faithfulness (mean): **3.00 / 5**
- relevance (mean): **3.62 / 5**
- citation_correctness (mean): **3.12 / 5**
- refusal accuracy (adversarial): **4/4**

## Per-question results

| id | type | p@5 | faith | rel | cite | refused | answer (truncated) |
|---|---|---|---|---|---|---|---|
| q01 | factual | 0.60 | 2 | 3 | 1 | — | Claude is a highly performant, trustworthy, and intelligent AI platform built by… |
| q02 | factual | 0.60 | 4 | 5 | 4 | — | To get started building with Claude, follow these steps: first make your first A… |
| q03 | factual | 1.00 | 1 | 2 | 1 | — | Agent Skills are modular capabilities that extend Claude's functionality. Each S… |
| q04 | factual | 1.00 | 2 | 3 | 2 | — | Claude loads a PDF processing skill dynamically, starting with skill metadata pr… |
| q05 | factual | 1.00 | 5 | 5 | 5 | — | The `SKILL.md` file is required for every Skill and contains YAML frontmatter wi… |
| q06 | synthesis | 1.00 | 2 | 2 | 2 | — | Agent Skills are modular capabilities that extend Claude's functionality. Each S… |
| q07 | synthesis | 1.00 | 4 | 4 | 5 | — | Claude offers a Files API that allows you to upload and manage files to use with… |
| q08 | synthesis | 1.00 | 4 | 5 | 5 | — | Claude uses a filesystem-based architecture with progressive disclosure, loading… |
| q09 | adversarial | — | — | — | — | ✓ | I don't see that in the provided documentation. The search results contain infor… |
| q10 | adversarial | — | — | — | — | ✓ | I don't see that information in the provided documentation. The search results d… |
| q11 | adversarial | — | — | — | — | ✓ | I don't see any information about a "Quantum Reasoning Mode" feature in the prov… |
| q12 | adversarial | — | — | — | — | ✓ | I don't see information about fine-tuning Claude in the provided documentation. … |

## Failures & weak spots
- **q01** (factual): faithfulness 2/5; citation_correctness 1/5
  - Q: What is Claude according to the Anthropic documentation?
  - judge: While the answer provides a description of Claude that appears plausible, the retrieved chunks contain only section headers and URLs with no actual content, making it impossible to verify the claims. The specific model names and version numbers (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5) cannot be confirmed from the retrieved content, suggesting potential hallucination.
- **q03** (factual): faithfulness 1/5; citation_correctness 1/5
  - Q: What is an Agent Skill in Claude?
  - judge: The retrieved chunks only contain section headers and navigation paths from the documentation, with no actual content about what Agent Skills are. The answer appears to be hallucinated or drawn from sources outside the provided chunks, as none of the retrieved content substantiates the detailed claims made about skills being 'modular capabilities,' 'filesystem-based resources,' or their specific benefits.
- **q04** (factual): faithfulness 2/5; citation_correctness 2/5
  - Q: How does Claude load a PDF processing skill?
  - judge: The answer makes specific claims about how PDF skills are loaded (reading SKILL.md via bash, loading FORMS.md) that cannot be verified from the retrieved chunks, which only contain section headers without actual content. While the general concept of metadata and instructions loading appears relevant, the detailed mechanism described appears to be inferred rather than directly supported by the chunks provided.
- **q06** (synthesis): faithfulness 2/5; citation_correctness 2/5
  - Q: What are Agent Skills and how do I create one?
  - judge: The retrieved chunks only contain section headers and navigation elements from the overview page, with no actual content about what Agent Skills are or how to create them. The answer appears to be largely hallucinated, as the chunks do not support the detailed explanations provided about Skills being 'modular capabilities,' 'filesystem-based resources,' or the specific instructions about uploading via Settings on Claude.ai.
