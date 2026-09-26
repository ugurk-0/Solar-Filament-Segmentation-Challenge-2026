# Scientific agent skills review — 2026-09-26

## Selection criterion

The useful unit is a reviewed, task-specific procedure, not a large installed collection.
GitHub stars measure repository interest, not skill usage, mathematical correctness or
improvements to segmentation PQ. No universal rating or per-skill adoption ranking was
verified. Counts below are rounded values displayed by GitHub when checked on this date.

| Repository | Stars | Forks | Relevance here |
|---|---:|---:|---|
| [Anthropic skills](https://github.com/anthropics/skills) | 178.4k | 21.1k | General skill examples; popularity does not establish specialist statistics quality |
| [K-Dense Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills) | 46.7k | 4.2k | Most relevant of these inspected collections for scientific reasoning, statistics and symbolic mathematics |
| [OpenAI skills](https://github.com/openai/skills) | 27.6k | 1.9k | Catalog marked deprecated in favor of OpenAI Plugins; not the preferred new scientific dependency |

This is a comparison of inspected repositories, not an exhaustive GitHub ranking.

## Focused shortlist

1. [Scientific critical thinking](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-critical-thinking/SKILL.md):
   applied in this session to hypothesis framing, confounding, evidence strength and the
   distinction between exploratory results and independently assessed generalization.
2. [Statistical analysis](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/statistical-analysis/SKILL.md):
   reviewed for paired design, effect sizes, uncertainty and assumptions. The experiment uses
   paired bootstrap differences in PQ, rather than applying a generic normality-test/test-selection
   recipe automatically. A date-block sensitivity interval addresses some dependence.
3. [SymPy](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/sympy/SKILL.md):
   reviewed for future exact symbolic derivations and assumption-aware identities. Numerical
   mask geometry and focused regression tests were sufficient for this experiment; no symbolic
   calculation was claimed or needed.
4. [SciVisAgentSkills](https://github.com/KuangshiAi/SciVisAgentSkills): inspected as an alternative
   for scientific visualization. Its ParaView/napari/VMD/TTK scope is less direct than the existing
   Matplotlib pipeline for this two-dimensional task.

Only the relevant remote guidance was used; no global skills or additional scientific software
stacks were installed. This avoids importing a large collection of unrelated procedures.
The existing skill-installer guidance was consulted for discovery and installation conventions.

## Evidence limitation and citation

The K-Dense paper explicitly states that it reports **no task-level evaluation and no host
selection rate**. Its repository popularity therefore does not support a claim that loading
these skills improves this agent's accuracy or this model's PQ. The code changes and measured
experiments provide the evidence for this task.

Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026).
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065).
DOI: [10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
The current arXiv record was checked before citing it.
