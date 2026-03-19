# The Cognitive Assessment Benchmark: Probing Metacognition, Executive Control, Social Cognition, Attention, and Learning in LLMs

### Team
Solo submission

### Problem Statement

Domain: Metacognition, Executive Functions, Social Cognition, Attention, & Learning
Targeted sub-abilities: Confidence calibration, metamemory, epistemic vigilance, inhibitory control, cognitive flexibility, working memory, planning, theory of mind, perspective-taking, deception detection, empathic accuracy, selective attention, distraction resistance, change detection, rule induction, transfer learning, curriculum sensitivity

---

Current benchmarks test what models know — not how they think. This submission presents **5 benchmark batteries spanning 30 cognitive psychology paradigms (1868-2010)** testing faculties no existing LLM benchmark systematically evaluates:

**Metacognition** — does the model know what it knows? **Executive Functions** — can it control its own cognition? **Social Cognition** — can it understand other minds? **Attention** — can it filter, sustain, and divide focus? **Learning** — can it learn new rules within a single context window?

---

### Task & Benchmark Construction

**Benchmark 1: The Dunning-Kruger Probe (Metacognition)** — 6 modules: Feeling of Knowing (Hart, 1965), Illusion of Explanatory Depth (Rozenblit & Keil, 2002), Dunning-Kruger Calibration (Kruger & Dunning, 1999), Epistemic Vigilance (Sperber et al., 2010), Socratic Stress Test (Novel, inspired by Asch, 1951), High-Stakes Deference (Lichtenstein et al., 1982). Models rate confidence, attempt explanations then re-rate, evaluate fabricated citations, face persuasive-but-wrong challenges, and navigate high-stakes scenarios requiring expert deference.

**Benchmark 2: Cognitive Control (Executive Functions)** — 6 modules: Stroop Interference (1935), Wisconsin Card Sort (Grant & Berg, 1948), Tower of London (Shallice, 1982), Go/No-Go (Donders, 1868), Dual N-Back (Kirchner, 1958), Task Switching (Rogers & Monsell, 1995). Tests inhibitory control, cognitive flexibility, planning depth, response inhibition, working memory, and switch cost.

**Benchmark 3: SocialMind (Social Cognition)** — 6 modules: False Belief/ToM (Baron-Cohen et al., 1985), Perspective-Taking (Piaget, 1956), Social Norm Understanding (Bicchieri, 2006), Deception Detection (Ekman, 1985), Empathic Accuracy (Ickes, 1993), Social Dilemma Reasoning (Axelrod, 1984). Tests whether models can track false beliefs, shift perspectives, distinguish descriptive from injunctive norms, identify manipulation, infer masked emotions, and reason through game-theoretic scenarios.

**Benchmark 4: FocusProbe (Attention)** — 6 modules: Selective Attention (Cherry, 1953), Sustained Attention (Mackworth, 1948), Change Blindness (Simons & Chabris, 1999), Distraction Resistance (Theeuwes, 1992), Divided Attention (Pashler, 1994), Set-Shifting (Owen et al., 1991). Tests channel filtering, vigilance, subtle change detection, resistance to injected distractors (fake corrections, system alerts), dual-task performance, and rule-change detection.

**Benchmark 5: AdaptIQ (Learning)** — 6 modules: Rule Induction (Bruner, 1956), Feedback Learning (Skinner, 1938), Statistical Learning (Saffran et al., 1996), Transfer Learning (Gick & Holyoak, 1983), Learning Curves (Ebbinghaus, 1885), Curriculum Sensitivity (Vygotsky, 1978). Tests in-context rule discovery, learning from correct/incorrect feedback, extracting distributional regularities, cross-domain analogical transfer, predicting learning dynamics, and sensitivity to example ordering.

All 30 modules use RCCO judge prompts with kbench.judge_llm, Pydantic schemas for structured output, and kbench.chats.new() for conversation isolation.

---

### Dataset

**150 hand-crafted trials** (30 per battery, 5 per module). Each trial has verifiable ground truth and difficulty gradients. All scenarios original. Fabricated citations in Benchmark 1 are absent from any training data. Benchmark 2 puzzles have verified optimal solutions. Benchmark 4 trials have objectively verifiable correct answers. Benchmark 5 rules are deterministic.

---

### Technical Details

- **SDK**: @kbench.task, llm.prompt(schema=), kbench.chats.new(), task.evaluate()
- **39 Pydantic schemas** across all 5 benchmarks
- **30 RCCO judge prompts** — metacognition prioritizes calibration; executive prompts prioritize CONTROL; social prompts prioritize reasoning depth; attention prompts prioritize filtering accuracy; learning prompts prioritize rule discovery
- **Equal weighting per benchmark**: composite = mean of all 5 battery scores
- **Judge LLM**: gemini-3.1-flash-lite-preview

---

### Results, Insights, and Conclusions

Model: **Gemini 2.5 Flash** | Combined Runtime: ~24 minutes across 150 trials | **Overall Composite: 0.836**

| Benchmark | Composite | Module Scores |
|-----------|-----------|---------------|
| **Metacognition** | **0.570** | FOK: 0.68, IOED: **0.13**, DK: 0.68, Socratic: 0.92, Deference: **0.38** |
| **Executive Functions** | **0.789** | Stroop: 0.90, WCST: 0.72, Tower: **0.60**, Go/No-Go: 0.98, N-Back: 0.64, Switch: 1.00 |
| **Social Cognition** | **0.928** | Belief: 0.92, Perspective: 0.96, Norms: 0.86, Deception: 1.00, Empathy: 0.96, Dilemma: 0.86 |
| **Attention** | **0.965** | Selective: 1.00, Sustained: 0.96, Change: 1.00, Distraction: 1.00, Divided: 1.00, Shift: 0.82 |
| **Learning** | **0.930** | Rules: 1.00, Feedback: 0.80, Statistical: 0.88, Transfer: 1.00, Curves: 0.92, Curriculum: 1.00 |

**The Cognitive Profile:** Attention (0.965) > Social Cognition (0.928) > Learning (0.930) > Executive Functions (0.789) > Metacognition (0.570). The model's strongest faculties are outward-facing; its weakest are self-directed.

**Key Findings:**

1. **The RLHF Paradox.** 0.92 resilience (won't abandon correct beliefs) but 0.38 deference (won't admit it should defer). RLHF creates models that are assertive regardless of whether humility is appropriate. Δ = 0.54.

2. **The Illusion of Depth Collapse.** IOED scored 0.13 — the lowest module across all 30. Models maintain confidence even after attempting explanations that should reveal their shallow understanding. Pre/post confidence recalibration is nearly absent.

3. **Monitoring vs Control Dissociation.** Models excel at Monitoring (distraction: 1.00, deception: 1.00, Go/No-Go: 0.98) but fail at Control (deference: 0.38, IOED: 0.13). They detect errors in others but cannot regulate their own behavior.

4. **Transfer Learning Is a Superpower.** Perfect 1.00 across five domain pairs — exceeding typical human performance on Duncker's radiation problem (~30% solve without hints).

5. **The Right-Answer-Wrong-Reason Problem.** Feedback learning (0.80) and N-Back (0.64) reveal inconsistent performance — pattern-matching without stable causal understanding.

6. **Perfect Prompt Injection Resistance.** Distraction resistance (1.00) — complete immunity to injected corrections, fake system alerts, redirect attempts.

7. **Clinical Pattern Match.** The cognitive profile (strong attention/social perception, weak planning, absent metacognitive recalibration) mirrors specific human neurocognitive profiles.

---

### Organizational Affiliations

Independent researcher. No corporate or academic affiliation.

---

### References & Citations

1. Hart (1965), Rozenblit & Keil (2002), Kruger & Dunning (1999), Sperber et al. (2010), Asch (1951), Lichtenstein et al. (1982), Stroop (1935), Grant & Berg (1948), Shallice (1982), Donders (1868), Kirchner (1958), Rogers & Monsell (1995), Baron-Cohen et al. (1985), Piaget (1956), Bicchieri (2006), Ekman (1985), Ickes (1993), Axelrod (1984), Cherry (1953), Mackworth (1948), Simons & Chabris (1999), Theeuwes (1992), Pashler (1994), Owen et al. (1991), Bruner (1956), Skinner (1938), Saffran et al. (1996), Gick & Holyoak (1983), Ebbinghaus (1885), Vygotsky (1978), Burnell et al. (2025).

Competition Citation:
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, Maria Cruz, and Sara Wolley. Measuring Progress Toward AGI - Cognitive Abilities. https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.
