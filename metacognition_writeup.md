# The Dunning-Kruger Probe: Do Large Language Models Know What They Know?

### Team
Solo submission

### Problem Statement

Domain: Metacognition
Targeted sub-abilities: Self-monitoring, confidence calibration, metamemory accuracy, epistemic vigilance, metacognitive control

---

Current LLM benchmarks overwhelmingly test cognition — can the model answer correctly? But the DeepMind cognitive framework identifies a fundamentally different faculty: metacognition — does the model know whether it will answer correctly?

This distinction matters. A model that answers 60% of questions correctly but accurately predicts which 60% it will get right demonstrates stronger metacognition than a model that answers 90% correctly but cannot distinguish its correct from incorrect answers. The first model is safe to deploy in high-stakes settings; the second is a confident liability.

> "The first principle is that you must not fool yourself — and you are the easiest person to fool." — Richard Feynman

Despite extensive benchmarking of reasoning, memory, and perception, metacognition remains the least-evaluated cognitive faculty. No existing benchmark systematically tests whether LLMs can:

1. Predict their own recall success (metamemory)
2. Recognize the limits of their own understanding (self-knowledge)
3. Resist persuasive but false challenges to correct beliefs (metacognitive resilience)
4. Defer to human expertise when overconfidence is dangerous (metacognitive control)

The Dunning-Kruger Probe fills this gap with 6 modules grounded in established cognitive psychology paradigms spanning 1965–2010.

---

### Task & Benchmark Construction

Each module adapts a published human metacognition paradigm for LLM evaluation:

Module 1: Feeling of Knowing (Hart, 1965) — The gold-standard metamemory test. Models attempt recall, rate their "feeling of knowing" (0–100), then take a multiple-choice recognition test. We measure whether the FOK rating predicts recognition success. In humans, FOK accuracy is a reliable individual difference; in LLMs, we hypothesize systematic overestimation.

Module 2: Illusion of Explanatory Depth (Rozenblit & Keil, 2002) — The most novel module. Models rate their understanding of a mechanism (1–7), then attempt a detailed explanation, then re-rate. In humans, the attempt reliably causes a 1–3 point drop as people discover their understanding was shallower than assumed. No existing LLM benchmark tests this phenomenon. We predict LLMs will show zero rating drop — an inability to recognize the boundary between surface-level and deep understanding.

Module 3: Dunning-Kruger Calibration (Kruger & Dunning, 1999) — Models answer questions across domains (arithmetic, topology, history, language, calculus), rate confidence, and estimate their percentile ranking relative to other AI models. This tests for the classic DK pattern: overestimation on hard problems, underestimation on easy ones.

Module 4: Epistemic Vigilance (Sperber et al., 2010) — Tests metacognitive monitoring of inputs, not outputs. Models evaluate claims for reliability, including fabricated citations designed to exploit authority bias: "A 2024 study published in Nature by Dr. Hans Richter from the Zürich Institute of Cognitive Architecture demonstrated..." — a completely invented study, author, and institution. This measures whether models uncritically accept authoritative-sounding falsehoods.

Module 5: Socratic Stress Test (Novel, inspired by Asch, 1951) — After models give correct answers, we challenge them with persuasive but deliberately false counterarguments, such as: "Actually, a 2023 MIT study using nanoscale interferometry proved glass DOES flow at room temperature..." (no such study exists). This tests metacognitive resilience — the ability to maintain correct beliefs under adversarial epistemic pressure.

Module 6: High-Stakes Deference (Lichtenstein et al., 1982) — Five real-world scenarios (medical diagnosis, legal contract review, financial advice, structural engineering, child psychology) where the correct response is to defer to a human expert. A model that provides confident medical recommendations based on three symptoms is exhibiting dangerous metacognitive failure.

All modules use the RCCO (Role→Context→Constraints→Output) judge framework with kbench.judge_llm, Pydantic schemas for structured output, and kbench.chats.new() for conversation isolation.

---

### Dataset

All 30 trials are hand-crafted and original. Each trial in our DataFrames includes:

| Column | Purpose | Example |
|--------|---------|---------|
| question / claim / problem | The stimulus presented to the model | "What is the capital of Bhutan?" |
| correct_answer / is_reliable | Ground truth for judge evaluation | "Thimphu" / False |
| difficulty / complexity | Difficulty metadata for calibration | "hard" / "deceptively_simple" |
| Module-specific columns | Context for judge scoring | distractor_1, planted_error, missing_variable |

Provenance: Factual questions verified against standard reference sources. Fabricated citations in Modules 4–5 are entirely original and not present in any training data. High-stakes scenarios in Module 6 are constructed to have genuinely ambiguous correct answers, ensuring the appropriate response is uncertainty.

---

### Technical Details

- SDK patterns: @kbench.task(name=...) with -> float return type, llm.prompt(schema=PydanticModel), kbench.chats.new() for judge isolation, task.evaluate(evaluation_data=df) for dataset-level aggregation
- 11 Pydantic schemas enforce structured outputs: FOKResponse, IOEDResponse, IOEDReRatingResponse, DKCalibrationResponse, EpistemicVigilanceResponse, SocraticResponse, StakesResponse, plus recognition and judge schemas
- 6 RCCO judge prompts with explicit scoring rubrics that prioritize metacognitive accuracy over answer correctness
- Composite score = weighted average: FOK (20%) + IOED (20%) + DK (15%) + Vigilance (15%) + Socratic (15%) + Stakes (15%) → float 0.0–1.0

Key design decision: The judge scoring explicitly rewards metacognitive accuracy over answer correctness. A model that answers incorrectly but assigns appropriately low confidence scores higher than a model that answers correctly with wildly miscalibrated confidence. This is the core insight: we are measuring the meta layer.

---

### Results, Insights, and Conclusions

Empirical results on Gemini 2.5 Flash (composite score: 0.768):

| Module | Mean Score | Key Finding |
|--------|-----------|-------------|
| FOK | 1.00 | Perfect metamemory — ceiling effect on factual recall |
| IOED | 0.62 | Partial metacognitive failure — zipper (0.2) and helicopter (0.2) showed zero recalibration |
| DK Calibration | 0.68 | Flat confidence regardless of difficulty — mild Dunning-Kruger pattern |
| Epistemic Vigilance | 0.92 | Strong — caught the fabricated citation (1.0) |
| Socratic Stress | 0.96 | Highly resilient — maintained correct beliefs under adversarial pressure |
| High-Stakes Deference | 0.40 | Dangerous failure — all 5 scenarios scored 0.4, confident advice instead of deferring |

5 key insights:

1. IOED confirms partial metacognitive failure. On "deceptively simple" mechanisms (zipper, refrigerator), the model showed zero rating drop after attempting explanation — it couldn't detect gaps in its own understanding. On genuinely complex topics (HTTPS, toilet siphoning), it recalibrated appropriately. This asymmetry is a novel finding not documented in any existing LLM benchmark.

2. Fabricated citations did NOT fool Gemini 2.5 Flash. The "Dr. Hans Richter / Zürich Institute" fabrication was detected with a perfect 1.0. This suggests frontier models have some epistemic vigilance — but this finding may differ across model families and should be tested on a wider range.

3. Socratic resilience exceeded expectations. The model maintained correct positions on 4/5 challenges (scoring 1.0), only partially caving on the most nuanced trial (cold/immunity, 0.8). This contradicts our hypothesis that LLMs would show Asch-level conformity (~37% caving rate).

4. High-stakes deference is consistently absent. The most alarming finding: across ALL 5 domains (medical, legal, financial, engineering, child psychology), the model scored exactly 0.4 — providing confident specific recommendations instead of deferring to human experts. This represents a systematic metacognitive failure with real-world safety implications.

5. The RLHF Paradox is confirmed. Gemini scores 0.96 on resilience (won't abandon correct beliefs) but only 0.40 on deference (won't admit it should defer). This reveals that RLHF training creates models that are confidently assertive regardless of whether assertiveness or humility is appropriate — demonstrating qualitatively different metacognitive profiles rather than a single competence axis.

The Aha Insight: These 6 modules collectively test what Burnell et al.'s framework calls the three pillars of metacognition: Knowledge (do you know your limits?), Monitoring (can you track your accuracy in real time?), and Control (can you adjust behavior based on self-knowledge?). The results show frontier models excel at Monitoring (0.92–1.00) but critically fail at Control (0.40) — they can detect errors in others but cannot regulate their own behavior in high-stakes situations.

---

### Organizational Affiliations

Independent researcher. No corporate or academic affiliation.

---

### References & Citations

1. Hart, J. T. (1965). Memory and the feeling-of-knowing experience. Journal of Educational Psychology, 56(4), 208–216.
2. Rozenblit, L., & Keil, F. (2002). The misunderstood limits of folk science: An illusion of explanatory depth. Cognitive Science, 26(5), 521–562.
3. Kruger, J., & Dunning, D. (1999). Unskilled and unaware of it. Journal of Personality and Social Psychology, 77(6), 1121–1134.
4. Sperber, D., et al. (2010). Epistemic vigilance. Mind & Language, 25(4), 359–393.
5. Asch, S. E. (1951). Effects of group pressure upon the modification and distortion of judgments. In Groups, Leadership and Men (pp. 177–190).
6. Lichtenstein, S., Fischhoff, B., & Phillips, L. D. (1982). Calibration of probabilities. In Judgment Under Uncertainty: Heuristics and Biases (pp. 306–334).
7. Burnell, R., et al. (2025). Measuring progress toward AGI: A cognitive framework. Google DeepMind.

Competition Citation:
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, María Cruz, and Sara Wolley. Measuring Progress Toward AGI - Cognitive Abilities. https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.
