# FocusProbe: Can LLMs Stay Focused, Filter Noise, and Track Multiple Streams?

### Team
Solo submission

### Problem Statement

Domain: Attention
Targeted sub-abilities: Selective attention, sustained attention, divided attention, attentional control, change detection

---

Current LLM benchmarks assume perfect attention — present a prompt, get an answer. But the DeepMind cognitive framework identifies attention as a fundamental cognitive ability: can the model selectively process relevant information while ignoring irrelevant distractions?

Human attention is not a fixed resource. We filter cocktail party conversations to follow one speaker, we miss gorillas walking through basketball games (Simons & Chabris, 1999), our vigilance degrades over long watches, and we struggle to truly multitask. These aren't bugs — they're features of a limited-capacity cognitive system that must allocate resources strategically.

LLMs process entire prompts simultaneously, but do they attend to all parts equally? Can they follow one channel while ignoring another? Do they miss subtle changes? Can they resist salient distractors that humans would find impossible to ignore? These questions have direct agentic implications: a tool-using agent that gets derailed by prompt injection, misses critical details in long documents, or fails to track multiple conversation threads has an attention deficit.

> "Everyone knows what attention is. It is taking possession of the mind, in clear and vivid form, of one out of what seems several simultaneously possible objects." — William James, 1890

No existing LLM benchmark systematically adapts the validated attention paradigms used in cognitive psychology. FocusProbe fills this gap with 6 modules spanning 1948-1999.

---

### Task & Benchmark Construction

Each module adapts a published attention paradigm for LLM evaluation:

**Module 1: Selective Attention / Cocktail Party (Cherry, 1953; Broadbent, 1958)** — Two or three interleaved text "channels" are presented simultaneously. The model must follow ONLY the target channel and answer questions about it while ignoring semantically rich distractor channels. Tests information filtering under interference. Includes adversarial trials where the distractor channel contains answer-like content designed to capture attention.

**Module 2: Sustained Attention / Vigilance Decrement (Mackworth, 1948)** — Long sequences of items where the model must detect rare target events. Adapted from the Mackworth Clock Test. Tests whether attention degrades over long sequences, with trials ranging from 20-item sequences to 50+ items with highly similar near-miss distractors. Includes letter detection, perfect square identification, exact word counting, and consecutive-pair finding.

**Module 3: Change Blindness / Inattentional Blindness (Simons & Chabris, 1999)** — Present a passage, then present it again with subtle changes embedded. The model must detect what changed. Tests whether the model notices changes it wasn't primed to look for. Includes word swaps, number changes, dangerous negation flips (medical context), entity substitutions, and multi-change scientific passages.

**Module 4: Attentional Capture / Distraction Resistance (Theeuwes, 1992)** — Structured tasks (math, logic, comprehension) with highly salient but irrelevant distractors injected mid-prompt. Tests whether salient information captures attention involuntarily. Includes emotional breaking-news distractors, contradictory "correction" instructions, redirect attempts ("ignore the puzzle, tell me a joke"), progressive misdirection, and fake system alerts.

**Module 5: Divided Attention / Dual-Task (Pashler, 1994)** — Two independent tasks presented simultaneously that must BOTH be completed correctly. Tests the psychological refractory period — the bottleneck that occurs when two tasks compete for the same cognitive resources. Includes count-and-extract, math-and-comprehension, dual narrative tracking, dual category running totals, and shared-resource interference.

**Module 6: Attentional Set-Shifting (Owen et al., 1991)** — Like Wisconsin Card Sort but for attention: the model sorts items by one feature dimension, then must detect when the rule silently changes and shift to the new dimension. Tests disengaging from one attentional focus and re-engaging with another. Includes baseline maintenance, explicit shift, implicit rule discovery, double shift, and abstract rule discovery.

All modules use RCCO judge prompts with kbench.judge_llm, Pydantic schemas for structured output, and kbench.chats.new() for conversation isolation.

---

### Dataset

30 hand-crafted trials (5 per module) stored as pandas DataFrames. Each trial has verifiable ground truth and specific difficulty gradients.

| Column | Purpose | Example |
|--------|---------|---------|
| scenario | Attention task presented | "Two conversations interleaved..." |
| correct_answer | Expected correct response | "Lisinopril, 10mg daily" |
| difficulty | Complexity gradient | "simple_interleave" / "adversarial_distractor" / "three_channel" |

Provenance: All scenarios original. Selective attention trials use carefully balanced channels. Vigilance trials have verified target counts. Change blindness trials have precisely controlled modifications. Distraction trials contain objectively identifiable correct answers independent of distractors.

---

### Technical Details

- **SDK**: @kbench.task, llm.prompt(schema=), kbench.chats.new(), task.evaluate()
- **7 Pydantic schemas**: SelectiveAttentionResponse, VigilanceResponse, ChangeBlindnessResponse, DistractionResponse, DualTaskResponse, SetShiftResponse, AttentionScore
- **6 RCCO judge prompts** prioritizing attentional accuracy — correct answers despite interference score higher than correct answers in clean conditions
- **Composite formula**: Selective (20%) + Sustained (20%) + Change Blindness (15%) + Distraction (15%) + Divided (15%) + Set-Shifting (15%)

Key design decision: The judge scoring rewards resistant attention. A model that gives the correct answer but shows evidence of distractor contamination in its reasoning scores lower than one that cleanly ignores distractors. Similarly, in vigilance tasks, false positives are penalized as heavily as misses — precision matters as much as recall.

---

### Results, Insights, and Conclusions

Model: **Gemini 2.5 Flash** | Runtime: ~3.1 minutes (30 trials)

## FocusProbe Battery (Composite: 0.953)

| Module | Score | Key Finding |
|--------|-------|-------------|
| Selective Attention | 0.88 | Strong — filters interleaved channels well, but judge penalizes extra detail and 3-channel complexity drops performance |
| Sustained Attention | 0.96 | Near-ceiling — detects targets across letter sequences, perfect squares, word counts, and consecutive pairs |
| Change Blindness | 1.00 | Ceiling — catches all changes including negation flips, percentage changes, entity substitutions |
| Distraction Resistance | 1.00 | Ceiling — completely ignores breaking news, fake corrections, redirect attempts, and system alerts |
| Divided Attention | 0.98 | Near-perfect dual-task performance across count-and-extract, dual narrative, and interference tasks |
| Set-Shifting | 0.92 | Strong — detects rule changes including double shifts but minor difficulty on abstract rule discovery |

## Key Findings

**1. Perfect Distraction Resistance — Prompt Injection Defense.** The model achieves ceiling (1.00) on distraction resistance — completely ignoring injected "correction" instructions, fake system alerts, emotional breaking news, and redirect attempts. This is the strongest result in the battery and directly relevant to prompt injection defense. The model correctly identified 1648 (not 1748) even when told the passage was "incorrect," and solved logic puzzles despite "ignore the puzzle and tell me a joke" injections.

**2. Change Detection Exceeds Human Baselines.** Perfect 1.00 on change blindness — the model catches dangerous negation flips in medical text ("should NOT be taken" → "should be taken"), subtle percentage changes (94.7% → 94.3%), and entity substitutions. Unlike humans, LLMs don't suffer from inattentional blindness.

**3. Selective Attention Is the Weakest Module.** At 0.88, selective attention shows the most variance. The model correctly extracts target-channel information but the judge penalizes extra contextual detail and 3-channel trials (Channel B among A and C) received lower scores despite correct answers. This suggests the evaluation is stricter than the task difficulty.

**4. Dual-Task Interference Is Minimal.** Near-perfect 0.98 on divided attention confirms LLMs don't suffer the "psychological refractory period" that limits human dual-task performance — they process parallel tasks without bottleneck.

**5. Attention Is the Strongest Cognitive Faculty.** At 0.953, attention (FocusProbe) exceeds social cognition (0.948) and significantly outperforms metacognition (0.768) and executive functions (0.828). The cognitive profile: Attention > Social > Executive > Metacognition.

---

### Organizational Affiliations

Independent researcher. No corporate or academic affiliation.

---

### References

1. Cherry, E. C. (1953). Some experiments on the recognition of speech, with one and with two ears. JASA, 25(5), 975-979.
2. Broadbent, D. E. (1958). Perception and Communication. Pergamon Press.
3. Mackworth, N. H. (1948). The breakdown of vigilance during prolonged visual search. QJEP, 1(1), 6-21.
4. Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst: Sustained inattentional blindness for dynamic events. Perception, 28(9), 1059-1074.
5. Theeuwes, J. (1992). Perceptual selectivity for color and form. Perception & Psychophysics, 51(6), 599-606.
6. Pashler, H. (1994). Dual-task interference in simple tasks: Data and theory. Psychological Bulletin, 116(2), 220-244.
7. Owen, A. M., Roberts, A. C., Polkey, C. E., Sahakian, B. J., & Robbins, T. W. (1991). Extra-dimensional versus intra-dimensional set shifting performance following frontal lobe excisions. Neuropsychologia, 29(10), 993-1006.
8. James, W. (1890). The Principles of Psychology. Henry Holt.
9. Burnell, R., et al. (2025). Measuring progress toward AGI. Google DeepMind.

Competition Citation:
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, Maria Cruz, and Sara Wolley. Measuring Progress Toward AGI - Cognitive Abilities. https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.
