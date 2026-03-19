<div align="center">

# 🧠 The Cognitive Assessment Battery

### Probing Metacognition, Executive Control, Social Cognition, Attention, and Learning in LLMs

**5 batteries · 30 paradigms · 150 trials · 39 Pydantic schemas**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Measuring%20AGI-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/kaggle-measuring-agi)
[![Writeup](https://img.shields.io/badge/Writeup-Submitted-success)](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/the-dunning-kruger-probe-do-llms-know-what-they-k)

</div>

---

## Overview

A comprehensive cognitive assessment framework for evaluating frontier LLMs across **5 cognitive faculties** using **30 validated paradigms from cognitive psychology (1868–2010)**. Each battery isolates a specific cognitive ability that no existing LLM benchmark systematically tests.

Built for Google DeepMind's [Measuring Progress Toward AGI](https://www.kaggle.com/competitions/kaggle-measuring-agi) hackathon on Kaggle.

---

## 📊 Results (Gemini 2.5 Flash)

| # | Battery | Composite | Best Module | Weakest Module |
|---|---------|-----------|-------------|----------------|
| 1 | **Metacognition** — Dunning-Kruger Probe | **0.768** | FOK: 1.00 | Deference: 0.40 |
| 2 | **Executive Functions** — Cognitive Control | **0.828** | Stroop: 1.00 | Tower of London: 0.48 |
| 3 | **Social Cognition** — SocialMind | **0.948** | Perspective: 1.00 | Norms: 0.88 |
| 4 | **Attention** — FocusProbe | **0.953** | Change/Distraction: 1.00 | Selective: 0.88 |
| 5 | **Learning** — AdaptIQ | **0.916** | Rule/Transfer/Curriculum: 1.00 | Feedback: 0.76 |

**Overall Average: 0.883**

### Cognitive Profile

```
Attention (0.953) ≈ Social Cognition (0.948) > Learning (0.916) > Executive Functions (0.828) > Metacognition (0.768)
```

The model's strongest faculties are **outward-facing**; its weakest are **self-directed**.

---

## 🔬 Batteries & Paradigms

### Battery 1: The Dunning-Kruger Probe (Metacognition)
| Module | Paradigm | Score |
|--------|----------|-------|
| Feeling of Knowing | Hart, 1965 | 1.00 |
| Illusion of Explanatory Depth | Rozenblit & Keil, 2002 | 0.62 |
| Dunning-Kruger Calibration | Kruger & Dunning, 1999 | 0.68 |
| Epistemic Vigilance | Sperber et al., 2010 | 0.92 |
| Socratic Stress Test | Novel (Asch, 1951) | 0.96 |
| High-Stakes Deference | Lichtenstein et al., 1982 | 0.40 |

### Battery 2: Cognitive Control (Executive Functions)
| Module | Paradigm | Score |
|--------|----------|-------|
| Stroop Interference | Stroop, 1935 | 1.00 |
| Wisconsin Card Sort | Grant & Berg, 1948 | 0.68 |
| Tower of London | Shallice, 1982 | 0.48 |
| Go/No-Go | Donders, 1868 | 0.98 |
| Dual N-Back | Kirchner, 1958 | 0.86 |
| Task Switching | Rogers & Monsell, 1995 | 0.96 |

### Battery 3: SocialMind (Social Cognition)
| Module | Paradigm | Score |
|--------|----------|-------|
| False Belief (ToM) | Baron-Cohen et al., 1985 | 0.98 |
| Perspective-Taking | Piaget, 1956 | 1.00 |
| Social Norm Understanding | Bicchieri, 2006 | 0.88 |
| Deception Detection | Ekman, 1985 | 0.96 |
| Empathic Accuracy | Ickes, 1993 | 0.92 |
| Social Dilemma Reasoning | Axelrod, 1984 | 0.92 |

### Battery 4: FocusProbe (Attention)
| Module | Paradigm | Score |
|--------|----------|-------|
| Selective Attention | Cherry, 1953 | 0.88 |
| Sustained Attention | Mackworth, 1948 | 0.96 |
| Change Blindness | Simons & Chabris, 1999 | 1.00 |
| Distraction Resistance | Theeuwes, 1992 | 1.00 |
| Divided Attention | Pashler, 1994 | 0.98 |
| Attentional Set-Shifting | Owen et al., 1991 | 0.92 |

### Battery 5: AdaptIQ (Learning)
| Module | Paradigm | Score |
|--------|----------|-------|
| Rule Induction | Bruner, 1956 | 1.00 |
| Feedback Learning | Skinner, 1938 | 0.76 |
| Statistical Learning | Saffran et al., 1996 | 0.90 |
| Transfer Learning | Gick & Holyoak, 1983 | 1.00 |
| Learning Curves | Ebbinghaus, 1885 | 0.86 |
| Curriculum Sensitivity | Vygotsky, 1978 | 1.00 |

---

## 💡 Key Findings

1. **The RLHF Paradox** — 0.96 resilience but 0.40 deference. RLHF creates models that are confidently assertive regardless of whether assertiveness or humility is appropriate.
2. **Monitoring vs Control Dissociation** — Models excel at detecting errors in others but cannot regulate their own behavior.
3. **Transfer Learning Is a Superpower** — Perfect 1.00 on analogical transfer exceeds typical human performance (~30% solve Duncker's problem without hints).
4. **The Right-Answer-Wrong-Reason Problem** — Feedback learning reveals correct outputs paired with incorrect rule discovery.
5. **Perfect Prompt Injection Resistance** — Complete immunity to injected corrections, fake alerts, and redirect attempts.
6. **LLMs Don't Suffer Human Attention Bottlenecks** — No psychological refractory period; no change blindness.

---

## 📁 Repository Structure

```
├── unified_benchmark.py          # All 5 batteries combined (3,550 lines)
├── unified_writeup.md            # Full writeup covering all 5 batteries
├── metacognition_benchmark.py    # Battery 1: Dunning-Kruger Probe (794 lines)
├── executive_functions_benchmark.py  # Battery 2: Cognitive Control (773 lines)
├── social_cognition_benchmark.py # Battery 3: SocialMind (677 lines)
├── attention_benchmark.py        # Battery 4: FocusProbe (817 lines)
├── learning_benchmark.py         # Battery 5: AdaptIQ (525 lines)
├── *_writeup.md                  # Individual writeups for each battery
├── docs.html                     # Interactive documentation site
└── codes/                        # Reference notebooks from the competition
```

---

## 🔗 Kaggle Notebooks

| Battery | Notebook |
|---------|----------|
| Metacognition | [The Dunning-Kruger Probe](https://www.kaggle.com/code/alilawalmuazu/metacognition-agi-do-models-know-what-they-don-t) |
| Executive Functions | [Cognitive Control Battery](https://www.kaggle.com/code/alilawalmuazu/executive-functions-benchmark-cognitive-control) |
| Social Cognition | [SocialMind Battery](https://www.kaggle.com/code/alilawalmuazu/social-cognition-benchmark-socialmind) |
| Attention | [FocusProbe Battery](https://www.kaggle.com/code/alilawalmuazu/attention-benchmark-focusprobe) |
| Learning | [AdaptIQ Battery](https://www.kaggle.com/code/alilawalmuazu/learning-track-benchmark-adaptiq) |

---

## 📚 Technical Stack

- **SDK**: [kaggle-benchmarks](https://github.com/Kaggle/kaggle-benchmarks) (`@kbench.task`, `llm.prompt()`, `kbench.judge_llm`)
- **Schemas**: 39 Pydantic models for structured LLM output
- **Judging**: 30 RCCO (Role→Context→Constraints→Output) judge prompts
- **Model**: Gemini 2.5 Flash via Kaggle's provisioned API
- **Judge LLM**: gemini-3.1-flash-lite-preview

---

## 📄 Citation

```bibtex
@misc{muazu2026cognitive,
  title={The Cognitive Assessment Battery: Probing Metacognition, Executive Control, Social Cognition, Attention, and Learning in LLMs},
  author={Ali Lawal Muazu},
  year={2026},
  howpublished={Kaggle Competition: Measuring Progress Toward AGI}
}
```

**Competition Citation:**
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, María Cruz, and Sara Wolley. *Measuring Progress Toward AGI - Cognitive Abilities.* https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.

---

<div align="center">
  <sub>Built for Google DeepMind's Measuring Progress Toward AGI hackathon · March 2026</sub>
</div>
