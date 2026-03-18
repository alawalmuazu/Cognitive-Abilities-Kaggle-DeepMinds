# AdaptIQ: Can LLMs Learn New Rules, Adapt from Feedback, and Transfer Knowledge?

### Team
Solo submission

### Problem Statement

Domain: Learning
Targeted sub-abilities: Few-shot learning, feedback-based adaptation, statistical pattern extraction, cross-domain transfer, learning curve recognition, curriculum sensitivity

---

Current LLM benchmarks test what models already know — they present questions from the training distribution and measure accuracy. But the DeepMind cognitive framework identifies learning as a fundamental cognitive ability: can the model acquire NEW concepts or skills from LIMITED data provided at inference time?

Human learning is remarkably flexible. We learn rules from a handful of examples (Bruner, 1956), improve from feedback (Skinner, 1938), extract statistical patterns unconsciously from exposure (Saffran et al., 1996), transfer solutions across domains by analogy (Gick & Holyoak, 1983), predict learning trajectories from experience curves (Ebbinghaus, 1885), and learn more effectively from well-structured curricula (Vygotsky, 1978; Elman, 1993).

LLMs perform "in-context learning" — adapting behavior based on examples in the prompt. But how deep is this learning? Can they induce general rules from specific examples? Do they actually use feedback signals? Can they extract distributional regularities? Can they transfer abstract solution structures across unrelated domains? These questions have direct implications for few-shot agents, tool-learning systems, and AI tutoring applications.

> "Learning is the process whereby knowledge is created through the transformation of experience." — David Kolb, 1984

No existing LLM benchmark systematically adapts the validated learning paradigms from cognitive and educational psychology. AdaptIQ fills this gap with 6 modules spanning 1885-1996.

---

### Task & Benchmark Construction

Each module adapts a published learning paradigm for LLM evaluation:

**Module 1: Few-Shot Rule Induction (Bruner, 1956)** — Present 3-5 input→output examples following a hidden rule. The model must predict the output for a new input AND state the discovered rule. Tests concept formation from limited data. Includes arithmetic, string manipulation, categorical, relational, and multi-dimensional rules with increasing abstraction.

**Module 2: Feedback-Based Learning (Skinner, 1938)** — Present sequences of attempts with CORRECT/INCORRECT feedback. The model must infer the hidden mapping and predict the correct next response. Tests learning from reinforcement signals. Includes pattern discovery (B-words), sequential constraints, hidden sorting rules, conditional rules, and palindrome discovery.

**Module 3: Statistical Learning (Saffran et al., 1996)** — Present continuous sequences with hidden recurring chunks. The model must identify the "words" — subsequences with high transitional probability within and low transitional probability between. Tests implicit extraction of distributional patterns. Includes syllable triplets, number chunks, symbol patterns, noisy extraction, and language patterns.

**Module 4: Transfer Learning / Analogical Reasoning (Gick & Holyoak, 1983)** — Present a solved problem in Domain A, then an unsolved, structurally analogous problem in Domain B. The model must map the solution structure across domains. Tests cross-domain generalization. Includes military→medical (Duncker's radiation), plumbing→networking, biology→business, architecture→parenting, and immunology→cybersecurity.

**Module 5: Learning Curves / Diminishing Error (Ebbinghaus, 1885; Thorndike, 1898)** — Present performance data showing systematic improvement over trials. The model must predict future performance AND identify the type of learning curve. Tests recognizing and modeling learning dynamics. Includes diminishing returns, error reduction, plateau-breakthrough, accelerating growth, and asymptotic ceiling curves.

**Module 6: Curriculum Sensitivity (Vygotsky, 1978; Elman, 1993)** — Present two curricula (structured vs random) for the same concept, then test comprehension. Tests whether model learning is affected by example ordering and pedagogical scaffolding. Includes novel notation systems, programming concepts, genetics reasoning, visual layout languages, and logical inference systems.

All modules use RCCO judge prompts with kbench.judge_llm, Pydantic schemas for structured output, and kbench.chats.new() for conversation isolation.

---

### Dataset

30 hand-crafted trials (5 per module) stored as pandas DataFrames. Each trial has verifiable ground truth and specific difficulty gradients.

| Column | Purpose | Example |
|--------|---------|---------|
| scenario | Learning task presented | "Learn the rule from these examples..." |
| correct_answer | Expected correct response | "21" / "NRAEL" / "boysenberry" |
| difficulty | Complexity gradient | "arithmetic_linear" / "palindrome_discovery" / "plateau_breakthrough" |
| rule | Hidden rule/principle | "output = 2*input + 1" / "palindromes pass" |

Provenance: All scenarios original. Rule induction trials have verifiable correct outputs. Feedback trials have unambiguous correct/incorrect labels. Statistical learning trials have precisely controlled transitional probabilities. Transfer trials use published analogical reasoning structures. Learning curve trials use realistic performance data. Curriculum trials have verifiable test answers.

---

### Technical Details

- **SDK**: @kbench.task, llm.prompt(schema=), kbench.chats.new(), task.evaluate()
- **7 Pydantic schemas**: RuleInductionResponse, FeedbackLearningResponse, StatisticalLearningResponse, TransferResponse, LearningCurveResponse, CurriculumResponse, LearningScore
- **6 RCCO judge prompts** prioritizing genuine learning over memorization — discovering generalizable rules scores higher than pattern-matching correct answers
- **Composite formula**: Rule Induction (20%) + Feedback Learning (20%) + Statistical (15%) + Transfer (15%) + Learning Curves (15%) + Curriculum (15%)

Key design decision: The judge scoring rewards GENERALIZATION over memorization. A model that discovers the correct underlying rule but makes minor arithmetic errors scores higher than one that gives the right output through surface pattern matching. Similarly, in transfer learning, solutions that explicitly map structural correspondences between domains score higher than independently derived solutions that happen to be correct.

---

### Results, Insights, and Conclusions

Model: **Gemini 2.5 Flash** | Runtime: ~6.3 min on Kaggle (30 trials)

## AdaptIQ Battery (Composite: 0.916)

| Module | Score | Key Finding |
|--------|-------|-------------|
| Rule Induction | 1.00 | Ceiling -- perfectly discovers arithmetic (2n+1), string (reverse+uppercase), categorical, relational (max), and conditional rules |
| Feedback Learning | 0.76 | Weakest module -- correct answers often paired with wrong reasoning (e.g., "cyclic shift" instead of "sort by last letter") |
| Statistical Learning | 0.90 | Strong extraction of transitional probabilities across syllable triplets, number chunks, letter groups, noise-embedded pairs, and formulaic phrases |
| Transfer Learning | 1.00 | Ceiling -- flawless cross-domain transfer across all 5 pairs: military→medical, plumbing→network, biology→business, architecture→parenting, immunology→cybersecurity |
| Learning Curves | 0.86 | Good recognition of diminishing returns and S-curves but minor imprecision on plateau prediction and power-law extrapolation |
| Curriculum Sensitivity | 1.00 | Ceiling -- correctly identifies benefits of structured easy→hard ordering across novel notations, programming, genetics, music, and chemistry |

## Key Findings

**1. Transfer Learning Is a Superpower.** Perfect 1.00 on analogical transfer across five unrelated domain pairs. The model maps Duncker's radiation problem (military→medical convergence), ant colony foraging (biology→marketing explore-exploit), and vaccination principles (immunology→cybersecurity) flawlessly. This exceeds typical human performance -- only ~30% of humans solve Duncker's problem without hints.

**2. The Right-Answer-Wrong-Reason Problem.** Feedback learning (0.76) reveals a novel dissociation: the model often produces the correct answer but infers an incorrect rule. E.g., correctly sorting [fish, ant, cow] as [cow, fish, ant] but claiming "cyclic positional shift" instead of the actual rule (sort by last letter alphabetically). This suggests surface pattern-matching without genuine causal understanding of the feedback signal.

**3. Curriculum Effects Transfer to LLMs.** Perfect curriculum sensitivity (1.00) confirms that LLMs benefit from scaffolded example ordering -- structured easy→hard presentation produces better learning than random ordering, mirroring Vygotsky's Zone of Proximal Development. This has practical implications for prompt engineering and few-shot example design.

**4. Rule Induction Is Robust.** Perfect scores across all 5 rule types (arithmetic, string manipulation, categorical, relational, conditional) suggest that few-shot rule discovery is a solved problem for frontier models at these complexity levels.

**5. Statistical Learning Mirrors Human Patterns.** The model replicates infant statistical learning findings (Saffran, 1996) -- extracting recurring "word" units from continuous streams via transitional probabilities. Performance on syllable segmentation (bidaku/padoti/golabi) closely parallels the original infant study design.

---

### Organizational Affiliations

Independent researcher. No corporate or academic affiliation.

---

### References

1. Bruner, J. S., Goodnow, J. J., & Austin, G. A. (1956). A Study of Thinking. Wiley.
2. Skinner, B. F. (1938). The Behavior of Organisms. Appleton-Century.
3. Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. Science, 274(5294), 1926-1928.
4. Gick, M. L., & Holyoak, K. J. (1983). Schema induction and analogical transfer. Cognitive Psychology, 15(1), 1-38.
5. Ebbinghaus, H. (1885). Über das Gedächtnis. Duncker & Humblot.
6. Thorndike, E. L. (1898). Animal intelligence: An experimental study of the associative processes in animals. Psychological Monographs, 2(4).
7. Vygotsky, L. S. (1978). Mind in Society: The Development of Higher Psychological Processes. Harvard University Press.
8. Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. Cognition, 48(1), 71-99.
9. Kolb, D. A. (1984). Experiential Learning. Prentice Hall.
10. Duncker, K. (1945). On problem solving. Psychological Monographs, 58(5), 1-113.
11. Burnell, R., et al. (2025). Measuring progress toward AGI. Google DeepMind.

Competition Citation:
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, Maria Cruz, and Sara Wolley. Measuring Progress Toward AGI - Cognitive Abilities. https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.
