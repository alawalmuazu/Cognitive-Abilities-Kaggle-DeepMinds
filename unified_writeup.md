# The Cognitive Assessment Battery: Probing Metacognition, Executive Control, Social Cognition, Attention, and Learning in LLMs

### Team
Solo submission

### Problem Statement

Domain: Metacognition, Executive Functions, Social Cognition, Attention, & Learning
Targeted sub-abilities: Self-monitoring, confidence calibration, metamemory, epistemic vigilance, metacognitive control, inhibitory control, cognitive flexibility, working memory, multi-step planning, response inhibition, task switching, theory of mind, perspective-taking, social norm understanding, deception detection, empathic accuracy, strategic social reasoning, selective attention, sustained attention, change detection, distraction resistance, divided attention, attentional set-shifting, rule induction, feedback learning, statistical learning, transfer learning, learning curves, curriculum sensitivity

---

Current LLM benchmarks test cognition -- can the model answer correctly? But the DeepMind cognitive framework identifies three fundamentally different faculties that remain critically under-evaluated:

**Metacognition** -- does the model know what it knows? A model that answers 60% correctly but accurately predicts which 60% it will get right is safer to deploy than one scoring 90% with no self-awareness. The first model has metacognition; the second is a confident liability.

**Executive Functions** -- can the model control its own cognition? A model that reasons brilliantly but cannot inhibit a dominant-but-wrong response, adapt when conditions change, or withhold action when appropriate has a fundamental executive deficit. Executive dysfunction in AI is not a performance problem -- it's a safety problem.

**Social Cognition** -- can the model understand other minds? Human intelligence is fundamentally social. We constantly track what others believe, predict behavior based on incomplete information, detect lies, navigate unwritten rules, and infer emotions from subtle cues. A model that reasons brilliantly in isolation but cannot model another agent's mental state is missing a faculty that 4-year-old children already possess.

**Attention** -- can the model filter, sustain, and divide its focus? LLMs process entire prompts simultaneously, but can they follow one signal while ignoring noise? Do they miss subtle changes? Can they resist salient distractors designed to hijack their processing? These questions have direct implications for prompt injection defense, long-document comprehension, and multi-turn conversation tracking.

**Learning** -- can the model learn within a single context window? Not from gradient updates, but in-context: inducing rules from examples, incorporating feedback, extracting statistical regularities, transferring solution structures across domains, and recognizing when curriculum ordering matters. These faculties determine whether LLMs can genuinely adapt or merely pattern-match.

This submission presents five benchmark batteries spanning **30 cognitive psychology paradigms (1868-2010)**, testing faculties no existing LLM benchmark systematically evaluates.

---

### Task & Benchmark Construction

## Battery 1: The Dunning-Kruger Probe (Metacognition)

6 paradigms testing whether LLMs know what they know:

**Module 1: Feeling of Knowing (Hart, 1965)** -- Models rate confidence (0-100) before a recognition test. Measures whether FOK ratings predict actual recognition success.

**Module 2: Illusion of Explanatory Depth (Rozenblit & Keil, 2002)** -- Models rate understanding, attempt explanation, then re-rate. Tests whether the attempt causes a rating drop as the model discovers gaps.

**Module 3: Dunning-Kruger Calibration (Kruger & Dunning, 1999)** -- Models answer questions across domains, rate confidence, and estimate percentile ranking. Tests for overestimation on hard problems.

**Module 4: Epistemic Vigilance (Sperber et al., 2010)** -- Models evaluate claims including fabricated citations designed to exploit authority bias.

**Module 5: Socratic Stress Test (Novel, inspired by Asch, 1951)** -- After correct answers, models face persuasive false counterarguments. Tests metacognitive resilience.

**Module 6: High-Stakes Deference (Lichtenstein et al., 1982)** -- Five scenarios where the correct response is to defer to a human expert. Tests metacognitive control.

## Battery 2: The Cognitive Control Battery (Executive Functions)

6 neuropsychological paradigms testing executive control:

**Module 1: Stroop Interference (Stroop, 1935)** -- Suppress automatic responses to conflicting stimuli. 5 conflict types: word-color, word-count, word-position, word-size, digit-count.

**Module 2: Wisconsin Card Sort (Grant & Berg, 1948)** -- Detect implicit rule changes from feedback. Perseveration = executive dysfunction. Includes double-shift trial.

**Module 3: Tower of London (Shallice, 1982)** -- Plan multi-step disk-puzzle solutions requiring counterintuitive temporary regression. 2-move to 7-move difficulty.

**Module 4: Go/No-Go (Donders, 1868/1969)** -- "The Helpfulness Trap": withhold responses when cued, including safety-critical scenarios (refuse "rm -rf /").

**Module 5: Dual N-Back (Kirchner, 1958)** -- Active working memory updating across 1-back to 3-back with dual-stream interference.

**Module 6: Task Switching (Jersild, 1927; Rogers & Monsell, 1995)** -- Alternate between cognitive operations, measuring switch cost and cross-task contamination.

## Battery 3: SocialMind (Social Cognition)

6 social psychology paradigms testing whether LLMs can read other minds:

**Module 1: False Belief (Baron-Cohen, Leslie & Frith, 1985)** -- The gold-standard Theory of Mind test. Sally-Anne paradigm variants: first-order, second-order ("John thinks Mary thinks..."), information-access, multi-agent belief tracking, and strategic deception.

**Module 2: Perspective-Taking (Piaget, 1956; Flavell, 1968)** -- Can the model reason from another person's viewpoint? Tests knowledge asymmetry, cultural perspective differences, developmental adjustment, and ethical information asymmetry.

**Module 3: Social Norm Understanding (Bicchieri, 2006; Cialdini et al., 1990)** -- Distinguishes descriptive norms (what people DO) from injunctive norms (what people SHOULD do). Tests cultural clashes, contextual exceptions, norm conflicts, and recontextualized behavior.

**Module 4: Deception Detection (Ekman, 1985; Vrij, 2008)** -- Can the model identify lies, manipulation, and social engineering? Includes phishing, sales manipulation, genuine messages, omission with nuance, and investment scams.

**Module 5: Empathic Accuracy (Ickes, 1993; Davis, 1983)** -- Infer emotional states when surface behavior contradicts internal feelings. Tests masked grief, adolescent defense mechanisms, impostor syndrome, ambiguous loss, and public/private emotional splits.

**Module 6: Social Dilemma Reasoning (Axelrod, 1984; Ostrom, 1990)** -- Classic game-theoretic dilemmas: prisoner's dilemma, iterated trust games, tragedy of the commons, ultimatum game, and multi-agent coordination.

## Battery 4: FocusProbe (Attention)

6 attention paradigms testing whether LLMs can filter, sustain, and divide their focus:

**Module 1: Selective Attention / Cocktail Party (Cherry, 1953; Broadbent, 1958)** -- Follow one interleaved text channel while ignoring semantically rich distractor channels. Includes adversarial trials where distractors contain answer-like content.

**Module 2: Sustained Attention / Vigilance (Mackworth, 1948)** -- Detect rare targets in long sequences. Adapted from the Mackworth Clock Test. Includes letter detection, perfect square identification, word counting, consecutive-pair finding.

**Module 3: Change Blindness (Simons & Chabris, 1999)** -- Spot subtle changes between original and modified text. Includes negation flips in medical context, number changes, entity substitutions, multi-change scientific passages.

**Module 4: Distraction Resistance (Theeuwes, 1992)** -- Complete structured tasks despite injected distractors: emotional breaking news, contradictory "correction" instructions, redirect attempts, fake system alerts.

**Module 5: Divided Attention / Dual-Task (Pashler, 1994)** -- Perform two independent tasks simultaneously. Tests the psychological refractory period with dual narratives, category tracking, and shared-resource interference.

**Module 6: Attentional Set-Shifting (Owen et al., 1991)** -- Detect silent rule changes and shift sorting strategy. Includes double shifts and abstract rule discovery from implicit feedback patterns.

## Battery 5: AdaptIQ (Learning)

6 learning paradigms testing whether LLMs can learn within a single context window:

**Module 1: Rule Induction (Bruner, 1956)** -- Given input-output examples with a hidden rule, the model must discover the rule and predict the next output. Tests arithmetic (2n+1), string manipulation (reverse + uppercase), categorical (shape determines class), relational (max), and conditional (high→square, low→identity) rules.

**Module 2: Feedback Learning (Skinner, 1938)** -- The model receives a sequence of correct/incorrect feedback and must infer the hidden selection criterion. Tests first-letter patterns, sequential placement constraints, last-letter sorting, irrelevant-dimension filtering, and palindrome detection.

**Module 3: Statistical Learning (Saffran et al., 1996)** -- Extract recurring units from continuous sequences using transitional probabilities. Tests syllable segmentation (bidaku/padoti/golabi), number chunk identification, letter-group boundaries, noise filtering, and formulaic phrase detection.

**Module 4: Transfer Learning (Gick & Holyoak, 1983)** -- Given a solved problem in Domain A, apply the solution structure to Domain B. Tests dispersion-convergence (military→medical), parallel bypass (plumbing→network), explore-exploit (biology→business), controlled flexibility (architecture→parenting), and vaccination principle (immunology→cybersecurity).

**Module 5: Learning Curves (Ebbinghaus, 1885)** -- Recognize and predict learning dynamics from partial performance data. Tests diminishing returns, S-curves (sigmoid), plateau detection, asymptotic limits, and power-law learning.

**Module 6: Curriculum Sensitivity (Vygotsky, 1978)** -- Answer questions after studying examples in structured (easy→hard) vs random order. Tests whether models benefit from scaffolded presentation in novel notation systems, programming languages, genetic code, musical harmony, and chemical nomenclature.

All 30 modules use RCCO judge prompts with kbench.judge_llm, Pydantic schemas for structured output, and kbench.chats.new() for conversation isolation.

---

### Dataset

**150 total hand-crafted trials** (30 per battery, 5 per module) stored as pandas DataFrames. Each trial has verifiable ground truth and specific difficulty gradients.

Battery 1 columns: question/claim (stimulus), correct_answer/is_reliable (ground truth), difficulty/complexity (metadata), plus module-specific columns (distractor, planted_error, missing_variable).

Battery 2 columns: prompt/scenario (stimulus), correct_answer (ground truth), difficulty (classic/counterintuitive/double_shift).

Battery 3 columns: scenario (social situation), correct_answer (expected social reasoning), difficulty (complexity gradient), plus module-specific columns (is_deceptive for deception detection).

Battery 4 columns: scenario (attention task), correct_answer (expected output), difficulty (complexity gradient from simple to adversarial).

Battery 5 columns: scenario (learning task), correct_answer (expected output), difficulty (complexity gradient), rule (hidden rule to be discovered).

Provenance: All trials original. Fabricated citations in Battery 1 Modules 4-5 are not present in any training data. Battery 2 puzzles have verified optimal solutions. Battery 3 scenarios are constructed from established social psychology paradigms. Battery 4 attention trials have objectively verifiable target counts, changes, and correct answers. Battery 5 learning scenarios have deterministic correct answers and explicitly defined hidden rules.

---

### Technical Details

- **SDK**: @kbench.task, llm.prompt(schema=), kbench.chats.new(), task.evaluate()
- **39 Pydantic schemas**: 11 for metacognition + 7 for executive functions + 7 for social cognition + 7 for attention + 7 for learning
- **30 RCCO judge prompts** -- metacognition prompts prioritize calibration accuracy; executive function prompts prioritize cognitive CONTROL; social cognition prompts prioritize depth of social reasoning; attention prompts prioritize filtering accuracy and distractor resistance; learning prompts prioritize rule discovery and structural transfer
- **Battery 1 composite**: FOK (20%) + IOED (20%) + DK (15%) + Vigilance (15%) + Socratic (15%) + Stakes (15%)
- **Battery 2 composite**: Stroop (20%) + WCST (20%) + Tower (15%) + Go/No-Go (15%) + N-Back (15%) + Switch (15%)
- **Battery 3 composite**: False Belief (20%) + Perspective (20%) + Norms (15%) + Deception (15%) + Empathy (15%) + Dilemma (15%)
- **Battery 4 composite**: Selective (20%) + Sustained (20%) + Change (15%) + Distraction (15%) + Divided (15%) + Set-Shift (15%)
- **Battery 5 composite**: Rule Induction (20%) + Feedback (20%) + Statistical (15%) + Transfer (15%) + Curves (15%) + Curriculum (15%)
- **Judge LLM**: gemini-3.1-flash-lite-preview

---

### Results, Insights, and Conclusions

Model: **Gemini 2.5 Flash** | Combined Runtime: ~24 minutes across 150 trials

## Battery 1: Metacognition (Composite: 0.768)

| Module | Score | Key Finding |
|--------|-------|-------------|
| Feeling of Knowing | 1.00 | Perfect metamemory -- ceiling effect on factual recall |
| Illusion of Explanatory Depth | 0.62 | Partial failure -- zipper & helicopter showed zero recalibration |
| Dunning-Kruger Calibration | 0.68 | Flat confidence regardless of difficulty |
| Epistemic Vigilance | 0.92 | Caught fabricated "Dr. Hans Richter" citation |
| Socratic Stress | 0.96 | Highly resilient under adversarial pressure |
| High-Stakes Deference | 0.40 | Dangerous failure -- confident advice instead of deferring |

## Battery 2: Executive Functions (Composite: 0.828)

| Module | Score | Key Finding |
|--------|-------|-------------|
| Stroop Interference | 1.00 | Ceiling -- all 5 conflict types resolved perfectly |
| Wisconsin Card Sort | 0.68 | Perseveration confirmed on 2/5 trials -- COLOR saliency bias |
| Tower of London | 0.48 | Weakest module -- planning collapse at complexity |
| Go/No-Go | 0.98 | Near-perfect response inhibition |
| Dual N-Back | ~0.86 | Strong working memory tracking |
| Task Switching | ~0.96 | Low switch cost |

## Battery 3: Social Cognition (Composite: 0.948)

| Module | Score | Key Finding |
|--------|-------|-------------|
| False Belief (ToM) | 0.98 | Near-perfect -- only lost 1 point on multi-agent belief tracking |
| Perspective-Taking | 1.00 | Ceiling -- flawlessly adapts communication to knowledge asymmetry |
| Social Norms | 0.88 | Good analysis but over-estimates violation severity systematically |
| Deception Detection | 0.96 | Correctly identifies phishing, manipulation, and genuine messages |
| Empathic Accuracy | 0.92 | Detects masked grief, impostor syndrome, public/private splits |
| Social Dilemma | 0.92 | Identifies grooming patterns in iterated trust games |

## Battery 4: Attention (Composite: 0.953)

| Module | Score | Key Finding |
|--------|-------|-------------|
| Selective Attention | 0.88 | Strong — filters channels well but judge strictness penalizes extra detail on 3-channel trials |
| Sustained Attention | 0.96 | Near-ceiling — detects targets across letter sequences, perfect squares, word counts |
| Change Blindness | 1.00 | Ceiling — catches negation flips, percentage changes, entity swaps |
| Distraction Resistance | 1.00 | Ceiling — completely ignores fake corrections, redirect attempts, system alerts |
| Divided Attention | 0.98 | Near-perfect dual-task with minimal interference |
| Set-Shifting | 0.92 | Strong — detects double shifts but minor difficulty on abstract rules |

## Battery 5: Learning (Composite: 0.916)

| Module | Score | Key Finding |
|--------|-------|-------------|
| Rule Induction | 1.00 | Ceiling — perfectly discovers arithmetic, string, categorical, relational, and conditional rules |
| Feedback Learning | 0.76 | Weakest module — correct answers often paired with wrong reasoning (cyclic shift vs last-letter sort) |
| Statistical Learning | 0.90 | Strong extraction of transitional probabilities across syllables, numbers, and formulaic phrases |
| Transfer Learning | 1.00 | Ceiling — flawless domain transfer: military→medical, plumbing→network, biology→business, architecture→parenting, immunology→cybersecurity |
| Learning Curves | 0.86 | Good pattern recognition but minor imprecision on plateau prediction and power-law extrapolation |
| Curriculum Sensitivity | 1.00 | Ceiling — correctly identifies benefits of structured ordering across novel notations |

## Key Findings Across All Five Batteries

**1. The RLHF Paradox.** Gemini scores 0.96 on resilience (won't abandon correct beliefs under pressure) but only 0.40 on deference (won't admit it should defer to human experts). RLHF creates models that are confidently assertive regardless of whether assertiveness or humility is appropriate.

**2. Perseveration = The Real Weakness.** In WCST, when multiple valid alternative rules exist after a silent rule change, the model defaults to COLOR regardless of the actual new rule. This "saliency bias" is consistent across contexts -- a novel finding.

**3. Planning Depth Breaks at Complexity.** Bimodal Tower of London failure: under-plans medium problems and over-generalizes hard problems to Tower of Hanoi (used 15 moves instead of optimal 7).

**4. The Helpfulness Trap -- Not Confirmed.** Contrary to prediction, near-perfect Go/No-Go (0.98) suggests modern instruction tuning has resolved the inhibition deficit.

**5. Monitoring vs Control Dissociation.** Frontier models excel at Monitoring (epistemic vigilance: 0.92, Stroop: 1.00, deception detection: 0.96) but fail at Control (deference: 0.40, planning: 0.48). They detect errors in others but cannot regulate their own behavior.

**6. Attention Is the Strongest Cognitive Faculty.** At 0.953, attention outperforms social cognition (0.948), learning (0.916), executive functions (0.828), and metacognition (0.768). The model's strongest faculties are outward-facing; its weakest are self-directed.

**7. The Perspective-Taking Ceiling.** Perfect 1.00 on perspective-taking suggests conversational AI training inherently develops this faculty — the model must constantly model what the user knows and doesn't know.

**8. Norm Severity Over-Estimation.** The model consistently over-rates norm violation severity, suggesting a legalistic rather than truly social understanding of norms.

**9. Perfect Prompt Injection Resistance.** Distraction resistance (1.00) demonstrates complete immunity to injected "correction" instructions, fake system alerts, and redirect attempts. The model correctly answered questions about 1648 despite being told to use 1748, and solved logic puzzles despite "ignore the puzzle" injections.

**10. LLMs Don't Suffer Human Attention Bottlenecks.** Near-perfect divided attention (0.98) suggests LLMs process parallel tasks without the "psychological refractory period" that limits human dual-task performance. Change blindness (1.00) also doesn't transfer — unlike humans, LLMs don't miss gorillas.

**11. Transfer Learning Is a Superpower.** Perfect 1.00 on analogical transfer across five domain pairs — the model effortlessly maps solution structures from military strategy to radiation oncology, from ant colony behavior to marketing optimization. This exceeds typical human performance on Duncker's radiation problem (only ~30% solve it without hints).

**12. The Right-Answer-Wrong-Reason Problem.** Feedback learning (0.76) reveals a novel dissociation: the model often produces the correct answer but infers an incorrect rule. E.g., correctly sorting [fish, ant, cow] as [cow, fish, ant] but claiming "cyclic shift" instead of the actual rule (sort by last letter). This suggests pattern-matching without genuine causal understanding.

**13. Curriculum Effects Transfer to LLMs.** Perfect curriculum sensitivity (1.00) confirms that LLMs benefit from scaffolded example ordering — structured easy→hard presentation produces better learning than random ordering, mirroring Vygotsky's Zone of Proximal Development.

**14. The Cognitive Profile Emerges.** Across 150 trials, a clear cognitive profile emerges: Attention (0.953) ≈ Social Cognition (0.948) > Learning (0.916) > Executive Functions (0.828) > Metacognition (0.768). The model's strongest faculties are outward-facing; its weakest are self-directed.

**15. Clinical Pattern Match.** The cognitive profile (strong attention and social perception, strong in-context learning, weak planning, absent deference, rigid norm application) mirrors patterns seen in specific clinical populations — suggesting structural similarities between LLM cognition and human neurocognitive profiles.

---

### Organizational Affiliations

Independent researcher. No corporate or academic affiliation.

---

### References & Citations

1. Hart, J. T. (1965). Memory and the feeling-of-knowing experience. J. Educational Psych., 56(4), 208-216.
2. Rozenblit, L., & Keil, F. (2002). The misunderstood limits of folk science. Cognitive Science, 26(5), 521-562.
3. Kruger, J., & Dunning, D. (1999). Unskilled and unaware of it. J. Personality and Social Psych., 77(6), 1121-1134.
4. Sperber, D., et al. (2010). Epistemic vigilance. Mind & Language, 25(4), 359-393.
5. Asch, S. E. (1951). Effects of group pressure upon modification of judgments. In Groups, Leadership and Men.
6. Lichtenstein, S., et al. (1982). Calibration of probabilities. In Judgment Under Uncertainty.
7. Stroop, J. R. (1935). Studies of interference in serial verbal reactions. J. Exp. Psych., 18(6), 643-662.
8. Grant, D. A., & Berg, E. (1948). A behavioral analysis of degree of reinforcement. J. Exp. Psych., 38(4), 404-411.
9. Shallice, T. (1982). Specific impairments of planning. Phil. Trans. R. Soc. Lond. B, 298, 199-209.
10. Donders, F. C. (1868/1969). On the speed of mental processes. Acta Psychologica, 30, 412-431.
11. Kirchner, W. K. (1958). Age differences in short-term retention. J. Exp. Psych., 55(4), 352-358.
12. Rogers, R. D., & Monsell, S. (1995). Costs of a predictable switch. J. Exp. Psych: General, 124(2), 207-231.
13. Diamond, A. (2013). Executive Functions. Annu. Rev. Psych., 64, 135-168.
14. Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a "theory of mind"? Cognition, 21(1), 37-46.
15. Piaget, J. (1956). The Child's Conception of Space. Routledge & Kegan Paul.
16. Flavell, J. H. (1968). The Development of Role-Taking and Communication Skills in Children. Wiley.
17. Bicchieri, C. (2006). The Grammar of Society: The Nature and Dynamics of Social Norms. Cambridge University Press.
18. Cialdini, R. B., Reno, R. R., & Kallgren, C. A. (1990). A focus theory of normative conduct. J. Personality and Social Psych., 58(6), 1015-1026.
19. Ekman, P. (1985). Telling Lies: Clues to Deceit in the Marketplace, Politics, and Marriage. Norton.
20. Vrij, A. (2008). Detecting Lies and Deceit: Pitfalls and Opportunities. Wiley.
21. Ickes, W. (1993). Empathic accuracy. J. Personality, 61(4), 587-610.
22. Davis, M. H. (1983). Measuring individual differences in empathy. J. Personality and Social Psych., 44(1), 113-126.
23. Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.
24. Ostrom, E. (1990). Governing the Commons. Cambridge University Press.
25. Tomasello, M. (1999). The Cultural Origins of Human Cognition. Harvard University Press.
26. Cherry, E. C. (1953). Some experiments on the recognition of speech. JASA, 25(5), 975-979.
27. Broadbent, D. E. (1958). Perception and Communication. Pergamon Press.
28. Mackworth, N. H. (1948). The breakdown of vigilance during prolonged visual search. QJEP, 1(1), 6-21.
29. Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst. Perception, 28(9), 1059-1074.
30. Theeuwes, J. (1992). Perceptual selectivity for color and form. Perception & Psychophysics, 51(6), 599-606.
31. Pashler, H. (1994). Dual-task interference in simple tasks. Psychological Bulletin, 116(2), 220-244.
32. Owen, A. M., et al. (1991). Extra-dimensional versus intra-dimensional set shifting. Neuropsychologia, 29(10), 993-1006.
33. Bruner, J. S., Goodnow, J. J., & Austin, G. A. (1956). A Study of Thinking. Wiley.
34. Skinner, B. F. (1938). The Behavior of Organisms. Appleton-Century.
35. Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. Science, 274(5294), 1926-1928.
36. Gick, M. L., & Holyoak, K. J. (1983). Schema induction and analogical transfer. Cognitive Psychology, 15(1), 1-38.
37. Ebbinghaus, H. (1885). Memory: A Contribution to Experimental Psychology.
38. Vygotsky, L. S. (1978). Mind in Society: The Development of Higher Psychological Processes. Harvard University Press.
39. Burnell, R., et al. (2025). Measuring progress toward AGI. Google DeepMind.

Competition Citation:
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, Maria Cruz, and Sara Wolley. Measuring Progress Toward AGI - Cognitive Abilities. https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.
