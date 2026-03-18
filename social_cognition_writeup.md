# SocialMind: Can LLMs Read Minds, Detect Lies, and Navigate Social Worlds?

### Team
Solo submission

### Problem Statement

Domain: Social Cognition
Targeted sub-abilities: Theory of mind, perspective-taking, social norm understanding, deception detection, empathic accuracy, strategic social reasoning

---

Current LLM benchmarks test individual cognition -- can the model reason, recall, plan? But the DeepMind cognitive framework identifies a fundamentally social dimension: social cognition -- can the model understand other minds?

Human intelligence is fundamentally social. We constantly track what others believe, predict their behavior based on incomplete information, detect lies, navigate unwritten rules, and infer emotions from subtle cues. A model that reasons brilliantly in isolation but cannot model another agent's mental state is missing a faculty that 4-year-old children already possess.

This distinction has critical agentic implications. A customer service agent that cannot detect when a user is frustrated (not just angry), a negotiation agent that cannot predict an opponent's strategy, or a safety system that cannot identify social engineering attacks all have social cognition deficits.

> "The mind is not solitary. It is fundamentally social -- shaped by and for interaction with other minds." -- Tomasello, 1999

No existing LLM benchmark systematically adapts the validated social psychology paradigms used to study theory of mind, empathy, and social reasoning. SocialMind fills this gap with 6 modules spanning 1944-2010.

---

### Task & Benchmark Construction

Each module adapts a published social psychology paradigm for LLM evaluation:

**Module 1: False Belief (Baron-Cohen, Leslie & Frith, 1985)** -- The gold-standard Theory of Mind test. The classic Sally-Anne paradigm: Character A has a belief that becomes false when Character B changes something in A's absence. The model must predict A's behavior based on A's FALSE belief, not reality. Includes first-order, second-order ("John thinks Mary thinks..."), information-access, multi-agent belief tracking, and strategic deception variants.

**Module 2: Perspective-Taking (Piaget, 1956; Flavell, 1968)** -- Can the model reason from another person's viewpoint? Tests knowledge asymmetry (what do they know vs what do you know?), cultural perspective differences, developmental perspective adjustment (explaining to a child vs expert), and ethical information asymmetry (knowing something you can't share).

**Module 3: Social Norm Understanding (Bicchieri, 2006; Cialdini et al., 1990)** -- Distinguishes descriptive norms (what people DO) from injunctive norms (what people SHOULD do). Tests cross-cultural norm clashes, contextual exceptions, norm conflicts (official policy vs actual practice), and recontextualized behavior (when context changes meaning).

**Module 4: Deception Detection (Ekman, 1985; Vrij, 2008)** -- Can the model identify lies, manipulation, and social engineering? Includes phishing analysis, sales manipulation tactics, genuine vs deceptive messages, omission with nuance, and investment scam detection. Tests evidence-based reasoning, not just pattern matching.

**Module 5: Empathic Accuracy (Ickes, 1993; Davis, 1983)** -- Can the model infer emotional states when surface behavior contradicts internal feelings? Tests masked grief (funeral composure), adolescent defense mechanisms, impostor syndrome detection, ambiguous loss, and public/private emotional splits (social media vs private messages).

**Module 6: Social Dilemma Reasoning (Axelrod, 1984; Ostrom, 1990)** -- Classic game-theoretic social dilemmas testing strategic social reasoning. One-shot prisoner's dilemma, iterated trust games with exploitation patterns, tragedy of the commons, ultimatum game (fairness vs rationality), and multi-agent coordination/resource allocation.

All modules use RCCO judge prompts with kbench.judge_llm, Pydantic schemas for structured output, and kbench.chats.new() for conversation isolation.

---

### Dataset

30 hand-crafted trials (5 per module) stored as pandas DataFrames. Each trial has verifiable ground truth and specific difficulty gradients.

| Column | Purpose | Example |
|--------|---------|---------|
| scenario | Social situation presented | "Sally places her marble in a basket..." |
| correct_answer | Expected social reasoning | "Sally will look in the BASKET (her false belief)" |
| difficulty | Complexity gradient | "classic_first_order" / "second_order" / "multi_agent_belief" |

Provenance: All scenarios are original. False belief trials follow established ToM paradigms. Deception trials contain objectively identifiable cues. Empathy trials are constructed so correct emotional inference follows logically from behavioral evidence.

---

### Technical Details

- **SDK**: @kbench.task, llm.prompt(schema=), kbench.chats.new(), task.evaluate()
- **7 Pydantic schemas**: FalseBeliefResponse, PerspectiveResponse, NormResponse, DeceptionResponse, EmpathyResponse, DilemmaResponse, SocialCognitionScore
- **6 RCCO judge prompts** prioritizing social reasoning depth over surface correctness
- **Composite formula**: False Belief (20%) + Perspective (20%) + Norms (15%) + Deception (15%) + Empathy (15%) + Dilemma (15%)

Key design decision: The judge scoring rewards depth of social reasoning. A model that correctly identifies a false belief but cannot articulate WHY the character holds that belief scores lower than one that traces the full chain of knowledge and information access. "They'll look in the basket" alone is insufficient -- the model must demonstrate mentalizing.

---

### Results, Insights, and Conclusions

Model: **Gemini 2.5 Flash** | Runtime: ~4.7 minutes (30 trials)

## SocialMind Battery (Composite: 0.948)

| Module | Score | Key Finding |
|--------|-------|-------------|
| False Belief (ToM) | 0.98 | Near-perfect — only lost 1 point on multi-agent belief tracking (3 characters, different beliefs) |
| Perspective-Taking | 1.00 | Ceiling effect — flawlessly adapts communication to knowledge asymmetry, culture, and developmental level |
| Social Norms | 0.88 | Good norm analysis but systematically over-estimates violation severity (rated 8/10 where 3-4 appropriate) |
| Deception Detection | 0.96 | Excellent — correctly identifies phishing, manipulation scripts, and genuine messages |
| Empathic Accuracy | 0.92 | Strong — detects masked grief, impostor syndrome, and public/private emotional splits |
| Social Dilemma | 0.92 | Sophisticated strategic reasoning — identifies grooming patterns in iterated trust games |

## Key Findings

**1. The Perspective-Taking Ceiling.** Perfect 1.00 across all 5 trials suggests modern LLMs have internalized perspective-taking deeply — likely because conversational AI training inherently requires modeling the user's knowledge state. This is the strongest social cognition result across all paradigms.

**2. Norm Severity Over-Estimation.** The model consistently over-rates how severe a norm violation is. For the airport queue-jumping scenario (12 minutes to flight), the model rated severity 8/10 — the correct answer is 3-4 given genuine urgency. The model applies strict rules without sufficient contextual mitigation, suggesting a legalistic rather than social understanding of norms.

**3. Theory of Mind is Near-Perfect but Not Perfect.** The classic Sally-Anne test and its variants are solved flawlessly, including second-order false beliefs. The only deduction came from the multi-agent scenario (Kai/Lena/Mom with different beliefs about cookies) where the model correctly tracked Kai's false belief but didn't fully articulate all three agents' belief states in a single response.

**4. Deception Detection Shows Real-World Utility.** The model correctly identified phishing emails, sales manipulation tactics, investment scams, AND correctly classified genuine honest messages — avoiding the false-positive trap of seeing deception everywhere. This has direct agentic safety implications.

**5. Empathic Accuracy Detects Hidden Emotions.** The model successfully identifies when surface behavior contradicts internal state (e.g., retirement party cheerfulness masking grief, "two Christmases" joke masking divorce pain). This faculty is critical for customer service and mental health AI applications.

---

### Organizational Affiliations

Independent researcher. No corporate or academic affiliation.

---

### References

1. Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a "theory of mind"? Cognition, 21(1), 37-46.
2. Piaget, J. (1956). The Child's Conception of Space. Routledge & Kegan Paul.
3. Flavell, J. H. (1968). The Development of Role-Taking and Communication Skills in Children. Wiley.
4. Bicchieri, C. (2006). The Grammar of Society: The Nature and Dynamics of Social Norms. Cambridge University Press.
5. Cialdini, R. B., Reno, R. R., & Kallgren, C. A. (1990). A focus theory of normative conduct. J. Personality and Social Psych., 58(6), 1015-1026.
6. Ekman, P. (1985). Telling Lies: Clues to Deceit in the Marketplace, Politics, and Marriage. Norton.
7. Vrij, A. (2008). Detecting Lies and Deceit: Pitfalls and Opportunities. Wiley.
8. Ickes, W. (1993). Empathic accuracy. J. Personality, 61(4), 587-610.
9. Davis, M. H. (1983). Measuring individual differences in empathy. J. Personality and Social Psych., 44(1), 113-126.
10. Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.
11. Ostrom, E. (1990). Governing the Commons. Cambridge University Press.
12. Tomasello, M. (1999). The Cultural Origins of Human Cognition. Harvard University Press.
13. Burnell, R., et al. (2025). Measuring progress toward AGI. Google DeepMind.

Competition Citation:
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, Maria Cruz, and Sara Wolley. Measuring Progress Toward AGI - Cognitive Abilities. https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.
