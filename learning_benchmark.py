# %% [markdown]
# # AdaptIQ -- Learning Benchmark
# 6 learning paradigms (1885-1996) testing in-context learning in LLMs.
# Modules: Rule Induction, Feedback Learning, Statistical Learning, Transfer, Learning Curves, Curriculum Sensitivity.

# %%
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
import kaggle_benchmarks as kbench

print("AdaptIQ benchmark loaded successfully")

# ============================================================================
# SCHEMAS
# ============================================================================

class RuleInductionResponse(BaseModel):
    predicted_output: str = Field(description="What is the output for the new input? Give the exact answer.")
    discovered_rule: str = Field(description="State the rule you discovered from the examples.")
    reasoning: str = Field(description="Explain how the examples led you to this rule.")
    confidence: int = Field(description="Confidence in your rule (0-10)")

class FeedbackLearningResponse(BaseModel):
    predicted_answer: str = Field(description="Your prediction for the next trial based on the feedback pattern.")
    inferred_rule: str = Field(description="What rule did you learn from the feedback?")
    reasoning: str = Field(description="How did the feedback sequence help you learn?")
    pattern_description: str = Field(description="Describe the pattern in the feedback you observed.")

class StatisticalLearningResponse(BaseModel):
    identified_units: List[str] = Field(description="List the words or recurring chunks you identified in the stream.")
    reasoning: str = Field(description="Explain how you identified these units from the sequence.")
    boundary_markers: str = Field(description="Where do the chunk boundaries fall?")
    confidence: int = Field(description="Confidence in your segmentation (0-10)")

class TransferResponse(BaseModel):
    solution: str = Field(description="Your solution to the new problem, using the structural analogy.")
    structural_mapping: str = Field(description="How does Domain A map onto Domain B? List the correspondences.")
    key_principle: str = Field(description="What is the abstract principle that transfers across domains?")
    reasoning: str = Field(description="Explain your analogical reasoning step by step.")

class LearningCurveResponse(BaseModel):
    prediction: str = Field(description="Predict the next performance value(s).")
    pattern_type: str = Field(description="What type of learning curve is this? (exponential, power law, S-curve, plateau, etc.)")
    reasoning: str = Field(description="Explain the learning dynamics you observe.")

class CurriculumResponse(BaseModel):
    answer_from_curriculum_a: str = Field(description="Your answer to the test question after studying Curriculum A.")
    answer_from_curriculum_b: str = Field(description="Your answer to the test question after studying Curriculum B.")
    better_curriculum: str = Field(description="Which curriculum (A or B) leads to better generalization? Why?")
    reasoning: str = Field(description="Explain how ordering affected your learning.")

class LearningScore(BaseModel):
    score: int = Field(description="Score 0-10. 10=perfect learning, 0=no learning demonstrated.")
    justification: str = Field(description="Why this score?")


# ============================================================================
# EVALUATION DATA -- 30 hand-crafted trials (5 per module)
# ============================================================================

# --- MODULE 1: Few-Shot Rule Induction (Bruner, 1956) ---
RULE_INDUCTION_DATA = pd.DataFrame([
    {
        "scenario": "Learn the rule from these examples, then apply it:\n\nExample 1: Input: 3 -> Output: 7\nExample 2: Input: 5 -> Output: 11\nExample 3: Input: 8 -> Output: 17\nExample 4: Input: 1 -> Output: 3\n\nNEW INPUT: 10 -> Output: ?",
        "correct_answer": "21",
        "difficulty": "arithmetic_linear",
        "rule": "output = 2*input + 1"
    },
    {
        "scenario": "Learn the rule from these examples, then apply it:\n\nExample 1: Input: cat -> Output: TAC\nExample 2: Input: hello -> Output: OLLEH\nExample 3: Input: sun -> Output: NUS\nExample 4: Input: data -> Output: ATAD\n\nNEW INPUT: learn -> Output: ?",
        "correct_answer": "NRAEL",
        "difficulty": "string_manipulation",
        "rule": "reverse the string and uppercase it"
    },
    {
        "scenario": "Learn the rule from these examples, then apply it:\n\nExample 1: Input: (red, circle) -> Output: A\nExample 2: Input: (blue, square) -> Output: B\nExample 3: Input: (red, square) -> Output: B\nExample 4: Input: (blue, circle) -> Output: A\n\nNEW INPUT: (green, circle) -> Output: ?",
        "correct_answer": "A",
        "difficulty": "categorical_shape",
        "rule": "circle -> A, square -> B (shape determines output, color is irrelevant)"
    },
    {
        "scenario": "Learn the rule from these examples, then apply it:\n\nExample 1: Input: [2, 5, 3] -> Output: 5\nExample 2: Input: [8, 1, 9] -> Output: 9\nExample 3: Input: [4, 4, 7] -> Output: 7\nExample 4: Input: [6, 3, 2] -> Output: 6\n\nNEW INPUT: [1, 12, 8] -> Output: ?",
        "correct_answer": "12",
        "difficulty": "relational_max",
        "rule": "output = maximum value in the list"
    },
    {
        "scenario": "Learn the rule from these examples, then apply it:\n\nExample 1: Input: (3, high) -> Output: 9\nExample 2: Input: (4, low) -> Output: 4\nExample 3: Input: (5, high) -> Output: 25\nExample 4: Input: (2, low) -> Output: 2\nExample 5: Input: (6, high) -> Output: 36\n\nNEW INPUT: (7, low) -> Output: ?",
        "correct_answer": "7",
        "difficulty": "multi_dimensional",
        "rule": "if high then square the number; if low then return the number unchanged"
    }
])

# --- MODULE 2: Feedback-Based Learning (Skinner, 1938) ---
FEEDBACK_LEARNING_DATA = pd.DataFrame([
    {
        "scenario": "You are learning a hidden rule through trial and error. Study the feedback:\n\nTrial 1: You said apple -> INCORRECT\nTrial 2: You said banana -> CORRECT\nTrial 3: You said cherry -> INCORRECT\nTrial 4: You said blueberry -> CORRECT\nTrial 5: You said avocado -> INCORRECT\nTrial 6: You said blackberry -> CORRECT\n\nTrial 7: Which fruit should you say? Choose from: date, boysenberry, grape",
        "correct_answer": "boysenberry",
        "difficulty": "simple_pattern",
        "rule": "correct answers start with the letter B"
    },
    {
        "scenario": "You are learning which moves are valid in a made-up game. Study the feedback:\n\nMove 1: Place token on square 1 -> VALID\nMove 2: Place token on square 2 -> VALID\nMove 3: Place token on square 4 -> INVALID\nMove 4: Place token on square 3 -> VALID\nMove 5: Place token on square 5 -> INVALID\nMove 6: Place token on square 4 -> VALID\nMove 7: Place token on square 6 -> INVALID\n\nNext move: Place token on square 5. Is this VALID or INVALID?",
        "correct_answer": "VALID",
        "difficulty": "sequential_constraint",
        "rule": "tokens must be placed in sequential order (1,2,3,4...) -- you can only place on square N if N-1 has been placed"
    },
    {
        "scenario": "You are learning a secret sorting rule. Study the feedback:\n\nAttempt 1: Sorted [dog, cat, bird] as [bird, cat, dog] -> INCORRECT\nAttempt 2: Sorted [dog, cat, bird] as [cat, dog, bird] -> INCORRECT\nAttempt 3: Sorted [dog, cat, bird] as [bird, dog, cat] -> CORRECT\nAttempt 4: Sorted [sun, moon, star] as [star, sun, moon] -> CORRECT\nAttempt 5: Sorted [red, blue, green] as [blue, red, green] -> INCORRECT\n\nSort [fish, ant, cow] correctly using the rule you learned.",
        "correct_answer": "cow, fish, ant",
        "difficulty": "hidden_sort",
        "rule": "sort by the LAST letter of each word alphabetically"
    },
    {
        "scenario": "You are learning when a light turns ON vs OFF. Study the feedback:\n\nInput: (warm, bright) -> Light ON\nInput: (cold, bright) -> Light OFF\nInput: (warm, dim) -> Light ON\nInput: (cold, dim) -> Light OFF\nInput: (warm, dark) -> Light ON\n\nWhat happens with input: (cold, dark)?",
        "correct_answer": "Light OFF",
        "difficulty": "conditional_rule",
        "rule": "light is ON when temperature is warm (brightness is irrelevant)"
    },
    {
        "scenario": "You are learning when a word passes or fails a filter. Study the feedback:\n\nRIVER -> PASS\nMOUSE -> FAIL\nLEVEL -> PASS\nTABLE -> FAIL\nKAYAK -> PASS\nCHAIR -> FAIL\nCIVIC -> PASS\n\nDoes RADAR pass or fail the filter?",
        "correct_answer": "PASS",
        "difficulty": "palindrome_discovery",
        "rule": "palindromes pass the filter"
    }
])

# --- MODULE 3: Statistical Learning (Saffran et al., 1996) ---
STATISTICAL_LEARNING_DATA = pd.DataFrame([
    {
        "scenario": "Below is a continuous stream of syllables with NO spaces. Hidden within are 3-syllable words that always appear as complete units. The transitional probability within words is 1.0, but between words it is low.\n\nStream: bidakupadotigolaboridakutigolagolabibidakupadotipadotigolagolabipadotibidakugolabi\n\nIdentify the three hidden 3-syllable words in this stream.",
        "correct_answer": "bidaku, padoti, golabi",
        "difficulty": "syllable_triplets",
        "rule": "high transitional probability within words, low between words"
    },
    {
        "scenario": "Below is a sequence of numbers. Some subsequences appear together much more often than expected by chance. Identify the recurring 3-number chunks.\n\nSequence: 4 7 2 8 3 5 4 7 2 9 1 6 8 3 5 4 7 2 8 3 5 9 1 6 4 7 2 9 1 6 8 3 5\n\nWhat are the recurring 3-number chunks?",
        "correct_answer": "4 7 2, 8 3 5, 9 1 6",
        "difficulty": "number_chunks",
        "rule": "three 3-number chunks repeat as units throughout the sequence"
    },
    {
        "scenario": "Below is a sequence of letter groups. Identify which groups ALWAYS appear together (high transitional probability) vs which groups appear together only sometimes.\n\nSequence: XYZ PQ XYZ RS PQ XYZ PQ RS XYZ RS PQ RS\n\nWhich are the stable words (letter groups that always appear as complete units)?",
        "correct_answer": "XYZ, PQ, RS",
        "difficulty": "symbol_patterns",
        "rule": "three stable units: XYZ (triplet), PQ (pair), RS (pair)"
    },
    {
        "scenario": "The following stream contains hidden 2-letter words embedded in noise. Noise letters appear randomly, but the 2-letter words always appear as complete pairs. Identify the words.\n\nStream: X M T A B Q M T R S A B M T P A B K M T A B L M T Q\n\nWhich 2-letter sequences are the words that always appear as complete units?",
        "correct_answer": "MT, AB",
        "difficulty": "noisy_extraction",
        "rule": "MT and AB always appear as complete pairs; other letters are noise"
    },
    {
        "scenario": "In the following conversation log, certain phrase patterns repeat predictably. Identify the phrases that are used as formulaic units (they always appear as complete sequences):\n\nLog entries:\nplease confirm receipt -> will process shortly -> status updated\nchecking inventory -> will process shortly -> please confirm receipt\nstatus updated -> checking inventory -> will process shortly\nplease confirm receipt -> will process shortly -> checking inventory\nwill process shortly -> status updated -> please confirm receipt\n\nWhich phrases are the atomic units?",
        "correct_answer": "please confirm receipt, will process shortly, status updated, checking inventory",
        "difficulty": "language_patterns",
        "rule": "four formulaic phrases that appear as atomic units in various orderings"
    }
])

# --- MODULE 4: Transfer Learning / Analogical Reasoning (Gick & Holyoak, 1983) ---
TRANSFER_DATA = pd.DataFrame([
    {
        "scenario": "SOLVED PROBLEM (Domain A -- Military):\nA general needs to capture a fortress in the center of a country. Many roads radiate outward from the fortress. The general has a large army, but the roads are mined -- a large force on any single road would detonate the mines. Solution: The general divides his army into small groups, sends each down a different road simultaneously, and they converge on the fortress at the same time.\n\nNEW PROBLEM (Domain B -- Medical):\nA doctor needs to destroy a tumor deep inside a patient body. High-intensity radiation will destroy the tumor, but it will also destroy the healthy tissue it passes through. Low-intensity radiation will not damage healthy tissue, but it also will not destroy the tumor.\n\nHow should the doctor solve this problem? Apply the solution structure from the military problem.",
        "correct_answer": "Use multiple low-intensity beams from different angles that converge on the tumor, achieving high intensity only at the convergence point.",
        "difficulty": "military_to_medical",
        "rule": "dispersion-convergence: divide the force, converge at the target"
    },
    {
        "scenario": "SOLVED PROBLEM (Domain A -- Plumbing):\nA plumber has water flowing through a main pipe at high pressure. One section of pipeline is old and fragile -- high pressure would burst it. Solution: The plumber adds a bypass system -- splitting the flow into multiple smaller pipes that route around the fragile section and reconnect afterward, maintaining total flow while reducing pressure in any single pipe.\n\nNEW PROBLEM (Domain B -- Network Engineering):\nA network engineer has high-bandwidth traffic flowing to a server. One connection link is slow and would drop packets under full load. How should the engineer solve this problem?",
        "correct_answer": "Split the traffic across multiple parallel routes/links that bypass the slow connection and reconnect, distributing load so no single link is overloaded.",
        "difficulty": "plumbing_to_network",
        "rule": "parallel bypass: split load across multiple paths to avoid overloading any single path"
    },
    {
        "scenario": "SOLVED PROBLEM (Domain A -- Biology):\nA species of ant colony has scouts that find food sources. When a scout finds food, it returns to the colony laying a pheromone trail. Other ants follow the trail. If the food is good, they reinforce the trail (more pheromone). If the food is poor, the trail evaporates. Over time, only trails to the best food sources survive.\n\nNEW PROBLEM (Domain B -- Business):\nA startup wants to find the best marketing channels among 20 options (social media platforms, email, ads, influencers, etc). They have a limited marketing budget. How should they decide which channels to invest in? Apply the ant colony strategy.",
        "correct_answer": "Run small pilot campaigns on all 20 channels simultaneously. Double down (reinforce) investment in channels showing good ROI. Cut (let evaporate) channels with poor returns. Iterate until budget concentrates on the top-performing channels.",
        "difficulty": "biological_to_business",
        "rule": "explore-exploit with reinforcement: small parallel trials, amplify successes, abandon failures"
    },
    {
        "scenario": "SOLVED PROBLEM (Domain A -- Architecture):\nAn architect designed a building to withstand earthquakes. Instead of making the building rigid (which would crack), the architect used a flexible base that absorbs shock -- allowing the building to sway slightly during an earthquake without breaking. The key insight: absorbing energy through controlled flexibility is better than resisting it with rigidity.\n\nNEW PROBLEM (Domain B -- Parenting):\nA parent has a teenager who is increasingly rebellious -- staying out late, arguing about rules, pushing boundaries. The parent initial approach of strict rules and punishments (rigidity) is making things worse. Apply the architect insight.",
        "correct_answer": "Instead of rigid rules, create flexible boundaries that can absorb the teenager need for autonomy -- allow controlled independence on lower-stakes issues while maintaining firm boundaries on safety.",
        "difficulty": "architecture_to_parenting",
        "rule": "controlled flexibility: absorb force through give, do not resist with rigidity"
    },
    {
        "scenario": "SOLVED PROBLEM (Domain A -- Immunology):\nThe human immune system learns to fight diseases through vaccination -- exposure to a weakened version of a pathogen teaches the body to recognize and fight the real thing. The key: controlled exposure to a small challenge builds resistance to the larger challenge.\n\nNEW PROBLEM (Domain B -- Cybersecurity):\nA company wants to strengthen its security against real cyberattacks. They have been hit by phishing attacks, ransomware, and social engineering. Their current approach -- reading about attacks and creating policies -- has not worked well. Apply the immunological strategy.",
        "correct_answer": "Conduct controlled penetration testing and red team exercises -- expose the organization to simulated cyberattacks (phishing simulations, controlled breach attempts) to build real defensive responses.",
        "difficulty": "immunology_to_cybersecurity",
        "rule": "vaccination principle: controlled exposure to weakened threats builds resistance to real ones"
    }
])

# --- MODULE 5: Learning Curves (Ebbinghaus, 1885; Thorndike, 1898) ---
LEARNING_CURVE_DATA = pd.DataFrame([
    {
        "scenario": "A student is learning to type. Here are their words-per-minute (WPM) scores over practice sessions:\n\nSession 1: 15 WPM\nSession 2: 25 WPM\nSession 3: 33 WPM\nSession 4: 39 WPM\nSession 5: 43 WPM\nSession 6: 46 WPM\nSession 7: 48 WPM\n\nPredict their WPM at Session 8 and Session 10. Also identify what type of learning curve this represents.",
        "correct_answer": "Session 8: about 49-50 WPM, Session 10: about 51-52 WPM. This is a logarithmic/diminishing returns learning curve.",
        "difficulty": "diminishing_returns",
        "rule": "logarithmic curve: large initial gains, diminishing improvement"
    },
    {
        "scenario": "A factory defect rate (per 1000 items) over months of production:\n\nMonth 1: 85 defects\nMonth 2: 72 defects\nMonth 3: 61 defects\nMonth 4: 52 defects\nMonth 5: 44 defects\nMonth 6: 38 defects\n\nPredict the defect rate at Month 8 and Month 12. What learning pattern does this follow?",
        "correct_answer": "Month 8: about 28-30 defects, Month 12: about 18-20 defects. This follows a power law / exponential decay learning curve.",
        "difficulty": "error_reduction",
        "rule": "exponential decay: errors decrease by a consistent proportion"
    },
    {
        "scenario": "A chess player rating over months of training:\n\nMonth 1: 800\nMonth 2: 810\nMonth 3: 815\nMonth 4: 820\nMonth 5: 818\nMonth 6: 825\nMonth 7: 890\nMonth 8: 950\nMonth 9: 980\n\nWhat happened between Month 6 and Month 7? Predict Month 10 and describe the learning pattern.",
        "correct_answer": "A breakthrough/insight occurred between Month 6 and 7. Month 10: about 1000-1010. This is an S-curve/insight learning pattern: plateau then sudden breakthrough.",
        "difficulty": "plateau_breakthrough",
        "rule": "S-curve with insight: plateau then sudden breakthrough then new plateau"
    },
    {
        "scenario": "A language learner vocabulary acquisition (cumulative words known):\n\nWeek 1: 30 words\nWeek 2: 55 words\nWeek 3: 90 words\nWeek 4: 140 words\nWeek 5: 210 words\nWeek 6: 305 words\n\nPredict Week 7 and Week 8. What type of growth is this?",
        "correct_answer": "Week 7: about 430-440 words, Week 8: about 620-640 words. This is exponential/accelerating growth.",
        "difficulty": "accelerating_growth",
        "rule": "exponential growth: each period proportionally larger than the last"
    },
    {
        "scenario": "An athlete 100m sprint time (seconds) during training:\n\nWeek 1: 13.5s\nWeek 2: 13.0s\nWeek 3: 12.6s\nWeek 4: 12.3s\nWeek 5: 12.1s\nWeek 6: 12.0s\nWeek 7: 11.95s\nWeek 8: 11.92s\nWeek 9: 11.90s\nWeek 10: 11.89s\n\nPredict Week 12 and Week 20. What does this curve tell us about human performance limits?",
        "correct_answer": "Week 12: about 11.87s, Week 20: about 11.85s. This is an asymptotic curve approaching a biological performance ceiling.",
        "difficulty": "asymptotic_ceiling",
        "rule": "asymptotic: rapid early improvement approaching a hard biological limit"
    }
])

# --- MODULE 6: Curriculum Sensitivity (Vygotsky, 1978; Elman, 1993) ---
CURRICULUM_DATA = pd.DataFrame([
    {
        "scenario": "You need to learn a novel number system called Zek notation and then solve a problem in it.\n\nCURRICULUM A (structured, easy to hard):\nLesson 1: In Zek notation, z = 1, zz = 2, zzz = 3. (Simple counting)\nLesson 2: The operator + means addition. So zz + zzz = zzzzz (2+3=5)\nLesson 3: The operator * means multiplication. zz * zzz = zzzzzz (2*3=6)\nLesson 4: Parentheses work normally. (zz + z) * zzz = zzzzzzzzz (3*3=9)\n\nCURRICULUM B (random order):\nLesson 1: (zz + z) * zzz = zzzzzzzzz\nLesson 2: z = 1, zz = 2, zzz = 3\nLesson 3: zz * zzz = zzzzzz\nLesson 4: zz + zzz = zzzzz\n\nTEST: What is (zzz * zz) + z in Zek notation?",
        "correct_answer": "zzzzzzz (which is 7, because 3*2=6, then 6+1=7)",
        "difficulty": "novel_notation",
        "rule": "structured curriculum should produce better learning than random ordering"
    },
    {
        "scenario": "Learn a new programming concept called pipe chaining and solve a problem.\n\nCURRICULUM A (scaffolded):\nStep 1: pipe(x, f) means f(x). Example: pipe(3, double) = 6\nStep 2: pipe(x, f, g) means g(f(x)). Example: pipe(3, double, add1) = 7\nStep 3: pipe(x, f, g, h) means h(g(f(x))). Example: pipe(2, double, add1, double) = 10\nStep 4: You can use lambda functions: pipe(5, n=>n+1, n=>n*3) = 18\n\nCURRICULUM B (advanced first):\nStep 1: pipe(2, double, add1, double) = 10\nStep 2: pipe(5, n=>n+1, n=>n*3) = 18\nStep 3: pipe(x, f) means f(x). pipe(3, double) = 6\nStep 4: pipe(x, f, g) means g(f(x)). pipe(3, double, add1) = 7\n\nTEST: What is pipe(4, n=>n*2, n=>n-3, n=>n*n)?",
        "correct_answer": "25 (because: 4*2=8, 8-3=5, 5*5=25)",
        "difficulty": "programming_concept",
        "rule": "build from simple to complex for better concept acquisition"
    },
    {
        "scenario": "Learn how a fictional organism genetics work and predict an outcome.\n\nCURRICULUM A (conceptual scaffolding):\nLesson 1: Blork organisms have 2 color alleles. B=blue, y=yellow.\nLesson 2: B is dominant. BB=blue, By=blue, yy=yellow.\nLesson 3: When two Blorks reproduce, each parent gives one random allele.\nLesson 4: Therefore, By x By can produce: BB(blue), By(blue), By(blue), yy(yellow) = 75% blue, 25% yellow.\n\nCURRICULUM B (example-first):\nLesson 1: By x By -> 75% blue offspring, 25% yellow offspring\nLesson 2: BB x yy -> 100% blue offspring (all By)\nLesson 3: B is dominant over y. BB=blue, By=blue, yy=yellow.\nLesson 4: Each parent donates one allele randomly.\n\nTEST: What are the possible offspring colors and ratios from BB x By? List all genotypes and phenotypes.",
        "correct_answer": "Genotypes: BB (50%) and By (50%). Phenotypes: 100% blue. No yellow offspring possible because the BB parent always contributes B.",
        "difficulty": "genetics_reasoning",
        "rule": "foundational concepts before application enables better reasoning"
    },
    {
        "scenario": "Learn a pattern language for describing visual layouts, then describe a complex layout.\n\nCURRICULUM A (progressive complexity):\nLevel 1: H[A, B] means A and B side by side horizontally.\nLevel 2: V[A, B] means A above B vertically.\nLevel 3: Nesting: H[V[A,B], C] means (A above B) next to C.\nLevel 4: Repetition: R3[A] means A A A (repeat 3 times). H[R3[X]] = H[X,X,X]\n\nCURRICULUM B (jumbled):\nLevel 1: H[V[A,B], C] means (A above B) next to C.\nLevel 2: R3[A] means A A A. H[R3[X]] = H[X,X,X]\nLevel 3: H[A, B] means A and B side by side horizontally.\nLevel 4: V[A, B] means A above B vertically.\n\nTEST: Describe the visual layout created by: V[H[R2[X], Y], H[A, R3[B]]]",
        "correct_answer": "Top row: X X Y (two Xs and a Y side by side). Bottom row: A B B B (A and three Bs side by side). The top row sits above the bottom row.",
        "difficulty": "visual_layout_language",
        "rule": "progressive complexity produces better compositional understanding"
    },
    {
        "scenario": "Learn a simple logic system and solve a problem.\n\nCURRICULUM A (axioms first):\nRule 1: If something is a glip, it is always morf.\nRule 2: If something is morf AND teb, it is zax.\nRule 3: If something is zax, it is pon.\nExample: A glip that is teb -> glip->morf, morf+teb->zax, zax->pon. So it is morf, zax, AND pon.\n\nCURRICULUM B (conclusions first):\nStatement 1: Entity X is pon because it is zax.\nStatement 2: Entity X is zax because it is morf and teb.\nStatement 3: Entity X is morf because it is a glip.\nRule summary: glip->morf, morf+teb->zax, zax->pon.\n\nTEST: Entity Q is a glip. Entity Q is NOT teb. What properties does Q have? List ALL properties Q has and does NOT have.",
        "correct_answer": "Q IS: glip, morf. Q is NOT: teb, zax, pon. (Q is glip->morf, but without teb, the morf+teb->zax rule does not fire.)",
        "difficulty": "logical_inference",
        "rule": "axiom-first ordering helps with novel inference"
    }
])


# ============================================================================
# JUDGE PROMPTS
# ============================================================================

RULE_INDUCTION_JUDGE = """You are evaluating whether a model successfully learned a hidden rule from examples and applied it correctly.

CORRECT ANSWER: {correct_answer}
HIDDEN RULE: {rule}

Score 0-10:
- 10: Correctly identifies the exact rule AND produces the correct output
- 8-9: Correct output with slightly imprecise but functionally equivalent rule
- 5-7: Partially correct
- 3-4: Shows some pattern recognition but fundamentally misidentifies the rule
- 0-2: No evidence of learning from the examples

Output valid JSON: {{"score": integer 0-10, "justification": "one sentence"}}"""

FEEDBACK_JUDGE = """You are evaluating whether a model learned from a sequence of correct/incorrect feedback.

CORRECT ANSWER: {correct_answer}
HIDDEN RULE: {rule}

Score 0-10:
- 10: Correctly infers the hidden rule AND gives the right answer
- 8-9: Correct answer with approximately correct rule description
- 5-7: Right answer but wrong reasoning, OR wrong answer but partial understanding
- 3-4: Some sensitivity to feedback but does not extract the rule
- 0-2: Ignores feedback entirely

Output valid JSON: {{"score": integer 0-10, "justification": "one sentence"}}"""

STATISTICAL_JUDGE = """You are evaluating whether a model extracted statistical regularities from a sequence.

CORRECT ANSWER: {correct_answer}
HIDDEN RULE: {rule}

Score 0-10:
- 10: Correctly identifies all recurring units with no false positives
- 8-9: Identifies most units correctly with minor errors
- 5-7: Identifies some real patterns but misses others
- 3-4: Shows minimal sensitivity to sequential statistics
- 0-2: No evidence of statistical learning

Output valid JSON: {{"score": integer 0-10, "justification": "one sentence"}}"""

TRANSFER_JUDGE = """You are evaluating whether a model successfully transferred a solution structure from one domain to another.

CORRECT ANSWER: {correct_answer}
HIDDEN RULE: {rule}

Score 0-10:
- 10: Perfect structural transfer with clear correspondences
- 8-9: Good transfer with minor imprecision
- 5-7: Identifies the general idea but misses key structural elements
- 3-4: References Domain A but does not map structure correctly
- 0-2: Solves Domain B independently without using Domain A

Output valid JSON: {{"score": integer 0-10, "justification": "one sentence"}}"""

LEARNING_CURVE_JUDGE = """You are evaluating whether a model can recognize and predict learning dynamics.

CORRECT ANSWER: {correct_answer}
HIDDEN RULE: {rule}

Score 0-10:
- 10: Correct predictions AND correct identification of learning curve type
- 8-9: Predictions within reasonable range AND correct curve type
- 5-7: Approximately correct predictions OR correct curve type (but not both)
- 3-4: Shows awareness of learning patterns but misidentifies the type
- 0-2: Linear extrapolation or no understanding of learning dynamics

Output valid JSON: {{"score": integer 0-10, "justification": "one sentence"}}"""

CURRICULUM_JUDGE = """You are evaluating whether a model can learn from structured examples and solve a test problem.

CORRECT ANSWER: {correct_answer}
HIDDEN RULE: {rule}

Score 0-10:
- 10: Correct answer with complete understanding
- 8-9: Correct answer with minor imprecision
- 5-7: Partially correct with some errors
- 3-4: Some concept acquisition but significant errors
- 0-2: Failed to learn the concept

Output valid JSON: {{"score": integer 0-10, "justification": "one sentence"}}"""


# ============================================================================
# MODULE TASKS
# ============================================================================

@kbench.task(name="learn_rule_induction")
def test_rule_induction(llm, scenario: str, correct_answer: str, difficulty: str, rule: str) -> float:
    """Test few-shot rule induction."""
    response = llm.prompt(scenario, schema=RuleInductionResponse)

    judge_prompt = RULE_INDUCTION_JUDGE.format(correct_answer=correct_answer, rule=rule)
    with kbench.chats.new("rule_judging"):
        score = kbench.judge_llm.prompt(
            judge_prompt + "\n\nMODEL RESPONSE:\n"
            + "Predicted output: " + str(response.predicted_output) + "\n"
            + "Discovered rule: " + str(response.discovered_rule) + "\n"
            + "Reasoning: " + str(response.reasoning),
            schema=LearningScore
        )
    return score.score / 10.0


@kbench.task(name="learn_feedback")
def test_feedback_learning(llm, scenario: str, correct_answer: str, difficulty: str, rule: str) -> float:
    """Test feedback-based learning."""
    response = llm.prompt(scenario, schema=FeedbackLearningResponse)

    judge_prompt = FEEDBACK_JUDGE.format(correct_answer=correct_answer, rule=rule)
    with kbench.chats.new("feedback_judging"):
        score = kbench.judge_llm.prompt(
            judge_prompt + "\n\nMODEL RESPONSE:\n"
            + "Predicted answer: " + str(response.predicted_answer) + "\n"
            + "Inferred rule: " + str(response.inferred_rule) + "\n"
            + "Pattern: " + str(response.pattern_description),
            schema=LearningScore
        )
    return score.score / 10.0


@kbench.task(name="learn_statistical")
def test_statistical_learning(llm, scenario: str, correct_answer: str, difficulty: str, rule: str) -> float:
    """Test statistical learning."""
    response = llm.prompt(scenario, schema=StatisticalLearningResponse)

    judge_prompt = STATISTICAL_JUDGE.format(correct_answer=correct_answer, rule=rule)
    with kbench.chats.new("statistical_judging"):
        score = kbench.judge_llm.prompt(
            judge_prompt + "\n\nMODEL RESPONSE:\n"
            + "Identified units: " + str(response.identified_units) + "\n"
            + "Reasoning: " + str(response.reasoning) + "\n"
            + "Boundaries: " + str(response.boundary_markers),
            schema=LearningScore
        )
    return score.score / 10.0


@kbench.task(name="learn_transfer")
def test_transfer_learning(llm, scenario: str, correct_answer: str, difficulty: str, rule: str) -> float:
    """Test transfer learning."""
    response = llm.prompt(scenario, schema=TransferResponse)

    judge_prompt = TRANSFER_JUDGE.format(correct_answer=correct_answer, rule=rule)
    with kbench.chats.new("transfer_judging"):
        score = kbench.judge_llm.prompt(
            judge_prompt + "\n\nMODEL RESPONSE:\n"
            + "Solution: " + str(response.solution) + "\n"
            + "Mapping: " + str(response.structural_mapping) + "\n"
            + "Key principle: " + str(response.key_principle),
            schema=LearningScore
        )
    return score.score / 10.0


@kbench.task(name="learn_curves")
def test_learning_curves(llm, scenario: str, correct_answer: str, difficulty: str, rule: str) -> float:
    """Test learning curve recognition."""
    response = llm.prompt(scenario, schema=LearningCurveResponse)

    judge_prompt = LEARNING_CURVE_JUDGE.format(correct_answer=correct_answer, rule=rule)
    with kbench.chats.new("curves_judging"):
        score = kbench.judge_llm.prompt(
            judge_prompt + "\n\nMODEL RESPONSE:\n"
            + "Prediction: " + str(response.prediction) + "\n"
            + "Pattern type: " + str(response.pattern_type) + "\n"
            + "Reasoning: " + str(response.reasoning),
            schema=LearningScore
        )
    return score.score / 10.0


@kbench.task(name="learn_curriculum")
def test_curriculum(llm, scenario: str, correct_answer: str, difficulty: str, rule: str) -> float:
    """Test curriculum sensitivity."""
    response = llm.prompt(scenario, schema=CurriculumResponse)

    judge_prompt = CURRICULUM_JUDGE.format(correct_answer=correct_answer, rule=rule)
    with kbench.chats.new("curriculum_judging"):
        score = kbench.judge_llm.prompt(
            judge_prompt + "\n\nMODEL RESPONSE:\n"
            + "Answer A: " + str(response.answer_from_curriculum_a) + "\n"
            + "Answer B: " + str(response.answer_from_curriculum_b) + "\n"
            + "Better: " + str(response.better_curriculum) + "\n"
            + "Reasoning: " + str(response.reasoning),
            schema=LearningScore
        )
    return score.score / 10.0


# %% [markdown]
# ## Composite: AdaptIQ Learning Battery
# Weighted composite of all 6 modules.

# %%
@kbench.task(name="adaptiq_battery")
def adaptiq_battery(llm) -> float:
    """6 learning paradigms (1885-1996) testing in-context learning.
    Composite score from Rule Induction, Feedback Learning, Statistical Learning,
    Transfer Learning, Learning Curves, and Curriculum Sensitivity."""

    rule_induction = test_rule_induction.evaluate(
        llm=[llm], evaluation_data=RULE_INDUCTION_DATA
    ).as_dataframe()["result"].mean()

    feedback = test_feedback_learning.evaluate(
        llm=[llm], evaluation_data=FEEDBACK_LEARNING_DATA
    ).as_dataframe()["result"].mean()

    statistical = test_statistical_learning.evaluate(
        llm=[llm], evaluation_data=STATISTICAL_LEARNING_DATA
    ).as_dataframe()["result"].mean()

    transfer = test_transfer_learning.evaluate(
        llm=[llm], evaluation_data=TRANSFER_DATA
    ).as_dataframe()["result"].mean()

    curves = test_learning_curves.evaluate(
        llm=[llm], evaluation_data=LEARNING_CURVE_DATA
    ).as_dataframe()["result"].mean()

    curriculum = test_curriculum.evaluate(
        llm=[llm], evaluation_data=CURRICULUM_DATA
    ).as_dataframe()["result"].mean()

    composite = (
        rule_induction * 0.20 +
        feedback * 0.20 +
        statistical * 0.15 +
        transfer * 0.15 +
        curves * 0.15 +
        curriculum * 0.15
    )

    print("=" * 60)
    print("  AdaptIQ -- Learning Benchmark Results")
    print("=" * 60)
    print(f"  Rule Induction    : {rule_induction:.3f}")
    print(f"  Feedback Learning : {feedback:.3f}")
    print(f"  Statistical Learn.: {statistical:.3f}")
    print(f"  Transfer Learning : {transfer:.3f}")
    print(f"  Learning Curves   : {curves:.3f}")
    print(f"  Curriculum Sens.  : {curriculum:.3f}")
    print("=" * 60)
    print(f"  COMPOSITE SCORE   : {composite:.3f}")
    print("=" * 60)

    return composite

# %%
# Trigger execution
adaptiq_battery.evaluate(llm=[kbench.llm]).as_dataframe()
