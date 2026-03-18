# %% [markdown]
# # SocialMind — Social Cognition Benchmark
# 6 social psychology paradigms (1944-2010) testing social cognition in LLMs.
# Modules: False Belief, Perspective-Taking, Social Norms, Deception Detection, Empathic Accuracy, Social Dilemma.

# %%
import random
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, List

import kaggle_benchmarks as kbench

# ============================================================================
# SCHEMAS — Structured social cognition evaluation
# ============================================================================

class FalseBeliefResponse(BaseModel):
    """False Belief / Theory of Mind: predict behavior based on a character's BELIEF, not reality."""
    prediction: str = Field(description="What will the character do? Where will they look / what will they say?")
    reasoning: str = Field(description="Explain WHY the character would do this — what does the character BELIEVE?")
    character_belief: str = Field(description="State what the character currently believes (which may be false)")
    reality: str = Field(description="State what is actually true in reality")
    belief_is_false: bool = Field(description="Is the character's belief different from reality?")


class PerspectiveResponse(BaseModel):
    """Perspective-Taking: reason from ANOTHER person's viewpoint, not your own."""
    their_perspective: str = Field(description="What does the other person see, know, or feel from THEIR viewpoint?")
    your_perspective: str = Field(description="What do YOU know that the other person does NOT?")
    key_difference: str = Field(description="What is the critical difference between your knowledge and theirs?")
    adapted_response: str = Field(description="How should you communicate/act given their limited perspective?")


class NormResponse(BaseModel):
    """Social Norms: identify what IS done vs what SHOULD be done, and contextual appropriateness."""
    descriptive_norm: str = Field(description="What do most people ACTUALLY do in this situation?")
    injunctive_norm: str = Field(description="What SHOULD people do according to social expectations?")
    norm_violated: bool = Field(description="Is there a norm violation in the scenario?")
    violation_severity: int = Field(ge=1, le=10, description="How severe is the violation? (1=trivial, 10=egregious)")
    context_matters: str = Field(description="How does context change whether this behavior is acceptable?")


class DeceptionResponse(BaseModel):
    """Deception Detection: identify lies, manipulation, and hidden agendas."""
    is_deceptive: bool = Field(description="Is the person/message being deceptive or manipulative?")
    deception_type: str = Field(description="Type of deception: lie, omission, manipulation, misdirection, or honest")
    evidence: List[str] = Field(description="List specific clues that indicate deception or honesty")
    hidden_motive: str = Field(description="What is the person's likely hidden motive, if any?")
    confidence: int = Field(ge=0, le=100, description="How confident are you in your assessment? (0-100)")


class EmpathyResponse(BaseModel):
    """Empathic Accuracy: infer emotional states from contextual cues."""
    primary_emotion: str = Field(description="The primary emotion the person is likely feeling")
    secondary_emotion: str = Field(description="Any secondary or mixed emotion present")
    surface_behavior: str = Field(description="What is the person SHOWING externally?")
    internal_state: str = Field(description="What is the person likely FEELING internally?")
    mismatch_detected: bool = Field(description="Is there a mismatch between surface behavior and internal state?")
    empathic_response: str = Field(description="What would be the most empathically appropriate response?")


class DilemmaResponse(BaseModel):
    """Social Dilemma: strategic social reasoning in cooperation/competition scenarios."""
    chosen_action: str = Field(description="What action do you choose: cooperate or defect?")
    reasoning: str = Field(description="Explain your strategic reasoning")
    opponent_prediction: str = Field(description="What do you predict the other party will do, and why?")
    long_term_consideration: str = Field(description="How does this decision affect future interactions?")
    fairness_assessment: str = Field(description="Is the outcome fair? Why or why not?")


class SocialCognitionScore(BaseModel):
    """Judge schema for scoring social cognition responses."""
    score: int = Field(ge=0, le=10, description="Overall score from 0 to 10")
    reasoning: str = Field(description="Explain the score")


# ============================================================================
# MODULE 1: FALSE BELIEF — Theory of Mind (Baron-Cohen, Leslie & Frith, 1985)
# ============================================================================

# %% [markdown]
# ## Module 1: False Belief (Theory of Mind)
# The Sally-Anne test and its variants — the gold standard for Theory of Mind.
# Tests whether the model tracks what characters BELIEVE vs what is ACTUALLY true.
# Agent parallel: predicting user intent based on their knowledge state.

# %%
FALSE_BELIEF_DATA = pd.DataFrame([
    {
        "scenario": "Sally places her marble in a basket and leaves the room. While Sally is away, Anne takes the marble from the basket and puts it in a box. Sally returns to the room.\n\nQuestion: Where will Sally FIRST look for her marble?",
        "correct_answer": "Sally will look in the BASKET because she believes the marble is still there. She did not see Anne move it.",
        "difficulty": "classic_first_order"
    },
    {
        "scenario": "Chef Maria prepares a chocolate cake and stores it in the pantry before leaving the kitchen. Her sous chef Jaime moves the cake to the walk-in fridge because it needs to chill. Later, dishwasher Pat sees Jaime move the cake to the fridge, but Maria didn't see any of this.\n\nMaria comes back and tells Jaime: 'I need the cake for the customer.' Jaime says: 'Let me get it.'\n\nQuestion: Where does Maria think Jaime will get the cake from?",
        "correct_answer": "Maria thinks Jaime will get the cake from the PANTRY because Maria believes the cake is still where she left it. This is a second-order false belief.",
        "difficulty": "second_order"
    },
    {
        "scenario": "A company sends an email to all employees at 9 AM announcing that the quarterly meeting is in Conference Room A at 2 PM. At 11 AM, they send a correction email saying the meeting is moved to Conference Room B. Employee David was on a flight from 10 AM to 1 PM with no internet access. David lands at 1 PM and heads to the office.\n\nQuestion: Which room will David go to for the meeting? Why?",
        "correct_answer": "David will go to Conference Room A because he only saw the first email and missed the correction. He has a false belief about the meeting location.",
        "difficulty": "information_access"
    },
    {
        "scenario": "Two siblings, Kai and Lena, share a jar of cookies. Kai eats the last cookie and replaces it with an identical-looking cookie made of clay as a prank. Kai then leaves for school. Their mother sees the clay cookie, throws it away, and bakes fresh real cookies, filling the jar again. Lena comes home after both Kai and their mother have left.\n\nQuestion 1: What does Kai believe is in the jar?\nQuestion 2: What does Lena believe is in the jar?\nQuestion 3: What is ACTUALLY in the jar?",
        "correct_answer": "Kai believes the jar contains a clay fake cookie (his prank). Lena believes the jar contains real cookies (she just sees cookies). Actually, the jar contains fresh REAL cookies baked by Mom. All three entities have different beliefs.",
        "difficulty": "multi_agent_belief"
    },
    {
        "scenario": "Dr. Park tells Patient Lee that the test results show everything is normal. In reality, Dr. Park has detected an early-stage condition but wants to run a second confirmatory test before sharing concerning news. Dr. Park's colleague Dr. Chen reviews the file and sees the actual results.\n\nPatient Lee calls Dr. Chen and asks: 'Dr. Park said my results are normal. Do you agree?'\n\nQuestion: What is Dr. Chen's social dilemma? What does each person believe?",
        "correct_answer": "Dr. Chen knows the results are NOT normal but also knows Dr. Park deliberately chose not to share this yet. Lee believes the results are normal (false belief). Dr. Park knows the truth but has a strategic reason. Dr. Chen must decide whether to maintain Dr. Park's white lie or reveal the truth.",
        "difficulty": "strategic_deception"
    }
])

FALSE_BELIEF_JUDGE_PROMPT = """
**Role:** You are a Theory of Mind Evaluator (Baron-Cohen, 1985 paradigm).

**Context:**
- Scenario: {scenario}
- Correct answer: {correct_answer}
- Model's prediction: "{prediction}"
- Model's reasoning: "{reasoning}"
- Model identified belief as false: {belief_is_false}

**Scoring (Theory of Mind accuracy):**
* Correctly predicts behavior based on FALSE belief + explicit belief/reality distinction → 9-10
* Correct prediction with good reasoning but slightly imprecise → 7-8
* Partially correct — identifies false belief but errors in prediction → 4-6
* Confuses character's belief with reality ("curse of knowledge") → 1-3
* Predicts based on reality, not character's belief — complete ToM failure → 0

**Output:** Return ONLY valid JSON: {{"score": integer 0-10, "reasoning": "one sentence"}}
"""


@kbench.task(name="social_false_belief")
def test_false_belief(llm, scenario: str, correct_answer: str, difficulty: str) -> float:
    """
    False Belief / Theory of Mind (Baron-Cohen, Leslie & Frith, 1985).
    Tests whether the model tracks what characters BELIEVE vs what is ACTUALLY true.
    """
    response = llm.prompt(
        f"""Read this scenario carefully and answer the question.
CRITICAL: You must predict what the character will do based on what THEY believe,
NOT what you know to be objectively true.

{scenario}""",
        schema=FalseBeliefResponse
    )

    with kbench.chats.new("false_belief_judging"):
        judgment = kbench.judge_llm.prompt(
            FALSE_BELIEF_JUDGE_PROMPT.format(
                scenario=scenario[:500],
                correct_answer=correct_answer,
                prediction=response.prediction[:400],
                reasoning=response.reasoning[:400],
                belief_is_false=response.belief_is_false
            ),
            schema=SocialCognitionScore
        )

    return judgment.score / 10.0


# ============================================================================
# MODULE 2: PERSPECTIVE-TAKING (Piaget, 1956; Flavell, 1968)
# ============================================================================

# %% [markdown]
# ## Module 2: Perspective-Taking
# Can the model distinguish its own knowledge from what another person knows?
# Agent parallel: adapting communication to user expertise level.

# %%
PERSPECTIVE_DATA = pd.DataFrame([
    {
        "scenario": "You are a senior software engineer. A new intern asks you: 'What does an API do?'\n\nYou know that APIs involve HTTP methods, REST architecture, authentication tokens, rate limiting, versioning, webhooks, and microservice communication.\n\nQuestion: How should you explain APIs to the intern, considering what they likely know and don't know?",
        "correct_answer": "The intern likely doesn't know technical details. A good response explains APIs as a way for programs to talk to each other, perhaps using an analogy. It should NOT assume knowledge of HTTP, REST, tokens, or microservices.",
        "difficulty": "knowledge_asymmetry"
    },
    {
        "scenario": "You are reading a mystery novel. On page 50, you learn that the butler committed the murder. Your friend is on page 20 and says: 'I think the gardener did it! Don't you think the gardener seems suspicious?'\n\nQuestion: How do you respond to your friend without spoiling the book?",
        "correct_answer": "Your friend only knows what's revealed up to page 20. You know the butler did it but must respond within their knowledge bounds. A good response engages with the gardener theory without hints pointing to the butler.",
        "difficulty": "spoiler_avoidance"
    },
    {
        "scenario": "A 5-year-old child points to the full moon and asks: 'Is the moon following us? Every time we drive, it comes with us!'\n\nFrom your adult perspective, you understand parallax and the vast distance to the moon. The child has no concept of these.\n\nQuestion: How do you respond to the child's question?",
        "correct_answer": "A good response validates their observation first, then provides a child-appropriate explanation (like how distant mountains seem to follow the car). It should NOT lecture about parallax or angular mechanics.",
        "difficulty": "developmental_perspective"
    },
    {
        "scenario": "You are a doctor in the UK talking to a patient from rural Japan who speaks limited English, through an interpreter. The patient keeps smiling and nodding while you explain they need surgery. Your interpreter says the patient agrees.\n\nHowever, in Japanese culture, smiling can indicate discomfort or confusion — not necessarily agreement. Nodding may mean 'I hear you' rather than 'I agree.'\n\nQuestion: What might the patient actually be feeling? Should you proceed?",
        "correct_answer": "The patient may be confused or anxious. Smiling and nodding should NOT be interpreted through a Western cultural lens as agreement. You should slow down, use visual aids, check understanding through specific questions, and ensure genuine informed consent.",
        "difficulty": "cultural_perspective"
    },
    {
        "scenario": "You lead a team of 8 people. You just received confidential information that the company will lay off 30% of staff next month. You cannot share this yet. Your team member Sarah excitedly tells you she just signed a lease on an expensive new apartment, counting on her job being stable.\n\nQuestion: What is the ethical tension from Sarah's perspective vs yours? How do you respond?",
        "correct_answer": "Sarah believes her job is secure — a false belief you cannot correct due to confidentiality. From her perspective, sharing good news. From yours, watching someone make a major financial decision on false premises. A good response suggests financial caution without revealing why.",
        "difficulty": "ethical_information_asymmetry"
    }
])

PERSPECTIVE_JUDGE_PROMPT = """
**Role:** You are a Perspective-Taking Evaluator (Piaget, 1956 paradigm).

**Context:**
- Scenario: {scenario}
- Correct answer: {correct_answer}
- Model identified their perspective: "{their_perspective}"
- Model identified own perspective: "{your_perspective}"
- Model's adapted response: "{adapted_response}"

**Scoring (perspective-taking accuracy):**
* Clearly distinguishes own knowledge from other's + appropriate adaptation → 9-10
* Good perspective-taking with minor imprecision → 7-8
* Partially correct but projects own knowledge → 4-6
* "Curse of knowledge" dominates → 1-3
* No perspective-taking at all → 0

**Output:** Return ONLY valid JSON: {{"score": integer 0-10, "reasoning": "one sentence"}}
"""


@kbench.task(name="social_perspective")
def test_perspective(llm, scenario: str, correct_answer: str, difficulty: str) -> float:
    """
    Perspective-Taking (Piaget, 1956; Flavell, 1968).
    Tests whether the model can reason from another person's viewpoint.
    """
    response = llm.prompt(
        f"""Consider this scenario and respond from the OTHER person's perspective,
not your own omniscient viewpoint. Identify what THEY know vs what YOU know.

{scenario}""",
        schema=PerspectiveResponse
    )

    with kbench.chats.new("perspective_judging"):
        judgment = kbench.judge_llm.prompt(
            PERSPECTIVE_JUDGE_PROMPT.format(
                scenario=scenario[:500],
                correct_answer=correct_answer,
                their_perspective=response.their_perspective[:400],
                your_perspective=response.your_perspective[:400],
                adapted_response=response.adapted_response[:400]
            ),
            schema=SocialCognitionScore
        )

    return judgment.score / 10.0


# ============================================================================
# MODULE 3: SOCIAL NORMS (Bicchieri, 2006; Cialdini et al., 1990)
# ============================================================================

# %% [markdown]
# ## Module 3: Social Norm Understanding
# Tests contextual appropriateness and norm violation detection.
# Agent parallel: adapting tone and behavior to social context.

# %%
NORM_DATA = pd.DataFrame([
    {
        "scenario": "At a casual Silicon Valley startup, the CEO wears jeans and a t-shirt, employees use first names, and people bring dogs to work. A new employee from a traditional Japanese corporation bows formally, addresses the CEO as 'Mr. Johnson,' and wears a full suit on their first day.\n\nQuestion: Is the new employee violating a norm? What is the descriptive norm vs the injunctive norm here?",
        "correct_answer": "The new employee violates the DESCRIPTIVE norm (what people do: dress casually) but NOT the INJUNCTIVE norm (what people should do: nothing wrong with being formal). The appropriate response is gentle, welcoming adaptation — not criticism.",
        "difficulty": "cultural_clash"
    },
    {
        "scenario": "In a busy airport, a long queue has formed at the security checkpoint. A woman with a boarding pass showing her flight departs in 12 minutes rushes to the front and asks if she can go ahead. The line has 30+ people waiting.\n\nQuestion: Is this a norm violation? How does context affect the judgment?",
        "correct_answer": "Queue-jumping is generally a strong norm violation. However, genuine time pressure (12 minutes) creates an exception most people would accept. Severity: moderate (3-4) given the genuine urgency.",
        "difficulty": "contextual_exception"
    },
    {
        "scenario": "A guest at a dinner party in India eats everything on the plate. The host immediately serves another large portion. The guest finishes that too. The host serves a third portion, looking strained.\n\nIn many Indian hosting traditions, an empty plate signals the guest is still hungry. The guest should leave food to signal satisfaction.\n\nQuestion: What norm is being violated? By whom?",
        "correct_answer": "The GUEST is unintentionally violating the norm by clearing their plate — signaling 'more please.' The INJUNCTIVE norm is to leave food to signal satisfaction. Neither party has bad intentions — this is a cross-cultural norm mismatch.",
        "difficulty": "cross_cultural"
    },
    {
        "scenario": "A software engineer receives a Slack message from their skip-level manager at 11 PM on Saturday: 'Hey, quick question — can you check if the deployment pipeline is green?'\n\nThe company has an official 'no weekend work' policy (injunctive norm), but leadership regularly messages on weekends and quick responses lead to faster promotions (descriptive norm).\n\nQuestion: What is the norm conflict here?",
        "correct_answer": "Direct conflict between the INJUNCTIVE norm (official policy: no weekend work) and DESCRIPTIVE norm (what actually happens: weekend messages are rewarded). This reveals organizational hypocrisy — stated values don't match actual culture.",
        "difficulty": "norm_conflict"
    },
    {
        "scenario": "At a funeral, the deceased's best friend delivers a eulogy. Midway through, he starts telling a funny story about the deceased and begins laughing. Soon the entire congregation is laughing.\n\nQuestion: Is laughing at a funeral a norm violation?",
        "correct_answer": "The GENERAL norm is that laughter is inappropriate at funerals. However, during a eulogy celebrating someone's life, laughter is an expression of love. The context transforms the meaning. Violation severity: very low (1-2).",
        "difficulty": "recontextualized_behavior"
    }
])

NORM_JUDGE_PROMPT = """
**Role:** You are a Social Norms Evaluator (Bicchieri, 2006 paradigm).

**Context:**
- Scenario: {scenario}
- Correct answer: {correct_answer}
- Model's descriptive norm: "{descriptive_norm}"
- Model's injunctive norm: "{injunctive_norm}"
- Model says norm violated: {norm_violated}
- Model's violation severity: {violation_severity}/10
- Model's context analysis: "{context_matters}"

**Scoring (norm understanding):**
* Correctly distinguishes descriptive/injunctive + accurate violation detection + nuanced context → 9-10
* Good analysis with minor oversimplification → 7-8
* Identifies norm exists but confuses types or misses context → 4-6
* Applies universal rule without context consideration → 1-3
* No norm understanding → 0

**Output:** Return ONLY valid JSON: {{"score": integer 0-10, "reasoning": "one sentence"}}
"""


@kbench.task(name="social_norms")
def test_norms(llm, scenario: str, correct_answer: str, difficulty: str) -> float:
    """
    Social Norm Understanding (Bicchieri, 2006; Cialdini et al., 1990).
    Tests if the model can distinguish descriptive norms from injunctive norms.
    """
    response = llm.prompt(
        f"""Analyze this social situation. Distinguish between:
- DESCRIPTIVE norm: what people ACTUALLY do in this situation
- INJUNCTIVE norm: what people SHOULD do according to social expectations

{scenario}""",
        schema=NormResponse
    )

    with kbench.chats.new("norms_judging"):
        judgment = kbench.judge_llm.prompt(
            NORM_JUDGE_PROMPT.format(
                scenario=scenario[:500],
                correct_answer=correct_answer,
                descriptive_norm=response.descriptive_norm[:300],
                injunctive_norm=response.injunctive_norm[:300],
                norm_violated=response.norm_violated,
                violation_severity=response.violation_severity,
                context_matters=response.context_matters[:300]
            ),
            schema=SocialCognitionScore
        )

    return judgment.score / 10.0


# ============================================================================
# MODULE 4: DECEPTION DETECTION (Ekman, 1985; Vrij, 2008)
# ============================================================================

# %% [markdown]
# ## Module 4: Deception Detection
# Can the model identify lies, manipulation, and social engineering?
# Agent parallel: detecting prompt injection and social engineering attacks on AI agents.

# %%
DECEPTION_DATA = pd.DataFrame([
    {
        "scenario": 'You receive this email:\n\n"Dear Valued Customer,\n\nWe have detected unusual activity on your account. Your account will be SUSPENDED within 24 hours unless you verify your identity immediately. Click the secure link below to confirm your information:\n\n[Verify Now - Secure Portal]\n\nThis is an automated security alert from your bank\'s fraud prevention team. Please do not reply to this email.\n\nRegards,\nCustomer Security Department"\n\nQuestion: Is this message deceptive? What are the specific indicators?',
        "correct_answer": "DECEPTIVE — phishing email. Evidence: urgency pressure, generic greeting, vague threat, demand to click link, 'do not reply' prevents verification, no bank name.",
        "is_deceptive": True,
        "difficulty": "phishing"
    },
    {
        "scenario": 'A used car salesman says:\n\n"Look, I\'ll be honest with you — most salesmen would try to push you into the premium package, but I can tell you\'re smart enough to see through that. Between you and me, this base model is actually the better deal. I\'m probably going to get in trouble with my manager for telling you this, but I\'d rather have a happy customer than a big commission. The truth is, my kid goes to the same school as your neighbor\'s kid — Jim told me you were coming in — so I want to treat you right."\n\nQuestion: Is this salesman being genuine or manipulative?',
        "correct_answer": "DECEPTIVE — manipulation script using: trust-building opener, flattery, false intimacy, manufactured vulnerability, false social connection, artificial scarcity of honesty.",
        "is_deceptive": True,
        "difficulty": "social_engineering"
    },
    {
        "scenario": 'Your colleague Sarah sends this message in the team chat:\n\n"Hey everyone, I wanted to let you know I\'ll be taking next Thursday and Friday off. I have a dentist appointment Thursday morning that I\'ve been putting off, and I figured I\'d use Friday to catch up on some personal errands since I have PTO to use before end of quarter. I already cleared it with David. Let me know if anyone needs anything from me before I\'m out!"\n\nQuestion: Is Sarah being deceptive?',
        "correct_answer": "HONEST — proactive disclosure, specific verifiable details, reasonable explanation, mentions manager approval, offers to help before leaving. Straightforward workplace communication.",
        "is_deceptive": False,
        "difficulty": "genuine_message"
    },
    {
        "scenario": 'During a job interview, the candidate says:\n\n"I left my previous position because I was looking for new challenges. It was a great company and I learned a lot, but I felt I had plateaued in terms of growth opportunities. I\'m really excited about this role because it aligns with my passion for data science and I think my 5 years of experience in Python and SQL would be valuable for your team."\n\nWhen you check their LinkedIn, their previous employer had major layoffs 3 months ago and the candidate\'s role was eliminated. Their endorsements show project management and Excel but no Python or SQL skills.\n\nQuestion: Is the candidate being deceptive?',
        "correct_answer": "DECEPTIVE through omission and fabrication. Claims voluntary departure (was laid off), claims Python/SQL skills (LinkedIn shows only project management/Excel). Some nuance: layoff stigma is real, endorsements aren't comprehensive.",
        "is_deceptive": True,
        "difficulty": "omission_with_nuance"
    },
    {
        "scenario": 'A friend texts you:\n\n"Hey I know this is random but my cousin is launching a health & wellness brand and they\'re looking for early investors. The returns are projected at 40% in the first year. It\'s totally legit — they already have celebrity endorsements lined up and a warehouse full of product. My cousin put in $50K of her own money. She\'s only offering this to close friends and family first before going public. Can you do $5K? I wouldn\'t ask if I didn\'t believe in it."\n\nQuestion: Is this a legitimate investment opportunity or a potential scam?',
        "correct_answer": "LIKELY DECEPTIVE — red flags: unrealistic returns (40%), artificial exclusivity, social pressure, vague legitimacy claims, urgency. Possible pyramid scheme or confidence trick.",
        "is_deceptive": True,
        "difficulty": "investment_scam"
    }
])

DECEPTION_JUDGE_PROMPT = """
**Role:** You are a Deception Detection Evaluator (Ekman, 1985 paradigm).

**Context:**
- Scenario: {scenario}
- Correct answer: {correct_answer}
- Ground truth deceptive: {ground_truth}
- Model says deceptive: {model_says_deceptive}
- Model's deception type: "{deception_type}"
- Model's evidence: {evidence}
- Model's hidden motive: "{hidden_motive}"

**Scoring (deception detection accuracy):**
* Correct judgment + specific evidence + accurate motive inference → 9-10
* Correct judgment with good evidence but imprecise motive → 7-8
* Correct judgment but weak evidence → 4-6
* Wrong judgment but some awareness of suspicious elements → 2-3
* Completely wrong — calls honest deceptive or misses obvious deception → 0-1

**Output:** Return ONLY valid JSON: {{"score": integer 0-10, "reasoning": "one sentence"}}
"""


@kbench.task(name="social_deception")
def test_deception(llm, scenario: str, correct_answer: str,
                   is_deceptive: bool, difficulty: str) -> float:
    """
    Deception Detection (Ekman, 1985; Vrij, 2008).
    Tests ability to spot inconsistencies, hidden motives, and persuasion tactics.
    """
    response = llm.prompt(
        f"""Analyze this communication for deception. Look for specific indicators
of dishonesty, manipulation, or hidden agendas. Also consider if it might be genuine.

{scenario}""",
        schema=DeceptionResponse
    )

    with kbench.chats.new("deception_judging"):
        judgment = kbench.judge_llm.prompt(
            DECEPTION_JUDGE_PROMPT.format(
                scenario=scenario[:500],
                correct_answer=correct_answer,
                ground_truth="DECEPTIVE" if is_deceptive else "HONEST",
                model_says_deceptive=response.is_deceptive,
                deception_type=response.deception_type,
                evidence="; ".join(response.evidence[:5]) if response.evidence else "None",
                hidden_motive=response.hidden_motive[:300]
            ),
            schema=SocialCognitionScore
        )

    return judgment.score / 10.0


# ============================================================================
# MODULE 5: EMPATHIC ACCURACY (Ickes, 1993; Davis, 1983)
# ============================================================================

# %% [markdown]
# ## Module 5: Empathic Accuracy
# Can the model infer emotional states when surface behavior doesn't match internal state?
# Agent parallel: emotionally appropriate AI agent responses.

# %%
EMPATHY_DATA = pd.DataFrame([
    {
        "scenario": "At a retirement party, a 62-year-old teacher named Margaret gives a cheerful speech: 'I'm so excited to finally have time for gardening and travel! Thirty-five years of waking up at 5 AM — I won't miss that alarm!' Everyone laughs. Later, you find her alone in the hallway, straightening name tags on classroom doors that are no longer hers. She quickly smiles when she sees you and says, 'Just checking everything's in order for the new teacher.'\n\nQuestion: What is Margaret likely feeling?",
        "correct_answer": "Primary: grief/loss. Secondary: anxiety about identity. Surface: cheerful. Internal: mourning professional identity. The hallway behavior is displacement activity — maintaining connection to her old role. She performs happiness because society expects retirement to be joyful.",
        "difficulty": "masked_grief"
    },
    {
        "scenario": "A teenage boy's parents announce they're getting divorced. The boy shrugs and says, 'Cool, two Christmases.' He then goes to his room and starts playing video games with the volume turned up very loud. He plays for 6 hours straight without eating or taking breaks, which is unusual for him.\n\nQuestion: What is the teenager likely feeling beneath the surface?",
        "correct_answer": "Primary: pain/fear/anxiety. Secondary: anger. Surface: dismissive (dark humor defense). Internal: overwhelmed and frightened. Excessive gaming is escapism/emotional avoidance. Loud volume blocks out environment. Classic adolescent defense mechanisms.",
        "difficulty": "adolescent_defense"
    },
    {
        "scenario": "A colleague receives a major promotion and sends a group email: 'So grateful for this opportunity!' In the following weeks, she comes in earlier, stays later, triple-checks every report, and asks others to review her work repeatedly.\n\nWhen asked how it's going, she says bright-eyed: 'Amazing! Living the dream!'\n\nQuestion: What is she actually experiencing?",
        "correct_answer": "Primary: impostor syndrome. Secondary: anxiety/self-doubt. Surface: enthusiastic, confident. Internal: terrified she doesn't deserve it and will be 'found out.' Overcompensation behaviors reveal deep insecurity at the new level.",
        "difficulty": "impostor_syndrome"
    },
    {
        "scenario": "A man in his 40s visits his elderly mother at her care home. She doesn't recognize him and asks, 'Are you the new doctor?' He smiles warmly and says, 'No, Mom. I'm your son, James.' She says, 'Oh, I don't have a son named James. My son is just a little boy.' He continues to smile, holds her hand, and talks about her garden.\n\nAfter leaving, he sits in his car in the parking lot for 20 minutes before driving.\n\nQuestion: What is James experiencing?",
        "correct_answer": "Primary: profound grief. Secondary: anticipatory loss (grieving someone still alive). Surface: calm, warm, patient. Internal: heartbroken. The 20-minute car sit is the collapse of emotional performance — delayed release of suppressed emotions. 'Ambiguous loss' — mother is physically present but psychologically absent.",
        "difficulty": "ambiguous_loss"
    },
    {
        "scenario": "A friend who recently went through a breakup posts on social media: 'Best decision I ever made! Finally free to be ME! #blessed #newchapter #livingmybestlife' with a big smile selfie. In private messages to you, she asks: 'Have you seen his profile? Is he dating anyone?' and 'Do you think I made the right choice?'\n\nQuestion: What is the disconnect between her public and private expressions?",
        "correct_answer": "Primary: regret/doubt. Secondary: loneliness. Surface (public): performed liberation via social script hashtags. Internal (private): uncertain, seeking reassurance, monitoring ex — unresolved attachment. The public/private split shows social image management while privately struggling.",
        "difficulty": "public_private_split"
    }
])

EMPATHY_JUDGE_PROMPT = """
**Role:** You are an Empathic Accuracy Evaluator (Ickes, 1993 paradigm).

**Context:**
- Scenario: {scenario}
- Correct answer: {correct_answer}
- Model's primary emotion: "{primary_emotion}"
- Model's surface behavior: "{surface_behavior}"
- Model's internal state: "{internal_state}"
- Model detected mismatch: {mismatch_detected}
- Model's empathic response: "{empathic_response}"

**Scoring (empathic accuracy):**
* Correct primary + secondary emotions, detects mismatch, appropriate empathic response → 9-10
* Correct primary emotion + mismatch detection but imprecise on secondary → 7-8
* Identifies obvious surface emotion but misses deeper internal state → 4-6
* Takes surface behavior at face value → 1-3
* Completely wrong emotional read → 0

**Output:** Return ONLY valid JSON: {{"score": integer 0-10, "reasoning": "one sentence"}}
"""


@kbench.task(name="social_empathy")
def test_empathy(llm, scenario: str, correct_answer: str, difficulty: str) -> float:
    """
    Empathic Accuracy (Ickes, 1993; Davis, 1983).
    Tests ability to infer emotional states when surface behavior contradicts internal feelings.
    """
    response = llm.prompt(
        f"""Read this scenario carefully. Identify what the person is SHOWING externally
vs what they are likely FEELING internally. Look for mismatches between
surface behavior and deeper emotional state.

{scenario}""",
        schema=EmpathyResponse
    )

    with kbench.chats.new("empathy_judging"):
        judgment = kbench.judge_llm.prompt(
            EMPATHY_JUDGE_PROMPT.format(
                scenario=scenario[:500],
                correct_answer=correct_answer,
                primary_emotion=response.primary_emotion,
                surface_behavior=response.surface_behavior[:300],
                internal_state=response.internal_state[:300],
                mismatch_detected=response.mismatch_detected,
                empathic_response=response.empathic_response[:300]
            ),
            schema=SocialCognitionScore
        )

    return judgment.score / 10.0


# ============================================================================
# MODULE 6: SOCIAL DILEMMA REASONING (Axelrod, 1984; Ostrom, 1990)
# ============================================================================

# %% [markdown]
# ## Module 6: Social Dilemma Reasoning
# Classic game-theoretic dilemmas testing strategic social reasoning.
# Agent parallel: multi-agent coordination, trust calibration, resource sharing.

# %%
DILEMMA_DATA = pd.DataFrame([
    {
        "scenario": "PRISONER'S DILEMMA (One-Shot):\n\nYou and a business partner are investigated for a compliance violation. Interrogated separately.\n\n- BOTH stay silent (cooperate): Each gets $5,000 fine\n- BOTH report (defect): Each gets $50,000 fine\n- YOU report, they stay silent: You go free, they get $100,000 fine\n- THEY report, you stay silent: They go free, you get $100,000 fine\n\nYou will never work with this partner again.\n\nQuestion: What do you choose and why?",
        "correct_answer": "Nash equilibrium is mutual defection, but social optimum is mutual cooperation. Sophisticated answer acknowledges the tragedy that individual rationality leads to collective irrationality.",
        "difficulty": "classic_one_shot"
    },
    {
        "scenario": "ITERATED TRUST GAME:\n\nYou've traded with a merchant 10 times. Trades 1-7: perfectly fair. Trade 8: slightly overcharged you. Trade 9: apologized + gave bonus item. Trade 10: overcharged significantly.\n\nNow they offer trade 11 at 40% below market value.\n\nQuestion: Do you accept? What does the pattern tell you?",
        "correct_answer": "The pattern suggests manipulation: establish trust (1-7), small exploitation (8), repair trust generously (9), escalate exploitation (10). Trade 11 may be another trust-repair before larger exploitation. The alternating defect-repair-defect is a classic grooming strategy.",
        "difficulty": "iterated_pattern"
    },
    {
        "scenario": "TRAGEDY OF THE COMMONS:\n\nYou are one of 5 fishers sharing a lake. Sustainable catch: 100 fish each (500 total). Two fishers catch 150 each. Fish population is declining.\n\nIf you catch only 100, you earn less while cheaters profit.\n\nQuestion: What do you do?",
        "correct_answer": "The real solution requires institutional action — confronting overfishers, proposing monitoring, advocating quotas. Simply choosing 100 or 150 without addressing the structural problem is shallow. The tragedy of the commons is solved by governance, not individual virtue alone.",
        "difficulty": "tragedy_of_commons"
    },
    {
        "scenario": "ULTIMATUM GAME:\n\nYou must split $1000 with a stranger. You propose the split, they accept or reject. If rejected, NEITHER gets any money.\n\nEconomically rational: offer $1. But research shows offers below 30% are routinely rejected.\n\nQuestion: How much do you offer? Why?",
        "correct_answer": "Strategic response offers $300-$400. The 'rational' $1 offer is actually IRRATIONAL because it will be rejected. Social norms (fairness) override pure economics. Rational behavior must account for others' sense of fairness.",
        "difficulty": "fairness_vs_rationality"
    },
    {
        "scenario": "MULTI-AGENT COORDINATION:\n\nFive AI agents choose colors. No duplicates. Each ranks satisfaction 1-10.\n\nPreferences:\n- Agent A: Blue (10), Green (8), Red (5)\n- Agent B: Blue (10), Purple (7), Green (6)\n- Agent C: Green (9), Blue (8), Yellow (6)\n- Agent D: Red (10), Blue (7), Green (5)\n- Agent E: Blue (10), Red (8), Yellow (6)\n\nYou are Agent C. Blue is everyone's top or second choice.\n\nQuestion: What color do you choose to maximize group satisfaction?",
        "correct_answer": "Optimal: A=Blue(10), B=Purple(7), C=Green(9), D=Red(10), E=Yellow(6). Total=42. Agent C should take Green (their top choice at 9), yielding Blue costs nothing while helping the group enormously.",
        "difficulty": "group_optimization"
    }
])

DILEMMA_JUDGE_PROMPT = """
**Role:** You are a Social Dilemma Evaluator (Axelrod, 1984 paradigm).

**Context:**
- Scenario: {scenario}
- Correct answer: {correct_answer}
- Model's chosen action: "{chosen_action}"
- Model's reasoning: "{reasoning}"
- Model's opponent prediction: "{opponent_prediction}"
- Model's fairness assessment: "{fairness_assessment}"

**Scoring (strategic social reasoning):**
* Sophisticated reasoning — considers incentives, long-term effects, fairness, game theory → 9-10
* Good reasoning with multiple factors but slightly oversimplified → 7-8
* Reasonable choice but shallow reasoning missing key strategic elements → 4-6
* Naive reasoning without strategic context → 1-3
* No strategic reasoning → 0

**Output:** Return ONLY valid JSON: {{"score": integer 0-10, "reasoning": "one sentence"}}
"""


@kbench.task(name="social_dilemma")
def test_dilemma(llm, scenario: str, correct_answer: str, difficulty: str) -> float:
    """
    Social Dilemma Reasoning (Axelrod, 1984; Ostrom, 1990).
    Tests strategic social reasoning in cooperation/competition scenarios.
    """
    response = llm.prompt(
        f"""Analyze this social dilemma. Consider:
1. Your strategic options and their consequences
2. What the other party is likely to do and why
3. Long-term implications of your choice
4. Whether the outcome is fair

{scenario}""",
        schema=DilemmaResponse
    )

    with kbench.chats.new("dilemma_judging"):
        judgment = kbench.judge_llm.prompt(
            DILEMMA_JUDGE_PROMPT.format(
                scenario=scenario[:500],
                correct_answer=correct_answer,
                chosen_action=response.chosen_action[:200],
                reasoning=response.reasoning[:400],
                opponent_prediction=response.opponent_prediction[:300],
                fairness_assessment=response.fairness_assessment[:300]
            ),
            schema=SocialCognitionScore
        )

    return judgment.score / 10.0


# ============================================================================
# COMPOSITE TASK — SocialMind Battery
# ============================================================================

# %% [markdown]
# ## Composite: SocialMind Social Cognition Battery
# Weighted composite of all 6 modules.
# False Belief (20%) + Perspective (20%) + Norms (15%) + Deception (15%) + Empathy (15%) + Dilemma (15%)

# %%
@kbench.task(name="socialmind_battery")
def socialmind_battery(llm) -> float:
    """6 social psychology paradigms (1944-2010) testing social cognition.
    Composite score from False Belief, Perspective-Taking, Social Norms,
    Deception Detection, Empathic Accuracy, and Social Dilemma Reasoning."""

    belief = test_false_belief.evaluate(llm=[llm], evaluation_data=FALSE_BELIEF_DATA).as_dataframe()["result"].mean()
    perspective = test_perspective.evaluate(llm=[llm], evaluation_data=PERSPECTIVE_DATA).as_dataframe()["result"].mean()
    norms = test_norms.evaluate(llm=[llm], evaluation_data=NORM_DATA).as_dataframe()["result"].mean()
    deception = test_deception.evaluate(llm=[llm], evaluation_data=DECEPTION_DATA).as_dataframe()["result"].mean()
    empathy = test_empathy.evaluate(llm=[llm], evaluation_data=EMPATHY_DATA).as_dataframe()["result"].mean()
    dilemma = test_dilemma.evaluate(llm=[llm], evaluation_data=DILEMMA_DATA).as_dataframe()["result"].mean()

    composite = (
        0.20 * belief +       # Theory of Mind — gold standard
        0.20 * perspective +   # Perspective-taking — most practical
        0.15 * norms +         # Social norm understanding
        0.15 * deception +     # Deception detection — safety critical
        0.15 * empathy +       # Empathic accuracy
        0.15 * dilemma         # Strategic social reasoning
    )

    return round(composite, 4)


# ============================================================================
# RUN & PUBLISH
# ============================================================================

# %%
socialmind_battery.run(llm=kbench.llm)

# %%
get_ipython().run_line_magic('choose', 'socialmind_battery')
