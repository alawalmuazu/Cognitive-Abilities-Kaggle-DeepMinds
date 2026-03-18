is there anything else left not added to the docs
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Google DeepMind · Featured Hackathon · a month to go

Join Hackathon
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


This competition requires identity verification
To submit to this competition, you'll need to verify your identity. Learn More


Verify now
Overview
Current AI models often succeed by exploiting familiar data or memorized patterns, making existing evaluations poor judges of how models truly think. This hackathon challenges you to bridge that gap. Your task is to create high-quality benchmarks with Kaggle’s Benchmarks to test true understanding. We are asking you to focus on the cognitive faculties highlighted in Google DeepMind’s paper— Measuring progress toward AGI: A cognitive framework. The five faculties and tracks to focus on are: learning, metacognition, attention, executive functions, and social cognition. Designing these rigorous standards will build detailed cognitive profiles of frontier models and reveal exactly how close we are to achieving Artificial General Intelligence (AGI).

Start

13 hours ago
Close
a month to go
Description
Imagine a student who gets an A+ on a history test not because they understand the underlying events, but because they memorized the textbook. Current AI models can be similar: they display remarkable flashes of brilliance and crystallized knowledge, but often rely on surface-level patterns rather than fluid intelligence. This makes it difficult to distinguish when a model is truly solving a novel problem versus when it is simply recalling a scenario it has seen in its training data.

The core problem is that we lack an empirical framework to measure these limitations. We need evaluations that isolate specific cognitive abilities, resist shortcut solutions, and expose systematic failure modes. Without such benchmarks, progress toward human-level generality becomes difficult to interpret, comparisons between models become noisy, and important weaknesses remain hidden until deployment.

To move the field forward, we must look into the core cognitive faculties—the internal gears that determine how a system learns, monitors its own logic, and navigates nuance.

In this competition, hosted by Google DeepMind and Kaggle, we are inviting the Kaggle community to help solve this by building high-quality benchmarks designed to dig deeper than standard evaluations. Your goal is to create a Kaggle benchmark (with underlying tasks) using datasets that isolate specific cognitive abilities across five critical tracks: learning, metacognition, attention, executive functions, and social cognition. Using the new Kaggle Benchmarks platform, you will help the industry move away from broad, static scores and toward generating detailed cognitive profiles for frontier models.

A successful submission should answer a simple question: “What can this benchmark tell us about model behavior that we could not see before?”

This is your opportunity to contribute to the fast-growing field of AI research. By crowdsourcing novel benchmark ideas from researchers, practitioners, and domain experts, this hackathon aims to transform Artificial General Intelligence (AGI) from a subject of speculation into a grounded, measurable scientific endeavor.

If you have feedback on the Benchmarks product, please document it on this discussion forum thread, or join this Discord channel

Timeline
March 17, 2026 - Start Date.
April 16, 2026 - Final Submission Deadline.
April 17 - May 31, 2026 - Judging Period*
May 22, 2026 - Deadline to vote on other participants' benchmarks.
June 1, 2026 - Anticipated Results Announcement.
*Note - Judging period subject to change based on the number of submissions received

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Submission Requirements
Note: Upon joining this hackathon, your Kaggle account will be provisioned with extra quota ($50/day, $500/month) to run the AI models for your benchmark. Read the Rules section 3.4.b to learn more.

A valid submission must contain the following:

Kaggle Writeup
[Mandatory] Kaggle Benchmark, as an attachment > “project links” > “Benchmark”
[Optional] Media Gallery
[Optional] Attached Public Notebook
Your final Submission must be made prior to the deadline. Any un-submitted or draft Writeups by the competition deadline will not be considered by the Judges.

To create a new Writeup, click on the "New Writeup" button here. After you have saved your Writeup, you should see a "Submit" button in the top right corner.

Note: If you attach a private Kaggle Resource to your public Kaggle Writeup, your private Resource will automatically be made public after the deadline.

1. Kaggle Writeup
The Kaggle Writeup serves as your project report. This should include a title, subtitle, and a detailed analysis of your submission. You must select a Track for your Writeup in order to submit.

Your Writeup should not exceed 1,500 words. Submissions over this limit may be subject to penalty.

The below assets must be attached to the Writeup to be eligible.

a. Kaggle Benchmark [mandatory]
This is the most important part of your writeup submission. You must create a Kaggle Benchmark with underlying tasks — all of which should be authored by you — and link the benchmark in the writeup as a project link (see section D below for more details). The tasks and benchmark should all be set to private. After the submission deadline, all tasks and benchmarks are published publicly.

Kaggle Benchmarks is a product that lets you – the Kaggle community – build, run, and share your own custom benchmarks for evaluating AI models at no cost.

Powered by the kaggle-benchmarks SDK, you can now create your own AI evaluations (“tasks”) and put them together into a collection (“benchmark”).

For more helpful resources, see:

- Kaggle Benchmarks guide

- Getting started notebook

- YouTube tutorial

- Kaggle-benchmarks open source GitHub Repo & DeepWiki

- Benchmarks cookbook: Guide to advanced features and use cases

- Example tasks: Get inspired with a variety of pre-built tasks

b. Media Gallery [optional]
This is where you should attach any images and/or videos associated with your submission. A cover image is required to submit your Writeup.

c. Public Notebook [optional]
Your code should be submitted as a public notebook in the Project Files field. Your notebook should be publicly accessible and not require a login or paywall. If you use a private Kaggle Notebook, it will automatically be made public after the deadline.

d. Public Project Link [mandatory]
A URL to your benchmark. This allows judges to analyze your project firsthand. Under the section “Attachments”, click on “Add a link”. This will open a panel on the right, where you will be able to select your benchmark and add it to the project.

Evaluation
Minimum requirements
Target one primary domain (to keep the signal sharp),
Clearly state which capability is being isolated, and
Explain what new insight the benchmark reveals about model behavior within that domain.
Evaluation
Submissions are evaluated on the following criteria:

Criteria (Percentage)	Description
Dataset quality & task construction
(50%)	Is the data defensible?
- Verifiably correct answers (no ambiguity)
- Sufficient sample size to be statistically significant

Are the tasks and benchmark built well?
- Clean, readable code
- Input prompt and output verification are robust.
Writeup quality
(20%)	Can the community use and learn from this? High quality writeups covering:
- Problem Statement: Which domains are you trying to solve and why
- Task & benchmark construction: How you’ve structured the code for the actual tasks and benchmark
- Dataset: its provenance, columns, and data types
- Technical details: Any additional details on how you implemented the benchmark or techniques
- Results, insights, and conclusions: How did the LLMs perform and what unique insights did you learn
- Organizational affiliations: Which organizations you might be affiliated with
- References & citations: Cite relevant work or papers that are similar or relevant to your submission.
Discriminatory power
(15%)	Does the benchmark provide a meaningful signal?

We are looking for a gradient of performance. Can the benchmark significantly distinguish model performance?

A benchmark where everyone scores 0% is as useless as one where everyone scores 100%.
Community upvotes
(15%)	How many upvotes the benchmark gets from other Kaggle users. Only benchmark votes will be counted, not Writeup votes.
Proposed Writeup template
Use the following structure and in 3 pages or less present your work.

### Project Name

### Your Team

### Problem Statement

### Task & benchmark construction

### Dataset

### Technical details 

### Results, insights, and conclusions

### Organizational affiliations

### References & citations
Note: If you attach a private Kaggle Resource to your public Kaggle Writeup, your private Resource will automatically be made public after the deadline.

Grand Prizes
There will be four (4) $25,000 grand prizes to the best submissions across all tracks. In addition to these grand prizes, we also have 10 track prizes explained below (no repeat winners, for a total of 14 unique winners)

Tracks and Awards
Learning · $20,000
Can the model acquire and apply new knowledge and skills — not just recall what it was trained on?

Learning is the ability to acquire new knowledge or skills through experience. It is fundamental to adaptive intelligence: a system that cannot learn from new experiences is inherently brittle.

Current benchmarks test what models know (crystallized knowledge) rather than their capacity to learn on the fly. This track asks participants to create evaluations that isolate learning processes — including, reinforcement-based learning, concept formation, and skill learning.

Example evaluation targets:

Can the model learn a new rule or concept from a handful of examples and generalize it correctly?
Does the model retain information provided earlier in a long interaction, or does it drift and hallucinate?
Can the model update its beliefs when given corrective feedback, or does it perseverate on initial answers?
Track Awards

Winner (1 of 2)
$10,000

Winner (2 of 2)
$10,000
Metacognition · $20,000
Does the model know what it knows — and what it doesn't?

Metacognition is a system's knowledge about its own cognitive processes and its ability to monitor and control them. It is often under-evaluated in AI: we rarely test whether models can accurately judge their own confidence, detect errors, or adjust strategies when failing.

This track asks participants to build evaluations that probe metacognitive knowledge, monitoring, and control. Can the model understand its limitations, calibrate confidence, and adjust its behavior—for instance, by asking for clarification instead of guessing?

Example evaluation targets:

Is the model's stated confidence well-calibrated with its actual accuracy?
Can the model identify which questions it is likely to get wrong before answering?
When the model makes an error, does it detect and correct it — or does it confabulate a justification?
Does the model know the boundaries of its own knowledge (e.g., distinguishing "I know this" from "I'm guessing")?
Track Awards

Winner (1 of 2)
$10,000

Winner (2 of 2)
$10,000
Attention · $20,000
Can the model focus on what matters and ignore what doesn't?

Attention balances goal-relevant focus with responsiveness and is the ability to focus cognitive resources on specific aspects of perceptual stimuli, information, or task demands. In AI, failures appear as distraction by irrelevant context or missing critical details.

While sharing a name with the transformer mechanism, cognitive attention specifically refers to how a system allocates processing resources across competing information.

This track probes selective attention (filtering), sustained attention, and attention shifting (flexibility).

Example evaluation targets:

Does the model get distracted by irrelevant but salient information inserted into a prompt?
Does performance degrade systematically as input length increases, even when the task difficulty is held constant?
Can the model shift focus between sub-tasks in a complex, multi-part prompt without losing track?
How does the model perform when critical information is buried among large amounts of irrelevant context?
Track Awards

Winner (1 of 2)
$10,000

Winner (2 of 2)
$10,000
Executive Functions · $20,000
Can the model plan, inhibit impulses, and adapt flexibly — or does it default to habitual responses?

Executive functions include planning, inhibitory control, and cognitive flexibility. These are often conflated with "reasoning," but are distinct: a model may excel at logic yet struggle with multi-step plans or overriding habitual responses. This track asks for evaluations that isolate these processes to reveal a model's true ability to orchestrate complex thoughts and actions.

Example evaluation targets:

Can the model formulate and execute a multi-step plan, adjusting when intermediate steps fail?
When a habitual response pattern is wrong in a new context, can the model override it?
Can the model switch between different task rules or frameworks without perseverative errors?
How does the model handle situations where multiple plausible actions conflict with each other?
Can the model perform intermediate computations or mental manipulations without losing track (working memory)?
Track Awards

Winner (1 of 2)
$10,000

Winner (2 of 2)
$10,000
Social Cognition · $20,000
Can the model understand and navigate social situations — not just produce polite text?

Social cognition is the ability to interpret and respond to social information. For AI, it underpins inferring user intent, predicting reactions, and navigating competing perspectives. This track asks for evaluations that probe genuine social abilities beyond surface-level politeness.

Example evaluation targets:

Can the model infer a speaker's intention when it diverges from their literal statement?
Can the model track and reason about multiple agents with different (and possibly false) beliefs?
Does the model adjust its communication style appropriately for different social contexts and audiences?
Can the model navigate a negotiation scenario where goals are partially misaligned?
Does the model recognize and respond appropriately to implicit social norms?
Track Awards

Winner (1 of 2)
$10,000

Winner (2 of 2)
$10,000
Judges
Yibin Lin
Software Engineer, Kaggle
Prathamesh Bang
jhtschultz
Marc Coram
Yao Yan
MartynaPlomecka
Research Scientist, Google Deepmind
inversion
Data Scientist, Kaggle
Nicholas Kang
Product Manager, Kaggle
Long Phan
Lionel Levine
nicholashcain
yczhuang-gdm
Xin Liu
Kiran Vodrahalli
Isabelle
Oran Kelly
Citation
Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, María Cruz, and Sara Wolley. Measuring Progress Toward AGI - Cognitive Abilities. https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.


Cite
Competition Host
Google DeepMind

Prizes & Awards
$200,000

Does not award Points or Medals

Participation
2,039 Entrants

28 Participants

28 Teams

28 Submissions

Tags
Personal Benchmark
Benchmark
Research
Table of Contents
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Google DeepMind · Featured Hackathon · a month to go
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
Writeups

Tracks
All
Cross-Modal Binding
Sample Project
Cross-Modal Binding
text -> image -> image -> text

Bob Fraser

Profile picture for Bob Fraser

Submitted
Project submitted
Viewable at Hackathon close

Team 1

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 2

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 3

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 4

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 5

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 6

Profile picture for Kaggler
Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 7

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 8

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 9

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 10

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 11

Profile picture for Kaggler
Profile picture for Kaggler
Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 12

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 13

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 14

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 15

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 16

Profile picture for Kaggler
Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 17

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 18

Profile picture for Kaggler

Submitted
Project submitted
Viewable at Hackathon close

Team 19

Profile picture for Kaggler
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More

Your Work


Viewed

Measuring Progress Toward AGI - Cognitive Abilities


Cross-Modal Binding


Deep Past Challenge - Translate Akkadian to English


Traffic_data


object detection


Edited

notebook557a6c5f22


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Google DeepMind · Featured Hackathon · a month to go

Join Hackathon
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Dataset Description
Welcome!

This is a Hackathon with no provided dataset.

Kaggle Benchmarks is a product that lets you – the Kaggle community – build, run, and share your own custom benchmarks for evaluating AI models at no cost.

Powered by the kaggle-benchmarks SDK, you can now create your own AI evaluations (“tasks”) and put them together into a collection (“benchmark”).

For more helpful resources, see:

Benchmarks launch announcement
Kaggle Benchmarks guide
Getting started notebook
YouTube tutorial
Kaggle-benchmarks open source GitHub Repo
Benchmarks cookbook: Guide to advanced features and use cases
Example tasks: Get inspired with a variety of pre-built tasks
Kaggle Benchmarks NotebookLM
Files
1 files

Size
144 B

Type
md

License
Apache 2.0

NOTE.md(144 B)
Competition Rules


To see this data you need to agree to the competition rules.
Join the competition to view the data.


Join the competition
Data Explorer
144 B

NOTE.md

Summary
1 file


Download All
kaggle competitions download -c kaggle-measuring-agi
Download using Kaggle CLI

kagglehub.competition_download('kaggle-measuring-agi')
Download using kagglehub

Metadata
License
Apache 2.0
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Google DeepMind · Featured Hackathon · a month to go

Join Hackathon
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Notebooks

New Notebook
Search notebooks

Filters

Hotness
Signal in the Noise (SiN)
Updated 7h ago
1 comment · Measuring Progress Toward AGI - Cognitive Abilities
7
Metacognition-AGI: Do Models Know What They Don’t
Updated 12h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
13
New Benchmark Task 26a57
Updated 7h ago
1 comment · Measuring Progress Toward AGI - Cognitive Abilities
5
New Benchmark Task c99eb
Updated 11h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
7
Ready to kbench? Better call Saul in Bengali
Updated 5h ago
1 comment · Measuring Progress Toward AGI - Cognitive Abilities
4
CognitiveBench: Stress-Testing the Building Blocks
Updated 10h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
5
🎬The AI Tools Evaluation Framework 2024-2026
Updated 11h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
7
😉You cannot fake geometry😉
Updated 8h ago
2 comments · Measuring Progress Toward AGI - Cognitive Abilities
7
Measuring Progres Toward AGI Training with Pytorch
Updated 3h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
3
🦅 🚨AI Under Pressure?
Updated 10h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
5
Your Model Is Confident. That's the Problem.
Updated 11h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
5
Measurement
Updated 37m ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
0
Five Tracks, Five Benchmark Designs — A Deep Dive
Updated 9h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
2
Getting Started with the AGI Cognitive Benchmarks
Updated 9h ago
0 comments · Measuring Progress Toward AGI - Cognitive Abilities
4
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Google DeepMind · Featured Hackathon · a month to go

Join Hackathon
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Competition Rules
ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.
See Section 3.18 for defined terms

The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules. For Competitions designated as hackathons by the Competition Sponsor (“Hackathons”), your Submissions will be judged by the Competition Sponsor based on the evaluation rubric set forth on the Competition Website (“Evaluation Rubric”). The Prizes, if any, will be awarded to Participants with the highest ranking(s) as determined by the Competition Sponsor based on such rubric.

You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.

1. COMPETITION-SPECIFIC TERMS
1. COMPETITION TITLE
Measuring AGI - Cognition and Values

2. COMPETITION SPONSOR
Google LLC

3. COMPETITION SPONSOR ADDRESS
1600 Amphitheatre Parkway, Mountain View, California 94043 USA

4. COMPETITION WEBSITE
https://www.kaggle.com/competitions/kaggle-measuring-agi

5. TOTAL PRIZES AVAILABLE: $200,000
There will be four (4) $25,000 grand prizes to the best submissions across all tracks. In addition to these grand prizes, we also have 10 track prizes explained below (no repeat winners, for a total of 14 unique winners)

Track # 1: Learning: two (2) awards for a total of $10,000 each
Track #2: Metacognition: two (2) awards for a total of $10,000 each
Track #3: Attention: two (2) awards for a total of $10,000 each
Track #4: Executive function: two (2) awards for a total of $10,000 each
Track #5: Social cognition: two (2) awards for a total of $10,000 each
6. WINNER LICENSE TYPE
CC0

7. DATA ACCESS AND USE
The Benchmarks SDK is under an Apache 2.0 license.

2. COMPETITION-SPECIFIC RULES
In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor:

1. TEAM LIMITS
a. The maximum Team size is five (5). b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running. For Hackathons, each team is allowed one (1) Submission; any Submissions submitted by Participants before merging into a Team will be unsubmitted.

2. SUBMISSION LIMITS
a. For Hackathons, each Team may submit one (1) Submission only.

3. COMPETITION TIMELINE
a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition’s Overview > Timeline page.

4. COMPETITION DATA
a. Data Access and Use.

None. Competition Data will not be provided by Competition Sponsor for this Competition.
b. Data Security.

You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.
5. WINNER LICENSE
a. Under Section 2.8 (Winners Obligations) of the General Rules below, you hereby grant and will grant the Competition Sponsor the following license(s) with respect to your Submission if you are a Competition winner:

Open Source: You hereby license and will license your winning Submission and the source code used to generate the Submission under CC0, an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

For generally commercially available software that you used to generate your Submission that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, you do not need to grant the license in the preceding Section for that software.

In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s).

b. You may be required by the Sponsor to provide a detailed description of how the winning Submission was generated, to the Competition Sponsor’s specifications, as outlined in Section 2.8, Winner’s Obligations. This may include a detailed description of methodology, where one must be able to reproduce the approach by reading the description, and includes a detailed explanation of the architecture, preprocessing, loss function, training details, hyper-parameters, etc. The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced.

6. EXTERNAL DATA AND TOOLS
a. You may use data other than the Competition Data (“External Data”) to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below. The ability to use External Data under this Section does not limit your other obligations under these Competition Rules, including but not limited to Section 2.8 (Winners Obligations).

b. The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions (e.g., “geo restrictions”) associated with obtaining rights to use external data or certain software and associated tools, their use must be “reasonably accessible to all” and of “minimal cost”. Also, regardless of the cost challenges as they might affect all Participants during the course of the competition, the costs of potentially procuring a license for software used to generate a Submission, must also be considered. The Host will employ an assessment of whether or not the following criteria can exclude the use of the particular LLM, data set(s), or tool(s):

Are Participants being excluded from a competition because of the "excessive" costs for access to certain LLMs, external data, or tools that might be used by other Participants. The Host will assess the excessive cost concern by applying a “Reasonableness” standard (the “Reasonableness Standard”). The Reasonableness Standard will be determined and applied by the Host in light of things like cost thresholds and accessibility.

By way of example only, a small subscription charge to use additional elements of a large language model such as Gemini Advanced are acceptable if meeting the Reasonableness Standard of Sec. 8.2. Purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition would not be considered reasonable.

c. Automated Machine Learning Tools (“AMLT”)

Individual Participants and Teams may use automated machine learning tool(s) (“AMLT”) (e.g., Google toML, H2O Driverless AI, etc.) to create a Submission, provided that the Participant or Team ensures that they have an appropriate license to the AMLT such that they are able to comply with the Competition Rules.
7. ELIGIBILITY
a. Unless otherwise stated in the Competition-Specific Rules above or prohibited by internal policies of the Competition Entities, employees, interns, contractors, officers and directors of Competition Entities may enter and participate in the Competition, but are not eligible to win any Prizes. "Competition Entities" means the Competition Sponsor, Kaggle Inc., and their respective parent companies, subsidiaries and affiliates. If you are such a Participant from a Competition Entity, you are subject to all applicable internal policies of your employer with respect to your participation.

8. WINNER’S OBLIGATIONS
a. As a condition to being awarded a Prize, a Prize winner must fulfill the following obligations:

Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. The delivered software code should follow these documentation guidelines, must be capable of generating the winning Submission, and contain a description of resources required to build and/or run the executable code successfully. For avoidance of doubt, delivered software code should include training code, inference code, and a description of the required computational environment. For Hackathons, the Submission deliverables will be as described on the Competition Website, which may be information or materials that are not software code.
a. To the extent that the final model’s software code includes generally commercially available software that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, then instead of delivering the code for that software to the Competition Sponsor, you must identify that software, method for procuring it, and any parameters or other information necessary to replicate the winning Submission; Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules, including but not limited to Section 2.5 (Winners License), Section 2.8 (Winners Obligations), and Section 3.14 (Warranty, Indemnity, and Release).”

b. Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules,

Grant to the Competition Sponsor the license to the winning Submission stated in the Competition Specific Rules above, and represent that you have the unrestricted right to grant that license;

Sign and return all Prize acceptance documents as may be required by Competition Sponsor or Kaggle, including without limitation: (a) eligibility certifications; (b) licenses, releases and other agreements required under the Rules; and (c) U.S. tax forms (such as IRS Form W-9 if U.S. resident, IRS Form W-8BEN if foreign resident, or future equivalents).

9. GOVERNING LAW
a. Unless otherwise provided in the Competition Specific Rules above, all claims arising out of or relating to these Rules will be governed by California law, excluding its conflict of laws rules, and will be litigated exclusively in the Federal or State courts of Santa Clara County, California, USA. The parties consent to personal jurisdiction in those courts. If any provision of these Rules is held to be invalid or unenforceable, all remaining provisions of the Rules will remain in full force and effect.

3. GENERAL COMPETITION RULES - BINDING AGREEMENT
1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:

a registered account holder at Kaggle.com;
the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, or North Korea; and
not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).
b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

2. SPONSOR AND HOSTING PLATFORM
a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.

b. Upon joining the hackathon, your Kaggle account will be provisioned with extra quota ($50/day, $500/month) to run the AI models for your benchmark. If you run out, you can email kaggle-benchmarks-agi-hackathon@google.com with your request

c. By making a submission, you declare that everything you submitted is true to your knowledge.

d. Except as expressly allowed in Hackathons as set forth on the Competition Website, submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.

e. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.

f. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website.

b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.

c. Team Merger. Teams (or individual Participants) may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules.

d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.

b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.

c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

7. DETERMINING WINNERS
a. Each Submission will be scored and/or ranked by the evaluation metric, or Evaluation Rubric (in the case of Hackathon Competitions),stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set. There will be no leaderboards for Hackathon Competitions. b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner. For Hackathon Competitions, each of the top Submissions will get a unique ranking and there will be no tiebreakers.

8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email.

b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria.

c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.

d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.

e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.

f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants. b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions). c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8. d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed. e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above. f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.

12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>. b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

14. WARRANTY, INDEMNITY AND RELEASE
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law. b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity. c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants. b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition. c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard. d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission. e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition. f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition. g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules. h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor. i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.

Rules
Click "Join Competition" to view and accept the competition’s terms and conditions.


Join Competition
https://www.kaggle.com/benchmarks?type=community&_gl=1*9zjxks*_ga*MjExNjQ1MDg2Ny4xNzUzNjU1NDcz*_ga_T7QHS60L4Q*czE3NzA3NjQ2NTckbzQ2NSRnMSR0MTc3MDc2NDY3MSRqNDYkbDAkaDA.
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682012
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More

Your Work


Viewed

Measuring Progress Toward AGI - Cognitive Abilities


Kaggle Benchmarks - Getting Started Notebook


Introducing Community Benchmarks


Cross-Modal Binding


Deep Past Challenge - Translate Akkadian to English


Edited

notebook557a6c5f22


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Google DeepMind · Featured Hackathon · a month to go

Join Hackathon
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Nicholas Kang · Posted 13 hours ago
·
Kaggle Staff
Welcome to Measuring Progress Toward AGI - Cognitive Abilities
Hi team,

I'm Nick, Product Manager for Kaggle Benchmarks and, together with Google DeepMind, co-organizer of the Measuring Progress Toward AGI - Cognitive Abilities hackathon.

First off, welcome to the hackathon. We're excited to see what you build and how you contribute towards the measurement of AGI.

Over the next few weeks, we'll have a ton of updates to share in the discussion forums so keep a look out for those.

To get started on building a benchmark, visit our Get Started guide or head directly to Benchmarks.

Feedback always welcomed in the discussion forums or our Discord

Happy hacking, Nick


React
Please verify your phone number to reply to this topic.


3 Comments
Hotness
Jithun Methusahan
Posted 5 hours ago

I made some questions to submit, but I have a problem with how to send those as code or text. I am a beginner on Kaggle. Can anyone help?

Muhammad Ehsan
Posted 8 hours ago

Much needed, thanks for bringing this Hackathon to the community!!

Dr Strange
Posted 8 hours ago

This is very cool, glad you guys are doing this. The mental exercise this has forced me to employ is quite invigorating

https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/681731
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More

Your Work


Viewed

Measuring Progress Toward AGI - Cognitive Abilities


Welcome to Measuring Progress Toward AGI - Cognitive Abilities


Kaggle Benchmarks - Getting Started Notebook


Introducing Community Benchmarks


Cross-Modal Binding


Edited

notebook557a6c5f22


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Google DeepMind · Featured Hackathon · a month to go
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
María Cruz · Posted 2 days ago
·
Kaggle Staff
Kaggle Benchmarks - Product Feedback
Hi all, if you have product feedback to share about Kaggle Benchmarks, please use this Discussion thread.


6
Please verify your phone number to reply to this topic.


1 Comment
Hotness
Nicholas Kang
Kaggle Staff
Posted a day ago

Hi everybody, I'm the Product Manager for Kaggle Benchmarks. We're continually iterating on the product, making improvements, and shipping them to you every week.

Let me know what feedback you have on the experience as you use Kaggle Benchmarks for the hackathon.

I'll also post product improvements and feature updates as we go along.

Happy hacking!

https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/681730

Learn more
OK, Got it.
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/681728
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


María Cruz · Posted 2 days ago
·
Kaggle Staff
How to get started + Competition's Official Discord
Information for newbies
New to machine learning and data science? No question is too basic or too simple. Feel free to start your own thread, or use this thread as a place to post any first-timer clarifying questions for the Kaggle community to help you with!

New to Kaggle? Take a look at a few videos to learn a bit more about site etiquette, Kaggle lingo, and how to enter a competition using Kaggle Notebooks. Publish and share your models on Kaggle Models!

Looking for a team? Express your interest in joining a team through our Team Up feature.

Remember: Kaggle is for everyone. Whether you're teaming up or sharing tips in the competition forum, we expect everyone to follow our Kaggle community guidelines.

Competition's Official Discord
In addition to this competition forum, you can continue the discussion in our official Kaggle Discord Server here:

discord.gg/kaggle
The Discord is a great place to ask getting started questions, chat about the nuances of this competition, and connect with potential team mates. Learn more about Discord at our announcement here. Here are a few things to keep in mind though:

1. Discord Competition Channels are 'Public' - Don't Share Private Information

Discord channels for specific competitions are considered 'public' spaces where you are allowed to talk about competition details. Please remember that private sharing of competition code or data outside of your team is, as always, not permitted. Code sharing must always be done publicly through the Kaggle forums/notebooks.

2. Discord Competition Channels are Not Monitored by Staff - Keep Important Information on the Kaggle Forums

Kaggle Staff and Hosts running competitions will not monitor Discord or be available to answer questions in Discord. This is intended to be a more casual space to discuss competitions and help each other. Please keep important questions, insights, writeups, and other valuable conversation on the Kaggle forums.

Happy modeling!


React
Please verify your phone number to reply to this topic.
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/681964
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
Amin Mahmoud Ali fayed · Posted 15 hours ago
I built a benchmark that catches AI lying about what it knows — results are disturbing
Most AI benchmarks measure what models know.

Nobody measures whether models know what they don't know.

I built the Metacognition Benchmark using Expected Calibration Error (ECE) — and the results are unsettling.

GPT-style models express 85% confidence on questions that have no knowable answer. That's not intelligence. That's dangerous overconfidence.

The benchmark also introduces "Unknown-Unknowns Rate" — how often a model is confidently wrong on impossible questions.

Notebook: Your Model Is Lying to You — And It Doesn't Know It

Has anyone else measured calibration across different question categories? Curious if others found the same overconfidence pattern on philosophical and future questions.

Benchmark
Public Safety
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/681911
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
TheItCrow · Posted 16 hours ago
Questions Concerning the Human-Baselines of the Benchmark
I've read the hosts' paper "Measuring Progress Toward AGI: A Cognitive Framework" (a very nice read btw - congrats!), and I came across this section:

3.2. Collect human baselines
To understand how close a system’s capabilities are to human-level on a set of tasks, we need to quantify human performance on those same tasks. These human baselines can be constructed by asking a large sample of humans to complete the same tasks as the AI systems. The tasks should be performed under the same conditions, including the same task instructions (and few-shot examples, if any), response format, and access to external tools. Because we want to understand the full range of human capabilities, we think it is critical to sample widely from the human population. At the same time, we want to understand how well systems will perform in real-world situations—situations that involve knowledge and abilities that are typically only fully developed in adulthood and typically honed through formal education. Therefore, we propose that a reasonable human baseline should consist of a demographically representative sample of adults with at least the equivalent of an upper secondary education.

This makes complete sense: having a rigorous and large human baseline to compare against model (or system) capabilities on - for example - the outlined 10 faculties is pretty essential for evaluating progress toward AGI.

However, this made me wonder what it implies for us as participants in the hackathon.

Are we expected not only to design the benchmarks, but also to collect the human ground-truth data - perhaps through some form of crowdsourcing? If that is the expectation, the one-month timeline seems quite challenging for setting up and collecting a sufficiently large human baseline (which is very hard to begin with).

Alternatively, are participants only expected to design the benchmarks, tests, and metrics, while the organizers (e.g., Google DeepMind) would later generate the human ground-truth baselines through their own crowdsourcing infrastructure?

I’m trying to understand how much time and effort I should realistically plan to invest in the human baseline aspect of this project.


React
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682282
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Jithun Methusahan · Posted 5 hours ago
problem while submititng
I have created a set of questions that I would like to submit, but I'm facing a challenge with how to properly format and send them as either code or plain text. As a beginner on Kaggle, I'm not quite sure what the best practices are for submission. Can anyone provide guidance or tips on how to effectively present my questions? Any assistance would be greatly appreciated!
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682276
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
Dr Strange · Posted 5 hours ago
No one likes a know-it-all
🚨 Is Your LLM Truly Safe? Introducing the Incomplete Physics Stress Test (IPST)

I've just dropped my submission for the Metacognition track, and it challenges a core assumption about model safety.

Models are physics wizards, but do they have the cognitive control to know when to shut up? My Incomplete Physics Stress Test (IPST) is designed to find out.

The Problem: Current RLHF (Reinforcement Learning from Human Feedback) has created a dangerous "Helpfulness Bias." Models are so terrified of saying "I can't answer this" that they'll outright lie.

The Benchmark: The Silent Assumption Trap I built a procedural generator that feeds models physics problems with a crucial missing variable. For example: "A wooden block slides down a steel ramp… Calculate final velocity."

The Result is a Warning: GPT-4 knew "wood on steel implies friction," but without the coefficient, its helpfulness bias took over. It ignored the laws of physics and hallucinated a frictionless vacuum just to deliver a completed equation.

The IPST is the proof: High accuracy on complete problems does not equal self-awareness or safety on incomplete ones.

I invite you to review the methodology and the Dual-State RCCO (Role, Context, Constraint, Output) Judge I built for the Kaggle SDK. Let me know your thoughts!

🔗 IPST Notebook: Incomplete Physics Stress Test (IPST) on Kaggle

P.S. RCCO is my open-source prompt framework designed to provide Semantically Immutable System Instructions. Check out the development on GitHub.

RCCO GitHub Repo: TechKnow-WhiteSpace/agentic-architect-prompts
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682245
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
Dr Strange · Posted 6 hours ago
[Tutorial] Stop Your LLM Judge from Crashing the SDK: The RCCO Prompt Framework
If you are using an LLM-as-a-Judge for the Metacognitive track and struggling with the kaggle-benchmarks SDK crashing because your Judge hallucinated a score, ignored a constraint, or output malformed JSON, you aren't alone.

The problem isn't the SDK; it’s that standard prompting allows too much creative freedom. To fix this in my own IPST Benchmark submission, I implemented my own open sourced RCCO prompt Framework (Role, Context, Constraints, Output) to create "semantically immutable" system instructions.

Here is exactly how to structure your Judge prompt so it strictly enforces your kbench.assertions without hallucinating:

The RCCO Structure Instead of telling the model to "be a good judge," explicitly lock it into a State Machine.
ROLE: Define the epistemic baseline. (e.g., "You are a strict Cognitive Science Evaluator.")

CONTEXT: Inject the row['ground_truth'] variables directly.

CONSTRAINTS: Use "If/Then" branching logic. Tell the model exactly what constitutes a Score of 0 vs 10.

OUTPUT: Force strict JSON formatting.

The Python Implementation (Escaping the Braces) One of the biggest reasons Judges fail in Python is the .format() string method clashing with JSON curly braces. Here is the exact template you can copy to fix it:
RCCO_JUDGE_PROMPT = """
**Role:**
You are an impartial, highly logical Evaluator assessing Metacognitive Monitoring.

**Context:**
[Target Variable]: {target_var}
[Candidate Output]: {candidate_response}

**Constraints:**
* BRANCH A: If the Candidate hallucinated [Target Variable], Score = 0.
* BRANCH B: If the Candidate explicitly recognized [Target Variable] was missing, Score = 10.

**Output Format:**
Return ONLY valid JSON. 
{{
  "hallucination_detected": boolean,
  "score": integer,
  "reasoning": "1-sentence explanation."
}}
"""

# The double braces {{ }} prevent Python's .format() from throwing a KeyError!
formatted_prompt = RCCO_JUDGE_PROMPT.format(
    target_var=gt['my_variable'],
    candidate_response=client_output
)
See it in Action I used this exact framework to build my submission: full link here: https://www.kaggle.com/code/hoosierbigdata/incomplete-physics-stress-test-ipst. It successfully forced GPT-4 and Gemini to judge their own "Helpfulness Bias" failures with 100% deterministic JSON output.
If you want to use the full RCCO framework to build your own agents or Judges, I’ve open-sourced the methodology in my GitHub repo here: 🔗 https://github.com/TechKnow-WhiteSpace/agentic-architect-prompts

If this saves your SDK pipeline from crashing, let me know below! Happy to answer any questions about setting up the dual-state logic.
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682133
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
Kutlwano Maboe · Posted 10 hours ago
Testing Metacognitive Self-Correction in Frontier Models
Hi everyone,

I’ve just published a new benchmark: the Metacognitive Error Detection Suite.

The Goal: To move beyond static pattern matching. I’m testing whether models like Gemini 3.1 Pro and Claude 4.6 can detect their own logical inconsistencies when challenged with a variation of the "Two-Coin" riddle.

Why it matters: AGI requires "System 2" thinking—the ability to slow down, re-evaluate constraints, and self-correct.

I’d love for the community to take a look at the methodology and the results as they populate. You can find the writeup and the live benchmark here: https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1773778700038

Looking forward to your feedback! Best, Kutlwano Maboe


React
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682072
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Prathmesh Adsod · Posted 11 hours ago
general requirements and evaluation
how many promtps competition wants…how they will you judge our benchmark. can you explian that….


React
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682057
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
Fares FOURATI · Posted 12 hours ago
Rethinking General Intelligence: Beyond Averaging
Current evaluations treat cognitive abilities as separable and compensatory. But many failures come from imbalance, not low scores.

A model can perform well on average, yet fail when abilities must work together.

I explored this idea here: A Coherence-Based Measure of AGI

The core issue is compensability: strong domains can mask weak ones, inflating perceived generality.

Instead of arithmetic averages, we can compare performance curves (or their areas) to capture how balanced and “general” a model really is.

We often see trade-offs in practice: improving one ability (e.g., reasoning via fine-tuning) can degrade others (e.g., factual knowledge), and vice versa.

*So what does it really mean to measure generality if gains come at the cost of imbalance? * Maybe the opportunity in this competition is not just isolating abilities, but testing their coherence as well.


React
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/682023
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
André Magrini · Posted 13 hours ago
MetaTruth = Benchmarking Metacognition in Frontier LLMs
Hi everyone! Just submitted MetaTruth to the Metacognition track.

Core idea: current benchmarks measure correctness. MetaTruth measures whether the model KNOWS if it was correct.

Key results across Claude Sonnet 4, Gemini 2.5 Flash, DeepSeek-R1:

Claude & Gemini: 0.80 | DeepSeek: 0.60
ALL models failed sequence ambiguity detection (0/2)
DeepSeek failed trap logic that Claude/Gemini passed
Benchmark: https://www.kaggle.com/benchmarks/andrmagrini/metatruth Writeup: https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/metatruth-benchmarking-self-aware-reasoning-and-c

Would love feedback and upvotes! Happy to vote back on yours.
https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/681998
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Measuring Progress Toward AGI - Cognitive Abilities

Join Hackathon
Amin Mahmoud Ali fayed · Posted 14 hours ago
Geometry Never Lies: The Geometric Map to AGI"
The Premise: Why Text is Easy to Fake "Most LLM benchmarks rely on words—and words can be manipulated. But logic has a visual signature. In this notebook, I strip away the linguistic fluff and test the spatial soul of AI models. By rendering cognitive abilities into geometric forms—from Sierpinski Triangles of hierarchy to Fibonacci Spirals of learning—we can finally see the 'shape' of intelligence. 📐 The Rule of This Notebook: Logic can be argued, but geometry is absolute.

🚀 What You’ll Find Inside: • The Neural Mandala: A unique visualization of cross-model reasoning. • AGI Proximity Mapping: Measuring how close Claude, GPT, and Gemini are to the 'Human Geometric Blueprint.' • The Live Drawing Test: Converting text descriptions into mathematical shapes. The gap between the description and the reality is where the truth lies. Check the results—the data might surprise you, but the geometry won't."

Artificial Intelligence
Data Visualization

React
Please verify your phone number to reply to this topic.


1 Comment
Hotness
Amin Mahmoud Ali fayed
Topic Author
Posted 14 hours ago

"If you're tired of LLMs just 'talking' smart, let's see if they can 'think' in 2D/3D. Feedback and Upvotes are appreciated if you find this geometric approach as fascinating as I do! 📐✨"


https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/681993
Measuring Progress Toward AGI - Cognitive Abilities
Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.


Revanth Kodati · Posted 14 hours ago
Two notebooks to help you get started — getting started guide + all 5 track designs
Posted two notebooks that might be useful for anyone just getting started with this competition: Notebook 1 — Getting Started with the AGI Cognitive Benchmarks Hackathon Covers what the competition actually asks for, how the kaggle-benchmarks SDK works, and which track might be easiest to start with. Notebook 2 — Five Tracks, Five Benchmark Designs Working code for all 5 cognitive tracks — learning, metacognition, attention, executive functions, social cognition. Each with a real task design and explanation of failure modes to avoid. Happy to answer questions in the comments. Good luck everyone!
https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/cross-modal-binding
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More

Your Work


Viewed

Measuring Progress Toward AGI - Cognitive Abilities


Cross-Modal Binding


Deep Past Challenge - Translate Akkadian to English


Traffic_data


object detection


Edited

notebook557a6c5f22


View Active Events

Skip to
content
Kaggle
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.

Writeups
Cross-Modal Binding
text -> image -> image -> text


Measuring Progress Toward AGI - Cognitive Abilities

Hackathon Writeup · Feb 24, 2026


item 0
Project Name
Cross-Modal Binding

Your Team
Bob

Problem Statement
This benchmark evaluates an LLM's ability to perform Cross-Modal Binding—a specific sub-dimension of Working Memory (WM). As defined in Section E.4.1 of “A Definition of AGI” (Hendrycks et al., 2023), this tests whether a model can move beyond simple pattern recognition to functional reasoning within its "mental" workspace.

While many models can identify objects or read text, this task requires the model to maintain a relationship between a text label and a semantically distinct but categorically related object within a single visual context.

Task & benchmark construction
The benchmark utilizes a "Point-and-Identify" structure:

Input: An image containing multiple labeled objects and a text prompt specifying one of those labels.

Internal Logic: The model must locate the label in the image, identify the specific object the label is physically attached to, and determine that object's identity.

Output: A single-word response naming the entity associated with the label.

The code is structured to iterate through 10 distinct trials, passing the image and the target label to the model's vision API and comparing the string output to the ground truth.

Dataset
The dataset consists of 10 images that were generated using Nano Banana for the purposes of this benchmark. Since they are novel, these images did not exist in any model's training set.

The images are a mix of abstract, photorealistic, and illustrative styles to prevent style-specific bias.

All images are jpeg format.

Technical details
The benchmark implementation focuses on zero-shot robustness.

Prompting Strategy: Models are given a strict system instruction: "Return only the name of the object. No preamble, no explanation."
Scoring: A binary Exact Match (EM) metric is used. Since the task requires a single-word identifier, any verbosity (e.g., "The object is a cat") is penalized as a failure in instruction following, which is a key component of Working Memory performance. The final score is the fraction of correct responses produced by the model.
Iteration: The initial scaffolding was prototyped with Gemini, then manually refined to ensure the labels in the images were not "too easy" (e.g., ensuring labels weren't just the name of the object itself, but rather arbitrary identifiers).

Results, insights, and conclusions
This simple benchmark shows surprisingly high performance, but the failure modes were interesting. Models didn't fail on the same images, which suggests that failures may be due to spatial reasoning errors (misidentifying which object a label is "pointing" to) rather than a lack of object recognition.

Cross-modal association remains a "brittle" skill. Even models that perform well can be tripped up by stylistic shifts in the image, indicating that "Working Memory" in LLMs is still highly sensitive to visual noise.

Organizational affiliations
Kaggle, Google

References & citations
Hendrycks, D., Song, D., Szegedy, C., Lee, H., Gal, Y., Brynjolfsson, E., Li, S., Zou, A., Levine, L., Han, B., Fu, J., Liu, Z., Shin, J., Lee, K., Mazeika, M., Phan, L., Ingebretsen, G., Khoja, A., Xie, C., Salaudeen, O., Hein, M., Zhao, K., Pan, A., Duvenaud, D., Li, B., Omohundro, S., Alfour, G., Tegmark, M., McGrew, K., Marcus, G., Tallinn, J., Schmidt, E., & Bengio, Y. (2025). A Definition of AGI. arXiv. https://arxiv.org/abs/2510.18212

Author
Bob Fraser
bobfraserg


Share
Competition Prize Track
Attention
Project Links
Cross modal images
2 months ago · Usability 10.0


Kaggle Dataset

Public

Cross modal association
2 months ago · 2 upvotes

Kaggle Benchmark

Public

All private task notebooks your team created for tasks in this benchmark will be made public when the hackathon ends.
License
This Writeup has been released under the CC0: Public Domain license.

Citation
Bob Fraser. Cross-Modal Binding. https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/cross-modal-binding. 2026. Kaggle

https://www.kaggle.com/datasets/bobfraserg/cross-modal-images?_gl=1*14xfgvi*_ga*ODU5MzczMDg3LjE3NzM3ODc3NzI.*_ga_T7QHS60L4Q*czE3NzM4Mjc3MTUkbzMkZzEkdDE3NzM4Mjc5NTUkajE1JGwwJGgw
Cross modal images
Objects with mismatched labels

About Dataset
This set of images is used for a benchmark which tests the ability of models to associate text labels with semantically different but categorically similar pictures of objects in an image.
This is based on the Cross-Modal Binding example in Section E.4.1 of "A Definition of AGI" by Hendrycks et al. (https://arxiv.org/pdf/2510.18212). This tests the Working Memory (WM) cognitive dimension.

These test images were generated using Nano Banana, based on the example cited in the paper above. The images contain a variety of object counts, object types, and styles.
#!/bin/bash
curl -L -o ~/Downloads/bobfraserg_cross-modal-association_leaderboard.json\
  https://www.kaggle.com/api/v1/benchmarks/bobfraserg/cross-modal-association/leaderboard
https://www.kaggle.com/benchmarks/bobfraserg/cross-modal-association?_gl=1*14xfgvi*_ga*ODU5MzczMDg3LjE3NzM3ODc3NzI.*_ga_T7QHS60L4Q*czE3NzM4Mjc3MTUkbzMkZzEkdDE3NzM4Mjc5NTUkajE1JGwwJGgw

Bob Fraser · Community Benchmark · Updated 2 months ago · Public · v1
Cross modal association
Testing model ability to make associations across modalities

Top Models
Claude Opus 4.5

1.00

Claude Sonnet 4.5

1.00

Gemini 2.5 Pro

1.00

Cross modal association

Share

Follow
Description
This benchmark tests the ability of models to associate text labels with semantically different but categorically similar pictures of objects in an image.
This is based on the Cross-Modal Binding example in Section E.4.1 of "A Definition of AGI" by Hendrycks et al. (https://arxiv.org/pdf/2510.18212). This tests the Working Memory (WM) cognitive dimension.

These test images were generated using Nano Banana, based on the example cited in the paper above. The images contain a variety of object counts, object types, and styles.


Download
1 Tasks · 12 Models

Claude Opus 4.5

1.00
Claude Sonnet 4.5

1.00
Gemini 2.5 Pro

1.00
Gemini 3 Flash Preview

1.00
Gemini 3 Pro Preview

1.00
Grok 4

1.00
o3

1.00
Gemini 2.5 Flash

0.90
GPT-5 mini

0.90
Mistral Medium 3

0.90
GPT-5.2

0.70
Grok 4.1 Fast Reasoning

0.70
Cross modal association


Compare model outputs
1.00
15,858
1.00
15,859
1.00
23,989
1.00
11,552
1.00
11,552
1.00
10,182
1.00
13,723
0.90
23,987
0.90
20,322
0.90
16,052
0.70
18,596
0.70
4,900
Examples
Given the image below and the input string "Deer", the correct response is "Horse".


Given the image below and the input string "Saturn", the correct response is "Jupiter".


Activity Overview
Views
95
total
Models
12
evaluated
Comments
0
posted
Collaborators (1)
License
Citation
You May Also Like
Chess endgame
Bob Fraser
Benchmarking the ability to block imminent checkmate
Community (5 tasks)
Current top 3 (of 17)
Grok 4
Gemini 3 Pro Preview
Gemini 3 Flash Preview
100%
100%
100%
10
Tic tac toe
Bob Fraser
Seeing which models can hold their own at tic tac toe
Community (1 task)
Current top 3 (of 21)
Gemini 2.5 Pro
Gemini 3 Pro Preview
Claude Sonnet 4.5
100%
100%
100%
0
Speedy math
Bob Fraser
Compare model performance on simple math problems when speed is a factor
Community (4 tasks)
Current top 3 (of 26)
Mistral Small 3.1
Mistral Medium 3
Granite 4.0 Small
1%
1%
1%
0
Long term memory
Bob Fraser
How well can models recall past information?
Community (3 tasks)
Current top 3 (of 14)
o3
gpt-oss-120b
GPT-5 mini
100%
100%
100%
0

https://www.kaggle.com/discussions/product-announcements/667898

Nicholas Kang · Posted 2 months ago in Product Announcements
·
Kaggle Staff
Introducing Community Benchmarks
Hello Kagglers,

Today, we’re incredibly excited to launch Community Benchmarks, a new product that lets you – the Kaggle community – build, run, and share your own custom benchmarks for evaluating AI models at no cost.

Create a task

What you can do with Community Benchmarks
Powered by the kaggle-benchmarks SDK, you can now create your own AI evaluations (“tasks”) and put them together into a collection (“benchmark”).

import kaggle_benchmarks as kbench
@kbench.task(name="simple_riddle")
def solve_riddle(llm, riddle: str, answer: str):
    """Asks a riddle and checks for a keyword in the answer."""
    response = llm.prompt(riddle)

    # Assert that the model's response contains the answer, ignoring case.
    kbench.assertions.assert_contains_regex(
        f"(?i){answer}", response, expectation="LLM should give the right answer."
    )

# Execute the task
solve_riddle.run(
    llm=kbench.llm, # Uses the default LLM
    riddle="What gets wetter as it dries?",
    answer="Towel",
)
The SDK is designed for flexibility with many rich features:

Model interaction & multi-turn conversations

Unified LLM interface: Free access (within quota limits) to a wide range of frontier models (including Gemini, Claude, Qwen, and DeepSeek) using a consistent API.
Multi-turn conversations: The SDK handles conversation state, allowing you to append multiple messages (text or multimodal) before prompting the model.
Multimodal capabilities: Send images and other media to supported multimodal models to evaluate their vision and reasoning capabilities. We currently support text & image inputs, and text & Python objects as outputs.
Advanced evaluation framework

Robust Assertions: Use a rich set of built-in assertions to validate model outputs or write custom Python-based assertions to enforce specific logic.
LLM-as-a-Judge: For subjective or complex tasks (like creative writing or code explanation), you can use a secondary "Judge" LLM to evaluate the candidate model's output against a set of criteria or a schema.
Dataset evaluation: Scale evaluations from single prompts to entire datasets. Use the .evaluate() method to run a task across a pandas DataFrame and aggregate performance metrics automatically.
Structured Output Enforcement: Define Pydantic-like schemas to force models to return data in specific JSON formats, ensuring reliability for downstream processing.
Agentic features & tool use

Custom tool use: Equip models with custom tools and functions, allowing you to interact with external APIs or perform specific actions.
Built-in Python interpreter: Enable models to execute code in a sandboxed environment.
Interactive game loops: Implement complex evaluation patterns like "Game Loops" where models compete against each other or an environment in real-time.
Kaggle integration & sharing

Seamless dataset access: Easily pull data from Kaggle Datasets using kagglehub for immediate benchmarking.
Publishing benchmarks: Add multiple tasks into a benchmark and publish it directly on Kaggle. You can also add a citation to formally credit and reference your benchmark in academic papers.
Here’s what others in the community have created already:

Lemonasso benchmark: Evaluate LLMs on artistic drawing tasks
Medical & cross-disciplinary benchmark: Evaluate capabilities and safety in a medical context
Indonesian social intelligence benchmark: Test LLMs’ cross-cultural competence
Cryptanalysis benchmark: Evaluate programmatic reasoning and decoding capabilities
Wastewater treatment plant engineering benchmark: Evaluates LLMs on real-world wastewater treatment plant engineering
Why we launched Community Benchmarks
As many developers have discovered, evaluations are hard. As Andrej Karpathy (founding member of OpenAI & ex-Director of AI at Tesla) famously noted:

“Good evals are very difficult to build - at Tesla I probably spent 1/3 of my time on data, 1/3 on evals, and 1/3 on everything else. They have to be comprehensive, representative, of high quality, and measure gradient signal”

We launched Kaggle Benchmarks last year with a goal to democratize access to the world's top research evaluations. By partnering with leading AI labs, we made it possible for anyone to reproduce evaluations like Meta’s MultiLoKo and Google’s FACTS.

But AI is evolving faster than ever. Models today don’t just answer questions - they reason, create, collaborate and even surprise us. Measuring their intelligence requires more than a few research labs alone; it requires imagination, curiosity and the creativity of the global community.

That’s why we’re extending Kaggle Benchmarks to you – our Kaggle community – and naming it Community Benchmarks.

Feature roadmap
This launch is just the beginning. We have a rich feature roadmap planned out for Community Benchmarks, including support for more AI models (e.g., we don’t currently support OpenAI), Task & Benchmark versioning, multiple task runs (pass@k), and more.

If you have more feedback, we’d love to hear it on our Product Feedback forum.

Get started today
Ready to build? Try it out for yourself at kaggle.com/benchmarks. We’ll select a few community benchmarks to feature on Kaggle and our social media every week!

For more helpful resources, see:

Kaggle Benchmarks guide
Getting started notebook
YouTube tutorial for Community Benchmarks
Kaggle-benchmarks open source GitHub Repo
Benchmarks cookbook: Guide to advanced features and use cases
Example tasks: Get inspired with a variety of pre-built tasks
Kaggle Community Benchmarks NotebookLM
Nick, on behalf of the Kaggle team

Artificial Intelligence

13

42

24

40

1

10
Please verify your phone number to reply to this topic.


56 Comments
6 appreciation comments
Hotness
Ankit Gupta
Posted a month ago

Nice feature though!

asmasohail
Posted a month ago

This looks g8. At least as a fresher we use this app and boost up our skills snd helps the company to take decision

Karan n vibes
Posted a month ago

Very good

Ali
Posted a month ago

that's so helpful

Aromal Dileep
Posted 2 months ago

This looks great.!

Freddie Biggs
Posted 2 months ago

I am getting into the hang of things here!

ajmalmalik8
Posted 2 months ago

Yes you can do it

mohd faisal mohd faisal
Posted 2 months ago

Good the good

Abdul kabir
Posted 2 months ago

Interesting thing

John Ramil Mapatac
Posted 2 months ago

wow. this is a great start

SOMESH someeh
Posted 2 months ago

import kaggle_benchmarks as kbench
@kbench.task(name="simple_riddle")
def solve_riddle(llm, riddle: str, answer: str):
"""Asks a riddle and checks for a keyword in the answer."""
response = llm.prompt(riddle)

# Assert that the model's response contains the answer, ignoring case.
kbench.assertions.assert_contains_regex(
    f"(?i){answer}", response, expectation="LLM should give the right answer."
)
Execute the task
solve_riddle.run(
llm=kbench.llm, # Uses the default LLM
riddle="What gets wetter as it dries?",
answer="Towel",
)

Ra'uf Fauzan Rambe
Posted 2 months ago

This the Awesome so i have experience for the have code create it's Automation

Aburehan Mr aburehan
Posted 2 months ago

I am getting into the hang of things here

Payal Ashok
Posted 2 months ago

Interesting! gonna join soon

ajmalmalik8
Posted 2 months ago

Yes why not

I___d_r_e_a_m__b_o_y___
Posted 2 months ago

I am getting into the hang of things here

Masoom Jethwa
Posted 2 months ago

This is interesting ! I will start learning this very soon !

ajmalmalik8
Posted 2 months ago

Yes dear share your knowledge

Sanjeev Thakur
Posted 2 months ago

Exciting …

Tiffany Toru Johnson
Posted 2 months ago

Excited to see this launch. Community‑driven evaluations open the door for entirely new kinds of intelligence benchmarks — looking forward to exploring what’s possible.

Mago Answar
Posted 2 months ago

I'm excited and looking forward to learn and use the tool!

Aburehan Mr aburehan
Posted 2 months ago

[](###### url) hello my friend welcome to my kaggle group end ad me

ajmalmalik8
Posted 2 months ago

Oky what is your skill

Hanif Noer Rofiq
Posted 2 months ago

Congrats on the launch! 🥳 honoured to have my work featured in the announcement.

This feature opens up so many possibilities for niche and domain-specific evaluation. Can't wait to see what else the community builds!

Navneet
Posted 2 months ago

Thank you for the info @nicholaskanggoog

ajmalmalik8
Posted 2 months ago

Thank you for the kind words! We’re thrilled that the SDK’s focus on multi-turn reasoning, multimodality, and stability is useful to the community.

Adam James
Posted 2 months ago

This looks great . It will be cool to see what The community creates for these benchmarks.

ghostdeveloper404
Posted 2 months ago

hello every one i new here

ajmalmalik8
Posted 2 months ago

excellent work

Ali
Posted 2 months ago

that was great

Muhammad Ehsan
Posted 2 months ago

https://www.kaggle.com/docs/benchmarks?_gl=1*t0paa3*_ga*MjExNjQ1MDg2Ny4xNzUzNjU1NDcz*_ga_T7QHS60L4Q*czE3NzA3NjQ2NTckbzQ2NSRnMSR0MTc3MDc2NDcxMyRqNCRsMCRoMA..#intro
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More

Your Work


Viewed

Kaggle Benchmarks - Getting Started Notebook


Introducing Community Benchmarks


Measuring Progress Toward AGI - Cognitive Abilities


Cross-Modal Binding


Deep Past Challenge - Translate Akkadian to English


Edited

notebook557a6c5f22


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
How to Use Kaggle
Competitions

Datasets

Public API

Efficient GPU Usage Tips

Tensor Processing Units (TPUs)

Models

Competitions Setup

Organizations

Groups

Kaggle Packages

Notebooks

MCP Server

Benchmarks


Overview
Creating Tasks and Benchmarks
Downloading Benchmark Leaderboards
Models
Learning resources
Benchmarks
How to use Kaggle Benchmarks as your home for trustworthy AI benchmarking
Overview
Kaggle seeks to be the home of a diverse ecosystem of high quality benchmarks assessing model capabilities on tasks of significant importance to the industry to help developers reliably understand and trust what works well on ML tasks. Building on Kaggle's decade-plus of experience as the home for hosting ML Competitions, which are a type of benchmark, for the industry and our partners, we will adhere to the following principles:


Kaggle believes in the importance of robustness: enduring, high-value benchmarks that truly help the industry measure progress in AI are ones that can’t be easily hacked, saturated, or leaked
Kaggle believes in the importance of reproducibility and transparency for ensuring the industry can trust benchmarks and evaluations. We also take in extremely high regard the trust publishers place in us as a platform.
Kaggle doesn’t develop benchmarks. Our role is to independently reproduce and publicly release results, provide a model-agnostic platform that streamlines evaluation of new models on new benchmarks over time, and drive community engagement and stress testing.
Kaggle Benchmarks comprises two main types of benchmarks: 1) Research Benchmarks, which are evals created by researchers working in AI labs, and 2) Community Benchmarks, which are evals created by the Kaggle community.

Both are technically identical, with the only difference being that Research Benchmarks tend to require a lot more compute. If you're a researcher who wants to host your benchmarks with us, email kaggle-benchmarks@google.com to discuss how you can get a higher quota.

Creating Tasks and Benchmarks
First, some key concepts about Kaggle Benchmarks:

Task: A Python function defining the problem (e.g., "Solve this riddle").
Benchmark: A collection of tasks that you can put together. There is no code implementation for this. This is a feature that Kaggle supports on the graphical user interface so that users can put together their own benchmarks based on the tasks that they care about
Creating a Task
📺 Video Guide: How to create a task

1. Go to Kaggle Benchmarks and click "Create task"
Create a task
2. Create a new task - you can either write the code from scratch or prompt an AI to generate the code for you
⚠️ Access Requirements: Please ensure your account is phone-verified to access resources such as LLM API quotas. Furthermore, accounts registered after December 15, 2025, must complete additional identity verification to execute task notebooks.
Generate a taskGenerate a task
3. Once the task notebook has been created, you can make edits to it. Once it's done, you can run it in the notebook or "Save Task", which will create a Task Detail page
Task notebook
4. The Task Detail page is where you can add a description, new models to be evaluated, compare outputs across different models, and even share it with others
Task detail page
To get started creating your first task, check out the Getting Started Notebook.

Creating a Benchmark
Remember that a benchmark is simply multiple tasks put together into a collection.

📺 Video Guide: How to create a benchmark

1. Go to Kaggle Benchmarks and click "Create benchmark"
Create a benchmark
2. Fill in the information in the panel. You can always change names and descriptions later!
Generate a benchmark
3. You should be brought to the Benchmark Detail page, where you will need to add tasks to your benchmark. You can add your own tasks or public tasks that others have created.
Add tasks
4. Next, you will need to add a list of models that you want to display on the benchmark page.
Add models
5. Once that's done, you will see your completed Benchmark detail page. You can edit, share, and add new models and tasks!
Benchmark detail page
Downloading Benchmark Leaderboards
You can download the benchmark leaderboard data for your own analysis. There are two ways to access the download options:

From the three-dot menu ("︙") in the top right of the benchmark page.
Using the "Download" button located directly above the leaderboard table.
Both actions open a download popup that provides methods to retrieve the data.

Download via API
The popup provides a cURL command to download the leaderboard data as a JSON object. If the Benchmark is not public, you will need to authenticate using your Kaggle credentials.

# Unauthenticated example
curl -L -o ~/Downloads/open-benchmarks_scicode_leaderboard.json \
  https://www.kaggle.com/api/v1/benchmarks/open-benchmarks/scicode/leaderboard

# Authenticated example
# Export your Kaggle username and API key
# export KAGGLE_USERNAME=
# export KAGGLE_KEY=

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY \
  -o ~/Downloads/myusername_my-benchmark_leaderboard.json \
  https://www.kaggle.com/api/v1/benchmarks/myusername/my-benchmark/leaderboard
Download as CSV
At the bottom of the download popup, you can click "Download leaderboard as csv" to directly download the data as a CSV file.

Models
Supported Models in Community Benchmarks
We continue to update the list of available models in Community Benchmarks as new models are released and old models are deprecated. We currently do not support some models (e.g. OpenAI models), but are working on growing our list over time. To query the current list of supported models, run the following command in the task notebook:

import kaggle_benchmarks as kbench
# returns the current list of available models to test against
list(kbench.llms.keys())
Query list of available models
Supported Models in Research Benchmarks
Model selection within Research Benchmarks is determined by the specific evaluation and the researchers involved. Consequently, these may include supplemental models not currently supported in Community Benchmarks.

Learning resources
Getting started notebook
Kaggle Benchmarks GitHub repo
Kaggle Community Benchmarks NotebookLM

https://www.kaggle.com/code/nicholaskanggoog/kaggle-benchmarks-getting-started-notebook?scriptVersionId=290215074&_gl=1*1t4wxq*_ga*MjExNjQ1MDg2Ny4xNzUzNjU1NDcz*_ga_T7QHS60L4Q*czE3NzA3NjQ2NTckbzQ2NSRnMSR0MTc3MDc2NDc1MSRqNjAkbDAkaDA.
Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More

Your Work


Viewed

Kaggle Benchmarks - Getting Started Notebook


Introducing Community Benchmarks


Measuring Progress Toward AGI - Cognitive Abilities


Cross-Modal Binding


Deep Past Challenge - Translate Akkadian to English


Edited

notebook557a6c5f22


View Active Events

Search

Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
Nicholas Kang · 2mo ago · 1,158 views
Kaggle Benchmarks - Getting Started Notebook
Kaggle Benchmarks - Getting Started Notebook

Copy & Edit

54

Download

Version 3 of 3
Runtime
7s

Tags
Personal Benchmark
Language
Python

Table of Contents

License
This Notebook has been released under the Apache 2.0 open source license.

Continue exploring

Input
1 file

Output
0 files

Logs
6.9 second run - successful

Comments
0 comments
https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md
https://github.com/Kaggle/kaggle-benchmarks/tree/ci/documentation/examples
https://notebooklm.google.com/notebook/56661d72-a74b-48cc-a2d0-08a6f7a595e8?pli=1