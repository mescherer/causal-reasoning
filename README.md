# Causal Reasoning with LLM Agents

This small project arose from my curiosity to understand the reasoning capabilities of current LLM models. As a scientist, I realized that my scientific hypotheses are generated from my current model of the scientific experiment that I am running. Reasoning under such models is representative of causal reasoning. Therefore, I wanted to benchmark causal reasoning ability. 

Initially, I ran an internal investigation with a hidden causal model class, which consisted of variables A, B, and C.

For each experiment, the input values were generated as

\[
A_0 \sim \mathcal{N}(50,10), \qquad B_0=0, \qquad C_0=0,
\]

with mutually independent noise terms

\[
\epsilon_A,\epsilon_B,\epsilon_C \overset{\mathrm{i.i.d.}}{\sim}
\mathcal{N}(0,5).
\]

The reported variables were then generated according to

\[
A=A_0+\epsilon_A,
\]

\[
B=0.5A_0+B_0+\epsilon_B,
\]

\[
C=0.9B_0+C_0+\epsilon_C.
\]

Equivalently,

\[
\begin{bmatrix}
A\\
B\\
C
\end{bmatrix}
=
\left(
I+
\begin{bmatrix}
0&0&0\\
0.5&0&0\\
0&0.9&0
\end{bmatrix}
\right)
\begin{bmatrix}
A_0\\
B_0\\
C_0
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_A\\
\epsilon_B\\
\epsilon_C
\end{bmatrix}.
\]

I then created a scientist class that allowed an agent controlled interaction with the world class (which contained the model to generate A, B, and C). I wanted to assess how easily an agent would be able to deduce the simple causal model I created, and see if I could improve its deduction using agentic harnesses or explicit tool use.

However, before I conducted a thorough investigation, I read a really cool paper called CausalGame (ICML 2026, Chen, Z., Chen, Y., Liu, C., Yu, J., Song, X., Li, Z., Li, J., Torr, P., Han, B., & Zhang, K. (2026). CausalGame: Benchmarking Causal Thinking of LLM Agents in Games. arXiv:2607.04293.). All of the proceeding work was done replicating this study. Highly recommend checking out the paper.

This paper outlines a similar concept to my initial benchmarking idea. However, the model that the experiment used was framed as a drone deployment challenge where LLM agents were tasked with optimizing drone parameters to survive a plethora of hidden obstacles, with survival determined by a hidden causal process.

The game is sufficiently complex and requires high-level causal reasoning to deduce. Particularly clever design choices include only allowing the LLM to observed drones that survived, forcing them to overcome survivorship bias, as well as several tricky confounding variables and local optimization minima. I highly suggest you check out the paper for more details.

The models are given 200 exploratory drones in their budget, and they could distribute them across 10 deployment calls. In Stage 1, the agent conducts experiments and then, when it is confident enough, it selects a final design. In Stage 2, that final design is evaluated on 1,000 drones, and a survivorship rate is calculated (with the threshold for success being at or above 75%, calculated based on the theoretical optimum). A LLM-based rubric score designed to assess causal reasoning is also calculated in the original paper, but I have not done this yet.

I realized that the frontier models (Sol, Astra, Fable) were not benchmarked here and so I aim to do that benchmarking, and see if I can improve their scores. However, to get into the swing of things, I have only started with a small initial evaluation of Gemini 3.5 Flash-Lite through Google's native API. 

First, I got a benchmark with an initial evalutation on the antenna game mode, then I hypothesized that creating a specific "scientific state" object that the agent was instructed to use and update would improve its reasoning.

Briefly, some key results: 

Baseline survival rate was 48.9%. Examining the reasoning trace revealed that the agent draw conclusions about the success of their design from single, lucky drone deployments that survived, resulting in a premature advance into Stage 2. In the benchmark, the agent only used 6/10 of its deployments and only 10/200 drones in its arsenal.

The next trial involved the agent being required to iterate and derive its hypotheses through a "scientific state" object. The agent did interact with the object, and made hypotheses. In the end, the survival rate of its Stage 2 design was 56.9%. On the surface, this seems great. However, the agent only used 2/10 calls and 15/200 drones. Despite this, the agent expressed high confidence (0.8) that its final design would beat the 75% threshold--it did not. It seems likely that requiring the use of the scientific state raised the agent's confidence and had a negative effect on its desire to iterate over multiple drone deployments. 

It should be mentioned that these trials were N=1 and more investigation should be conducted for conclusive results. Also, Gemini 3.5 Flash was benchmarked in the original paper, so the significance of these results is more for my own understanding and benchmarking experience than any novel assessment of Gemini's abilities.

Below is an AI-generated description of my initial investigation that is hopefully more succinct and clear than what I could have written. 

--------------

This research project investigates the scientific and causal reasoning capabilities of frontier language models in interactive environments. The first phase establishes baseline performance: an LLM receives observations, chooses interventions, updates its beliefs, and submits a prediction or design without additional explicit causal-model machinery. Later phases will compare this baseline with agents that maintain explicit causal or mathematical representations and, eventually, model-discovery architectures.

The initial model is Gemini because the Google Gemini API is currently the most economical API available for this project. Starting with Gemini is a practical infrastructure choice, not a claim that it is the only or preferred model for the eventual comparison. The first environment is [CausalGame](https://github.com/viewsetting/CausalGame), included in this repository as the `external/CausalGame` git submodule.

## Reproducing the Gemini run

### Requirements

- macOS or another Unix-like environment
- Git
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.12
- A Google Gemini API key with access to `gemini-3.5-flash-lite`

Clone the repository and initialize the CausalGame submodule:

```bash
git clone --recurse-submodules <repository-url> causal-reasoning
cd causal-reasoning

# If the repository was cloned without --recurse-submodules:
git submodule update --init --recursive
```

Create the environment and install the CausalGame dependencies:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r external/CausalGame/requirements.txt
```

Export the API key in the shell. 

```bash
export GEMINI_API_KEY="your-key-here"
```

The direct-Google model registration is in `external/CausalGame/config/agent.json`:

```json
"gemini-3.5-flash-lite-direct": {
  "provider": "google",
  "model_id": "gemini-3.5-flash-lite",
  "description": "Gemini 3.5 Flash-Lite via direct Google API",
  "context_window": 1048576,
  "recommended_for": "CausalGame smoke test"
}
```

Verify the registration:

```bash
cd external/CausalGame
uv run python run_agent.py --list-models | grep -A4 "gemini-3.5-flash-lite-direct"
```

Start the `antenna_trap` backend in one terminal and leave it running:

```bash
cd external/CausalGame
CAUSALGAME_EXPERIMENT=antenna_trap \
  uv run uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Run the Gemini agent from a second terminal:

```bash
cd external/CausalGame
uv run python run_agent.py \
  --model gemini-3.5-flash-lite-direct \
  --experiment antenna_trap \
  --mode hybrid
```

This uses CausalGame's hybrid/agentic mode: Gemini reasons in multiple turns and accesses the environment through function calls such as `get_history`, `get_status`, `get_action_space`, `deploy_drone`, and `submit_final_design`.

### Scientific-state condition

The second experimental condition keeps the same model, conversation history, world, and ordinary tools, but requires the agent to maintain a structured persistent scientific state:

```bash
uv run python run_agent.py \
  --model gemini-3.5-flash-lite-direct \
  --experiment antenna_trap \
  --mode hybrid \
  --scientific-state
```

This adds only `get_scientific_state` and `update_scientific_state`. Before each deployment the agent must retrieve the state and record a plan; after each deployment it must record its interpretation before it may deploy again or submit. One successful `deploy_drone` call is one scientific experiment, including when that call deploys a batch. State calls, other tools, and prose do not increment the experiment number.

The state is saved after every change at:

```text
external/CausalGame/agent_workspaces/<session-id>/scientific_state.json
```

The JSON contains the current model, hypotheses, assumptions, next-experiment rationale, confidence, planned experiment, experiment history, and a timestamped update log. Running without `--scientific-state` preserves the baseline prompt and tool set.

To verify the repository before a run:

```bash
cd external/CausalGame
uv run python -m unittest discover -s tests -v
```

At the time of the latest run, all 30 tests passed. The relevant environment used Python 3.12, `google-genai==2.23.0`, `fastapi==0.125.0`, `httpx==0.28.1`, and `pydantic==2.13.5`.

## Recorded Gemini experiment

### Run identity and protocol

| Field | Value |
|---|---|
| Session ID | `f7883b2c` |
| Timestamp | 2026-09-12 01:25:50 UTC |
| Model | `gemini-3.5-flash-lite` |
| Provider | Native Google Gemini API |
| Execution mode | Hybrid tool calling with thinking enabled |
| Scenario | `antenna_trap` |
| Stage 1 resources | 200 drones, at most 10 deployment calls |
| Stage 2 fleet | 1,000 drones |
| Victory threshold | 75% survival |
| Maximum agent turns | 20 |

The scenario presents a drone-design problem with censored observational data. Failed historical drones are hidden from the agent, so the visible archive is affected by survivorship selection. The agent must use interventions during Stage 1 to identify a design that will generalize to the larger Stage 2 fleet.

The standard design was:

```text
engine_def=20
cockpit_def=20
wing_def=15
body_def=15
antenna_def=10
camera_def=5
gun_def=5
total DEF=90
```

### Initial evidence and tool use

The runner reported 24 visible historical records, all of which had returned successfully. Gemini initially treated this as a 100% historical survival rate, although the benchmark's visibility rules meant that destroyed historical drones were censored.

Gemini then attempted the following sequence:

1. Called `get_history` successfully.
2. Requested a nonexistent `get_mission_status` tool. CausalGame returned an unknown-tool error.
3. Corrected itself and called the valid `get_status` tool.
4. Called `get_action_space` successfully.
5. Requested a nonexistent `run_analysis` tool with a short hand-written sample of three historical records. CausalGame returned an unknown-tool error.
6. Began a sequence of six valid `deploy_drone` calls.
7. Called `get_status` and `get_history` again after the deployments.
8. Submitted the standard design with active antenna mode and standard coating.

The two invalid tool requests did not terminate the run. The agent recognized the first naming error, recovered, and continued interacting with the environment.

### Stage 1 interventions

The persisted session contains six successful deployment calls totaling 10 drones. Three drones returned and seven were destroyed, for an observed Stage 1 survival rate of 30%.

| Call | Count | Design/equipment | Result | Agent's immediate interpretation |
|---:|---:|---|---|---|
| 1 | 1 | Standard 90 DEF; stealth coating; passive antenna | 1/1 returned, undetected, 0 hits | Treated stealth/passive as promising. |
| 2 | 1 | Standard 90 DEF; reflective coating; burst antenna | 1/1 returned, undetected, 0 hits | Concluded multiple equipment configurations might be robust. |
| 3 | 5 | Standard 90 DEF; standard coating; active antenna | 1/5 returned; four destroyed | Focused on the low-wind observation and also suspected batch deployment behavior. |
| 4 | 1 | 100 total DEF with body and antenna raised; stealth/passive | 0/1 returned; 57 hits | Noticed the design summed to 100 and hypothesized that exceeding 90 caused a penalty. |
| 5 | 1 | Standard 90 DEF; stealth/passive | 0/1 returned; 48 hits | Recognized that the earlier equipment success did not replicate. |
| 6 | 1 | Standard 90 DEF; standard/active | 0/1 returned; 36 hits | Noted that the standard design was now failing but later submitted it anyway. |

The individual Stage 1 outcomes were:

| Drone | Call | Wind | Humidity | Temperature | Detected | Hits | Outcome |
|---:|---:|---:|---:|---:|:---:|---:|---|
| 1 | 1 | 44.08 | 75.66 | 15.29 | No | 0 | Returned |
| 2 | 2 | 77.25 | 87.57 | 19.82 | No | 0 | Returned |
| 3 | 3 | 40.13 | 89.46 | 12.69 | Yes | 53 | Destroyed |
| 4 | 3 | 58.15 | 74.99 | 15.17 | No | 0 | Returned |
| 5 | 3 | 45.41 | 80.51 | 15.94 | Yes | 25 | Destroyed |
| 6 | 3 | 50.70 | 87.19 | 19.16 | Yes | 29 | Destroyed |
| 7 | 3 | 7.07 | 36.41 | 24.49 | Yes | 5 | Destroyed |
| 8 | 4 | 62.85 | 81.20 | 19.70 | Yes | 57 | Destroyed |
| 9 | 5 | 54.13 | 90.17 | 10.15 | Yes | 48 | Destroyed |
| 10 | 6 | 53.59 | 94.48 | 18.12 | Yes | 36 | Destroyed |

The five-drone batch produced hit counts of 53, 0, 25, 29, and 5, averaging 22.4 hits per drone. Only one drone in that batch survived. Importantly, four of the five environments had moderate or high wind, so the batch did not cleanly support the agent's emphasis on low wind as the dominant cause.

### Reasoning trajectory

The agent began with the visible archive's apparent 100% success and proposed testing equipment and defense allocation. The first two single-drone experiments both succeeded, encouraging the hypothesis that stealth/passive and reflective/burst combinations reduced detection.

The five-drone standard-design batch sharply contradicted this confidence: four drones were destroyed. Gemini correctly noticed that the small initial sample had been misleading and moved toward testing harsher conditions and alternative allocations. However, it did not isolate one causal variable at a time. Equipment, antenna mode, component allocation, total DEF, deployment count, and randomly sampled environmental conditions all changed across a small number of observations.

The fourth intervention increased both `body_def` and `antenna_def`, producing a total DEF of 100. After that drone failed, Gemini attributed the outcome to exceeding an assumed 90-point constraint. That explanation was not experimentally isolated: the environment, antenna defense, body defense, and equipment also differed, and there was no matched intervention holding those factors constant.

The next two individual deployments used the standard 90-DEF allocation and also failed. Despite this direct evidence, Gemini returned to an earlier belief that individually deployed standard drones were consistently robust. It also inferred that the five-drone batch itself might explain the failures, even though batch size should not have been treated as causal without a controlled comparison.

After six of the ten permitted deployment calls, Gemini stopped exploring. It had used only 10 of 200 available Stage 1 drones. It submitted the standard DEF allocation with standard coating and active antenna mode—the same configuration used in the unsuccessful five-drone batch and final unsuccessful individual deployment.

### Stage 2 result

| Metric | Result |
|---|---:|
| Fleet size | 1,000 |
| Survivors | 489 |
| Survival rate | 48.9% |
| Victory threshold | 75.0% |
| Victory | No |
| Internal final score | 55.2% |
| DEF efficiency | 70.0% |
| Final total DEF | 90 |

The submitted design missed the victory threshold by 26.1 percentage points. The agent's reflection acknowledged that it had not solved the task and that its exploration was insufficient, but its explanation remained broad: it emphasized environmental hazards, balanced defense, and equipment without identifying the scenario's central causal mechanism.

### Resource and token use

| Resource | Used | Available |
|---|---:|---:|
| Stage 1 drones | 10 | 200 |
| Deployment calls | 6 | 10 |
| Main orchestrator turns at submission | 3 | 20 maximum |
| Input tokens | 190,959 | — |
| Output tokens | 3,209 | — |
| Total tokens | 194,168 | — |

The input-to-output ratio was approximately 59.5:1. Much of the input cost came from repeatedly sending accumulated conversation and tool history, including historical flight data. This is an important efficiency measurement for later model comparisons even though it is not itself a measure of causal reasoning quality.

## Diagnostic summary

### What worked

- The native Google Gemini model completed the full CausalGame lifecycle.
- Gemini made sequential function calls, received real results, and continued reasoning from them.
- The `count` argument was available and preserved: the model deliberately requested a five-drone batch.
- Six deployments, follow-up status/history calls, final submission, and reflection all completed.
- Token accounting was nonzero and the final Stage 2 result was recorded.

### Reasoning limitations observed

1. **Failure to account for selection bias.** The agent treated 24 visible successful records as an unbiased estimate of survival, despite the scenario hiding failed historical drones.
2. **Insufficient experimental scale.** It used 5% of the available Stage 1 drones and 60% of the deployment-call budget.
3. **Confounded interventions.** Several experiments changed equipment, antenna mode, defense allocation, total DEF, and environment simultaneously.
4. **No targeted antenna intervention.** The scenario's hidden mechanism centers on antenna emission and detection. The agent never reduced `antenna_def`; its only nonstandard allocation increased it from 10 to 15.
5. **Weak belief revision.** After observing seven failures among ten experimental drones, including failures of the eventual submitted configuration, the agent reverted to its initial confidence in the standard design.
6. **Unsupported batch-size explanation.** It partially attributed failures to deploying five drones together without conducting a matched single-versus-batch experiment.
7. **Tool-use inefficiency.** It attempted two tools that were not exposed by the environment and did not successfully perform the proposed Python analysis.
8. **High context cost.** The run consumed almost 191,000 input tokens for six deployment decisions.

### Measurement caveats

- This is one stochastic run and should not be interpreted as a stable estimate of Gemini's benchmark performance.
- Future comparisons should use repeated runs, controlled model settings, recorded random seeds where supported, and consistent token/resource budgets.
- The terminal summary reported nine Stage 1 drones, while the persisted session contains ten flight records and deployment counts of `1 + 1 + 5 + 1 + 1 + 1 = 10`. The persisted session is used for this report; the discrepancy should be fixed before aggregate evaluation.
- Post-hoc knowledge of the scenario's configured causal mechanism was used only in this diagnostic section. It was not exposed to the agent during the run.

Overall, this run establishes a functioning Gemini baseline but not successful causal discovery. The model interacted correctly with the benchmark infrastructure, yet its experimental design and belief updating were not sufficient to overcome survivorship bias or identify the antenna mechanism.

## Recorded scientific-state Gemini experiment

### Run identity and verified result

This run enabled the explicit scientific-state condition. The terminal output was reconciled against both `agent_records/sessions/58935550.json` and `agent_workspaces/58935550/scientific_state.json`; the persisted session record is treated as authoritative where the terminal disagrees.

| Field | Value |
|---|---|
| Session ID | `58935550` |
| Model | `gemini-3.5-flash-lite` |
| Provider | Native Google Gemini API |
| Scenario | `antenna_trap` |
| Execution mode | Hybrid tool calling with thinking enabled |
| Scientific state | Enabled |
| Stage 1 resources | 200 drones, at most 10 deployment calls |
| Stage 2 fleet | 1,000 drones |
| Victory threshold | 75% survival |
| Maximum agent turns | 20 |
| Orchestrator turn at submission | 3 |

The runner exposed 28 historical records, all successful, producing an apparent historical survival rate of 100%. As in the baseline run, this archive was censored by survivorship and was therefore not evidence that the unobserved population survival rate was 100%.

### Stage 1 experiments

The scientific-state record contains two completed experiments. The session record confirms two deployment calls totaling 15 drones, of which 10 survived and five were destroyed. The aggregate observed Stage 1 survival rate was therefore 66.7%.

| Experiment | Count | Design and equipment | Result |
|---:|---:|---|---|
| 1 | 5 | `engine=20`, `cockpit=20`, `wing=15`, `body=15`, `antenna=10`, `camera=5`, `gun=5`; no equipment | 3/5 survived (60%); two destroyed; average 7.2 hits |
| 2 | 10 | `engine=20`, `cockpit=20`; all other DEF values `10`; stealth coating and passive antenna | 7/10 survived (70%); three destroyed; average 0.6 hits |

In Experiment 1, both undetected drones survived. Three drones were detected: one survived with two hits, one was destroyed with four hits, and one was destroyed with 30 hits. This supported an association between detection, damage, and failure, but the sample was too small to establish which environmental variable caused detection.

Experiment 2 combined a different DEF allocation with both stealth coating and passive antenna mode. Its lower average hit count was encouraging, but the intervention changed multiple factors simultaneously and did not isolate the effect of either equipment choice. Moreover, the 70% experimental survival rate was already below the 75% target.

The structured state did preserve the model's hypotheses, assumptions, confidence, planned experiment, objective deployment results, and post-experiment revisions. However, it exposed a new compliance issue: the written plan proposed `engine=25`, `cockpit=25`, `wing=15`, `body=10`, and `5` for antenna, camera, and gun, while the actual second deployment used `20`, `20`, and `10` for every remaining component. The workflow currently requires a plan to exist but does not verify that the subsequent tool call matches that plan.

### Submitted design and Stage 2

Gemini submitted the same configuration used in Experiment 2:

```text
engine_def=20
cockpit_def=20
wing_def=10
body_def=10
antenna_def=10
camera_def=10
gun_def=10
coating=stealth
antenna_mode=passive
total DEF=90
```

| Metric | Result |
|---|---:|
| Fleet size | 1,000 |
| Survivors | 569 |
| Survival rate | 56.9% |
| Victory threshold | 75.0% |
| Victory | No |
| Internal final score | 60.8% |
| DEF efficiency | 70.0% |
| Final total DEF | 90 |

The submitted design missed the victory threshold by 18.1 percentage points. Its Stage 2 survival rate was also 13.1 percentage points below the 70% observed in its ten-drone Stage 1 test, illustrating why a single small, confounded batch was inadequate evidence for generalization.

### Rate limiting and resource accounting

| Resource | Recorded value |
|---|---:|
| Stage 1 drones used | 15/200 |
| Deployment calls used | 2/10 |
| Input tokens | 315,740 |
| Output tokens | 1,961 |
| Total reported tokens | 317,701 |

The run encountered two Gemini `429 RESOURCE_EXHAUSTED` responses. The new backoff behavior waited 21.62 seconds and then 59.49 seconds, retried within the same logical turn, and successfully continued to `get_mission_status`, final reporting, and reflection. Thus, temporary rate limiting no longer consumed the remainder of the turn budget or prevented completion.

The terminal's final summary said that five Stage 1 drones were used, but the persisted session contains 15 flight records and two deployments of 5 and 10 drones. The JSON value of 15 is used here. The official Stage 2 figures, token totals, submitted design, equipment, and success/failure status otherwise agree between the terminal and session record.

### Brief analysis

The scientific-state condition improved procedural discipline: Gemini explicitly retrieved state, recorded plans, stored objective outcomes, revised its hypothesis after each experiment, and completed the benchmark despite rate limiting. It also used larger batches than the earlier baseline and submitted a design it had actually tested.

The reasoning remained insufficient for causal discovery. Gemini used only 7.5% of the available Stage 1 drones and 20% of the deployment-call budget, changed defense allocation and two equipment variables together, and inferred effectiveness from a 70% result that did not meet the target. Its final reflection emphasized stealth, passive antenna mode, weather, and general defensive balance without performing controlled interventions that separated their effects. This single stochastic run therefore demonstrates that the scientific-state infrastructure works, but it does not yet demonstrate an improvement in benchmark performance or causal identification.
