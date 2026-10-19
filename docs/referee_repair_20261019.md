# Referee repair and next-run protocol

The October 16 trial completed 27 adjudicated actions: 26 engine-validation failures and one partial material-gathering success. Failed repair was converted into an ordinary physical failure, contaminating the agents' experience. The 12-hour continuation failed during model loading with an uncorrectable GPU ECC error. The archive filesystem was over quota on October 19.

## Implemented changes

- Referee-only instructions show explicit `op` fields, material extraction and alias-based transfer. Examples are conditional on an independent physical judgment, not guarantees of success or instructions to inhabitants.
- Operation validation names the offending delta index, required fields and supported operations. Trying to transfer a source explains the extraction interface without claiming extraction is feasible. Malformed source lists fail transactionally.
- Repair receives the rejected ruling and exact validation error against unchanged state. Transport/envelope repair also receives the rejected response, rather than just an error message.
- Exhausted referee validation raises an operational adjudication error. The scheduler restores the prior world time and entities, retains the pending event, logs the fault and pauses. The batch runner's existing error handling checkpoints and exits with a nonzero status.
- If an inhabitant already selected an action, that exact output and call ID are retained in the pending event across restart. Recovery retries adjudication without asking the inhabitant to choose again. A stale action completion is retained in the same way.
- Ordinary valid physical failures still complete and become sensory experience. There are no new survival recipes, goals, or agent-prompt changes in this fix.

This is runtime validation and bounded repair, not constrained token generation. The current Transformers adapter does not support JSON-schema-constrained generation. Reliability of the revised prompt with the actual Qwen weights remains unmeasured.

## Saved-context audit

Read-only inspection of the final archive at revision 4148 / simulated second 28544.130240609287 found no generic referee-failure message in either founder's current observation. Their latest result is arrival, their recent actions are repeated movement, and their recent experiences are conversations about a heart symbol. Adam's only private file is a love note; Eve has no private files. Current deliberations concern collecting materials and meeting at the outcrop.

This does not establish an uncontaminated experiment: earlier false failures may have changed subsequent decisions and dialogue. Do not erase or rewrite their memories. Preserve the original checkpoint and ledger as evidence.

## Recommended sequence

1. On a healthy GPU, run isolated referee cases with the same Qwen weights: extract material, pick up an existing object, transform accounted materials, attempt consumption, and attempt something physically infeasible. Include malformed-operation and source-transfer repair cases. Use independent fixture state and a separate call log, never the founders' memories. Repeat each case about ten times. Measure first-pass validity, repaired validity, exhaustion, latency, and mass/custody preservation; manually review plausibility. A valid failure is not a failed test. Any unresolved serialization error warrants investigation before the long run.
2. Resume a COPY of revision 4148 for a short observation window (roughly 200 decisions or 30 wall minutes), recording the code change as an operator intervention. Keep the original archive immutable. Verify that it makes progress, that actions have plausible results, and that any operational fault pauses without creating false experience. Do not force successful construction or food acquisition.
3. If that passes, extend the copied continuation toward 24 simulated hours, with a wall-time budget. Run a fresh-founder comparison separately if measuring autonomous behavior; the continued history includes previous interventions and failures.
4. Before scheduling, resolve archive quota without deleting research data indiscriminately and avoid the GPU/node that produced the ECC fault. Neither issue is repaired by changing prompts.

Local automated tests cover atomic repair, malformed operations, exact-decision recovery across restart, stale completion recovery, genuine physical failure, and transport repair. They do not demonstrate behavioral success with Qwen. No production job was restarted as part of this patch.

## Submitted experiment layout

The follow-up uses `/home/dss227/mesocosm/eden_referee_20261019` for staged source and `/home/dss227/mesocosm/runs/` for archives, because the project volume is over quota. The original checkpoint is copied into `resume_repaired_20261019`; `referee_eval_20261019` contains independent fixtures and model-call logs.

`python -m eden.referee_eval` executes eight cases ten times each. It checks runtime validity, atomicity, pickup and consumption effects in explicitly favorable fixtures, and rejection of material creation from nothing. Two cases inject a bad first ruling to exercise actual-model repair. A passing report is necessary for the dependent continuation, but does not substitute for reviewing physical plausibility. The scheduler dependency and a second report check prevent a time-limited or failed validation job from accidentally starting the continuation.

The continuation uses `--max-wall-seconds 1800` after model startup. Both allocations have a one-hour wall limit and exclude `a1125u05n01`, which reported the prior GPU ECC fault. The long day-scale extension is deferred until these results are reviewed.

Submission confirmed October 19:
- `30046424`: referee evaluation, `gpu_devel`, one hour, released after all 47 tests passed on Bouchet.
- `30046429`: copied-checkpoint continuation, `afterok:30046424`, regular GPU partitions, one-hour allocation and 1,800-second trial budget. Invalid dependencies cancel it. The development queue rejected a second submission due to its per-user limit, so this job uses regular GPU queues.
- Both exclude `a1125u05n01`. No unrelated held jobs were changed.

## Two-stage follow-up

Physical assessment now precedes translation. `eden/referee.py` supplies an exact entity lookup, positions and computed interaction distances. The assessment records assumptions, referenced IDs, intended effects and sensory outcome. The encoder returns only operations; repairs cannot overwrite the outcome or sensory narrative. Explicit scene-referenced revision requests permit one separate reassessment; repeated requests or representation limits pause operationally. Every stage is logged in the model-call archive, and accepted rulings include assessment history.

Declared material removal, object creation, custody and need relief are checked against the candidate state. Other novel consequences remain available and require an actual encoding or representation-limit report. These checks do not prove arbitrary physical reasoning or free-text claims: plausibility review remains necessary. No new survival instructions are given to inhabitants.

The follow-up suite has 110 cases (11 cases repeated ten times), including distant food, an unidentified plant, attached branches, and deliberately malformed encodings. Known favorable fixtures require real gathering, pickup or consumption effects; ambiguous physical attempts permit different coherent outcomes. Unit regressions cover explicit reassessment and representation limits. A separate Qwen job will evaluate the new behavior before any continuation is considered.

Two-stage validation submitted as job `30061875`, October 19. Archive: `/home/dss227/mesocosm/runs/referee_twostage_20261019`. The accepted small batch script unpacks `/home/dss227/mesocosm/eden-twostage-source-20261019.tar.gz` on node-local storage, runs the full unit suite, then executes 110 Qwen checks. It requests one hour on gpu_devel and excludes a1125u05n01. Local validation: 49 passed, two socket tests skipped. The first per-file staging attempt stalled on filesystem I/O; embedded-source submissions were rejected by Slurm without creating jobs. No founder continuation is submitted for this iteration pending result review.
