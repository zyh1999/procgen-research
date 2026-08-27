# Task54 launch report

Task-ID: `PROCGEN-TASK51-GPUH-QUICK-ETA-MIN-1OVER256-VALIDATOR-RECOVERY-BOSS-CAVE-2M-S0-20260827-54`

## Identity and bounded repair

Task53 remains terminal `PRECHECK_BLOCKED` and was not modified, restarted, or
resubmitted. Task54 is a fresh implementation, campaign, root set, and Slurm
step. The only trainer change from the frozen Task51 trainer is the validator's
expected `dualtrust_eta_min`, changed from `1/64` to the configured `1/256`.
The runtime eta update function and every other validation/scientific field are
unchanged.

- Implementation/origin commit: `f6214f02c499bb5ef7a35767d93a0c36c44ee741`
- Trainer SHA256: `49d0a05d9bcaaa6d6aeb1b49751beee70403d3c0d61ed2de999b2ac7a5a3be9b`
- Beta1 config SHA256: `ac3f389a1788ab09c6687feaccb9e246f462c4bf3b6dfc897c74d0dfa1956239`
- Beta4 config SHA256: `c8b7d8e3a37aa496d10fa058ac3319c18830b7b04842c584df394b82afd56303`
- Wrapper SHA256: `2f60c33e0490517493ca334c0ba0cc1e1914a2003296540ec0df0e7a32c27402`
- Bundle SHA256: `3318e7407e331ba1e38163bfdb1ba1206a5b71a6e970dc2fb797358794ea09ee`

The concise compile, shell syntax, source diff, bundle hash, CUDA/import, and
start checks passed. No negative-test or additional preflight chain ran.

## Placement and launch

- Parent allocation: `19487252`, `RUNNING`, node822, one H200
- Task54 persistent step: `19487252.4`, `RUNNING`
- Launch time: `2026-08-27T08:38:00+01:00`
- Campaign: `/scratch/h99859yz/procgen_task51_gpuh_quick_eta_min_1over256_validator_recovery_boss_cave_2m_s0_20260827_54`

All four cells were launched concurrently exactly once without MPS:

| Arm | Environment | shell/Python PID | Initial state | First transition |
|---|---|---:|---|---:|
| beta1 | BossFight | 4164825 / 4164828 | RUNNING | 40,960 |
| beta1 | CaveFlyer | 4164810 / 4164812 | RUNNING | 40,960 |
| beta4 | BossFight | 4164806 / 4164811 | RUNNING | 40,960 |
| beta4 | CaveFlyer | 4164803 / 4164805 | RUNNING | 40,960 |

Exact roots are under the campaign at
`runs/<arm-method>/<environment>/seed0/2m_quick_eta_min_1over256_validator_recovery`.
Each has its own status, PID, command, stdout/stderr, progress, and runtime
directory.

At the launch snapshot all four had passed the repaired validator, constructed
the 938,979-parameter production network, and entered PPO warmup. Each had a
first progress row at transition 40,960 and a populated metric trace. The phase
switch and rollout-scheduler ledgers were correctly still empty because the
fixed 503,808-transition switch had not yet been reached.

H200 snapshot: 143,771 MiB total, 7,222 MiB used, 98% utilization. The four
Python processes were visible independently. Hard infrastructure error scan was
zero; only benign Pillow deprecation warnings were present.

## Isolation

Task52 Slot A, Bede Task51, Slot B's Jupyter service, and all Task53 roots were
left untouched. There was no retry, restart, requeue, resubmit, extra network
port, model/checkpoint commit, or credential exposure.

Current bounded conclusion: `RUNNING_QUICK_MIRROR`.
