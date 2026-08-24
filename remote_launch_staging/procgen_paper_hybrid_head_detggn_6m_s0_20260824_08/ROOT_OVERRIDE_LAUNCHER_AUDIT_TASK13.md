# Task 13 root-override launcher equivalence audit

- Task: `PROCGEN-HYBRID-HEAD-ROOT-OVERRIDE-6M-MISSING3-20260824-13`
- Assignment: `6f7032a8fe3f3350efd7d2df7e68b597f8384332`
- Base launcher SHA256:
  `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e`
- Versioned root-override launcher SHA256:
  `26f06ec93f84277c7e8d099b75f4e7d053cc74850926e0f04417813133cb07dd`
- Normalized launcher SHA256:
  `33f3fb9ed0485b6eda031bd22711f76c434b4744b5320f42c0d7ed06d3e57b4a`
- Audit result: `ROOT_OVERRIDE_LAUNCHER_EQUIVALENCE_PASS`

## Complete line diff

```diff
--- hybrid_head_detggn_6m_gpuh.sbatch
+++ hybrid_head_detggn_6m_gpuh_root_override_task13.sbatch
@@ -15,7 +15,18 @@
 set -uo pipefail
 ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
 case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
-CAMPAIGN=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08
+# ROOT_OVERRIDE_BEGIN
+BASE_LAUNCHER_SHA=ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e
+TASK_ID=PROCGEN-HYBRID-HEAD-ROOT-OVERRIDE-6M-MISSING3-20260824-13
+TASK11_CAMPAIGN=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08
+CAMPAIGN=${PROCGEN_CAMPAIGN_ROOT:?PROCGEN_CAMPAIGN_ROOT required}
+case "$CAMPAIGN" in /*) ;; *) echo "campaign must be absolute: $CAMPAIGN" >&2; exit 65;; esac
+CAMPAIGN=$(readlink -m -- "$CAMPAIGN")
+if [ "$CAMPAIGN" = "$TASK11_CAMPAIGN" ] || [[ "$CAMPAIGN" = "$TASK11_CAMPAIGN"/* ]]; then
+  echo "campaign resolves into immutable Task 11 campaign: $CAMPAIGN" >&2
+  exit 66
+fi
+# ROOT_OVERRIDE_END
 CODE="$CAMPAIGN/code"
 FROZEN="$CAMPAIGN/frozen"
 PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
@@ -30,6 +41,11 @@

 if [ -e "$ROOT" ]; then echo "collision: $ROOT" >&2; exit 90; fi
 mkdir -p "$ROOT" "$CODE/logs"
+# ROOT_PROVENANCE_BEGIN
+printf '%s\n' "$BASE_LAUNCHER_SHA" > "$ROOT/base_launcher.sha256"
+printf '%s\n' "$TASK_ID" > "$ROOT/task_id.txt"
+printf '%s\n' "$CAMPAIGN" > "$ROOT/campaign_root.txt"
+# ROOT_PROVENANCE_END
 echo PREFLIGHT_RUNNING > "$ROOT/status"
 printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
 hostname -f > "$ROOT/hostname"
```

After removing the two marked root/provenance blocks and normalizing the base
literal campaign line to `CAMPAIGN=<ARTIFACT_ROOT>`, every remaining line is
byte-identical. The normalized trainer command remains exactly:

```text
CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)
```

The normalized preflight invocation remains exactly:

```text
if ! "$PY" "$PREFLIGHT" "$TRAINER" "$CONFIG" "$ROOT/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA" > "$ROOT/compatibility.out" 2> "$ROOT/compatibility.err"; then
```

The variant requires an absolute `PROCGEN_CAMPAIGN_ROOT`, canonicalizes it with
`readlink -m`, rejects the Task 11 campaign and every descendant, and retains
the original pre-training root collision exit90. It records only the base
launcher hash, Task-ID, and resolved campaign root as added provenance.

## Frozen dependencies

| Artifact | SHA256 |
|---|---|
| trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` |
| config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |
| corrected preflight harness | `704278e8b5802498b8e065b9f12945e2cb72a665cdd28845b2401091b2e993ea` |
| structural manifest | `3f91f5c313480c089d300a7dd5aff4a664f28ff9d9f4718bbab430200298d623` |

Task 12's four accepted GPU validations are reused and are not rerun.
