# Experiment Index

| Phase | Main Label | What It Means | Where To Read |
|---|---|---|---|
| Early omni gait | `S0-S8` | The leg curriculum that established the working walking base | `README.md` |
| Shoulder search | `NightA` to `NightN7P` | Frozen-leg shoulder exploration for zero-base, forward swing, side lift, and symmetry | `docs/research/2026-05-fullbody-camera-stable/timeline.md` |
| Shoulder winner | `OfficialStrongZeroVelOpp` | Best shoulder-only outcome with zero-ish stand base and contralateral side lift | `docs/research/2026-05-fullbody-camera-stable/final_result.md` |
| Full-body residual | `Upper9` | Camera-stability-first full-body coordination with arm/elbow/waist participation | `docs/research/2026-05-fullbody-camera-stable/final_result.md` |
| Speed push | `SpeedPush model_500` | Best full-body tradeoff before the speed-unlock probe | `docs/research/2026-05-fullbody-camera-stable/final_result.md` |
| Speed unlock probe | `SpeedUnlock` | Follow-up expansion that was useful to test but did not beat the speed-push winner | `docs/research/2026-05-fullbody-camera-stable/lessons.md` |

## Artifact Layout

- `booster_gym/` — runnable code and YAML configs
- `artifacts/autoresearch/` — experiment logs, summaries, helper scripts, and probes
- `release_assets/` — web players and checkpoints for GitHub Release
- `docs/research/` — readable long-form retrospective for another AI

