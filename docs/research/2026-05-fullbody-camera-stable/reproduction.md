# Reproduction

## Shoulder-Only

Run from the `booster_gym/` directory:

```bash
python train_shoulder4_frozen.py \
  --task T1Shoulder4OfficialStrongZeroVelOpp_from7000LegFrozen_train500 \
  --leg_checkpoint logs/2026-05-05-11-09-07/nn/model_4000.pth \
  --max_iterations 500
```

## Full-Body Residual

```bash
python train_fullbody_residual.py \
  --task T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedPush_train300 \
  --leg_checkpoint logs/2026-05-05-11-09-07/nn/model_4000.pth \
  --upper_checkpoint logs/2026-05-14-08-59-29/nn/model_500.pth \
  --checkpoint logs/2026-05-14-15-08-00/nn/model_500.pth \
  --max_iterations 300
```

## Readback Loop

1. Train one variant family.
2. Run fixed-command evaluation.
3. Compare the numbers and the replay.
4. Append the winning replay to the web player.
5. Commit the code, the config, and the experiment record together.

