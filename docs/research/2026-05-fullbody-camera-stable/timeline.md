# Timeline

## Shoulder-only phase

- `NightA` to `NightE`: proved that simple shoulder penalties were not enough; sign and gate mistakes showed up easily.
- `NightF` to `NightH`: moved from surface fixes to the real issue, which was the inward boundary and the need for explicit outward margin.
- `NightI` to `NightK`: pushed toward low base, then blocked the lazy-roll escape route.
- `NightL`: compacted the shoulder reward into four stable groups.
- `NightM`: tried zero-base dynamics and found that stand base can be separated from moving lift.
- `NightN1` to `NightN7`: iterated on zero-spring, deadband, velocity trigger, pitch-first, inequality floor, smooth servo, and diagonal decoupling.
- `NightN7P` and `SideAmp`: kept forward/back swing while increasing lateral reach.

## Frozen-leg shoulder winner

- `LegLikeClean`: removed survival/tracking pressure and focused on arm shape plus side events.
- `LegLikeSagGate`: restored a strong forward/back gate so pitch swing would not disappear.
- `OfficialStrongZeroVelOpp`: made zero-base and contralateral side-lift explicit and became the shoulder-only winner.

## Full-body / residual phase

- `Upper9 Frozen`: froze the leg policy and learned a camera-stable whole-body style.
- `Residual open-leg`: reopened the legs with residual control instead of a full restart.
- `BalanceCap`, `CameraHard`, `CameraPriority`, `LowGrid`, `SpeedPolish`: explored how much speed and camera stability could coexist.
- `SpeedPush`: selected as the best tradeoff at 1.8 m/s.
- `SpeedUnlock`: useful probe, but it did not beat the speed-push winner.

