# Lessons

## What Worked

- Separate the standing shoulder base from the moving arm shape.
- Use a sign-aware forward/back gate before adding more amplitude.
- Use contralateral side-lift logic for lateral stepping.
- Keep the standing baseline close to zero and move the dynamic lift into the moving regime.
- Freeze the legs first when the upper body is still unclear, then open residual leg control later.
- Trust fixed-command eval and replay inspection more than the raw reward curve.

## What Failed Or Regressed

- Pure anti-sway can erase forward/back swing.
- A static moving base can hide the real behavior goal.
- Side-lift can look improved while forward/back motion quietly disappears.
- Camera stability can improve while the policy becomes too stiff.
- Opening the command grid too quickly can raise resets and hurt the winner.

## Why The Final Full-Body Winner Was Chosen

- It kept the camera stable enough to be usable.
- It preserved the 1.8 m/s forward case better than the later probe.
- It kept visible shoulder motion instead of collapsing into a stiff pose.
- It was a better tradeoff than the later speed-unlock probe.

## Useful Warning

Ugly training curves are not automatically bad. In this project they were sometimes caused by episode boundaries, logging artifacts, or the fact that the fixed eval was better than the online curve looked.

