# Project Notes

## Known Issues (identified in review) — All Fixed

1. **CS bounds bug** — `sigma2` was computed as `var_c/n_c + var_t/n_t` (variance of the mean), which collapses at large N. Fixed: changed to `var_c + var_t` (per-observation variance), which is what the mSPRT formula expects. CS bounds are now ~2x wider than frequentist CI, as expected for anytime-valid inference.

2. **P(B>A) = 100% display** — Fixed: displays ">99.99%" and "<0.0001" instead of "100%" and "0" to signal Monte Carlo precision limits.

3. **Overpowered case study** — Fixed: reduced MDE from 0.8 to 0.08 hrs with variance=4.5^2. Required sample now ~50K per group, matching actual 50K (80% power). Experiment is properly powered, not 51x overpowered.

4. **Crash rate guardrail noise** — Fixed: true difference increased from 0.1pp to 0.5pp (control=1.9%, treatment=2.4%). Guardrail now catches a real regression, not noise.

5. **Quick Start path bug** — Fixed: moved `docker compose up -d postgres redis` before `cd backend` so docker-compose.yml is found.

6. **"10 methods" framing** — Fixed: README now says "4 core testing methods + 6 supporting techniques" with a Role column in the methods table.

7. **Segment vs overall not reconciled** — Fixed: deeper analysis table now explains that the overall +0.14 hrs is a weighted average (60% iOS, 40% Android) masking a null result on Android.
