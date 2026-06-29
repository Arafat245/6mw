# Sway feature reference papers

Supporting literature for the mediolateral-sway WalkSway features used in the
POMS 6MWD project (clinic 6MWT). All establish that **mediolateral (ML) trunk
sway discriminates MS from controls**, motivating the intensity-invariant
directional-ratio features `ml_over_enmo`, `ml_over_vt`, `ml_energy_frac`,
`ml_spec_horiz_frac`.

1. **Huisinga et al. 2013** — *Accelerometry reveals differences in gait
   variability between patients with multiple sclerosis and healthy controls.*
   Ann Biomed Eng 41(8):1670–1679. doi:10.1007/s10439-012-0697-y. PMC3987786.
   → File: `Huisinga2013_AnnBiomedEng_gait_variability_MS_accelerometry.pdf`
   Key: greater Lyapunov exponent (ML & AP), greater **ML frequency dispersion**
   (p=0.034), greater **ML mean velocity** (p=0.045), lower **AP RMS** (p=0.040)
   in PwMS — matches our forward→lateral redistribution finding.

2. **Solomon et al. 2015** — *Detection of postural sway abnormalities by
   wireless inertial sensors in minimally disabled patients with multiple
   sclerosis: a case–control study.* J NeuroEng Rehabil 12:74.
   doi:10.1186/s12984-015-0066-9. PMC4556213 (Open Access).
   → File: `Solomon2015_JNER_postural_sway_inertial_MS.pdf`
   Key: **ML sway path length** and **ML range of sway acceleration** were the
   only two independent predictors separating MS from controls (87.5% accuracy).

3. **Alkathiry 2025** — *Key accelerometry measures for understanding walking
   sway during dual-task exercises.* Heliyon 11:e42160.
   doi:10.1016/j.heliyon.2025.e42160 (CC BY, Open Access).
   → File: `Alkathiry2025_Heliyon_walking_sway_accelerometry_measures.pdf`
   Key: RMS (variability of COM acceleration) and **normalized path length** in
   AP and ML during walking; mean-referencing + 3.5 Hz low-pass enhance
   sensitivity — supports our intensity-normalized ML sway construction.
