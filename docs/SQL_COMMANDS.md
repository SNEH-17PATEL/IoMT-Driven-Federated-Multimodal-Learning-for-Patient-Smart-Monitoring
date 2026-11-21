# SQL Commands — Rebuilding the Training Dataset in BigQuery

This file records every BigQuery command used to rebuild the two tables the project reads:

| Final table | Read by | Columns (must stay exactly the same) |
|---|---|---|
| `ml_dataset_final` | `train_federated.py` Phase 0 | subject_id, hadm_id, icustay_id, window_time, HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, Temp_mean, SBP_mean, DBP_mean, MAP_mean, latest_HR, latest_RR, latest_SpO2, latest_Temp, latest_SBP, latest_DBP, latest_MAP, sofa_score, GCS_eye_opening, stress_score (23) |
| `clinical_notes` | `train_federated.py` Phase 0 | subject_id, hadm_id, chartdate, category, text (5) |

**Rule:** only *how* the data is built changes. The column names, their meaning and their types stay identical, so no Python code (training, backend, dashboard) needs to change.

**How to run:** open the BigQuery console with project `mimic-project-2` selected, paste one command at a time into the query editor, and click **Run**. Run the commands in order; each one depends on the tables made before it.

---

## The plan

All intermediate tables go into a new working dataset, **`mimic-project-2.dataset_v2`**, so the current tables stay untouched until the very end.

| Stage | Table | What it fixes compared with the old pipeline |
|---|---|---|
| 1 | `cohort` | Adults only (≥ 16), ICU stays ≥ 6 hours measured from the real in/out times. The old data included about 8,000 newborns. |
| 2 | `vitals_clean` | Uses `mimiciii_derived.pivoted_vital` (both CareVue and MetaVision, includes cuff BP), drops impossible values and readings outside the ICU stay. |
| 3 | `vitals_hourly` | One row per stay per hour, aligned to the hourly SOFA timeline. |
| 4 | `vital_trends` | Trend columns over the previous 6 *hours* of time (the old version used "the previous 20 rows", which could span days). |
| 5 | `latest_vitals` | Latest value carried forward for a limited time instead of being empty. |
| 6 | `gcs_hourly` | GCS Eye from `pivoted_gcs`, which includes CareVue item 184 (the old query only used MetaVision item 220739, so GCS was 83% empty). |
| 7 | `stress_hourly` | Nurse pain score from **both** systems: MetaVision item 223791 (as before) plus CareVue items 1044 and 524, whose text values ("0-None" … "10-Worst") are converted to 0–10. Coverage goes from about 21,600 stays to about 40,000. |
| 8 | `labels` | **SOFA for the same hour** from `pivoted_sofa.SOFA_24hours`. The old label was one first-day SOFA copied onto every window of the stay. |
| 9 | `ml_dataset_final` | Joins everything into the same 23 columns as before. |
| 10 | `clinical_notes` | Uses the bedside categories Nursing, `'Nursing/other'`, `'Physician '` and `'Respiratory '` (exact MIMIC spellings), drops discharge summaries and error notes, and keeps only notes written during the ICU stay (by `CHARTTIME`). |
| 11 | Checks | Profiles of the new tables, compared with the old ones. |
| 12 | Switch over | Back up the old tables in `Dataset`, then copy the two new tables in under the same names. |

### Source tables (chosen after surveying all of `physionet-data`)

Our account can access 4 datasets: `mimiciii_clinical` (25 tables), `mimiciii_derived` (113), `mimiciii_notes` (1) and `mimiciii_notes_derived` (5). MIMIC-IV and eICU are not accessible.

| Final column(s) | Source | Why this one |
|---|---|---|
| Which stays | `mimiciii_derived.icustay_detail` | Age, ICU in/out times, one row per stay |
| HR, RR, SpO₂, Temp, SBP, DBP, MAP | `mimiciii_derived.pivoted_vital` | Both CareVue and MetaVision, includes cuff BP: 52,436 stays with BP vs 39,926 with the old arterial-only item list |
| GCS_eye_opening | `mimiciii_derived.pivoted_gcs` (`GCSEyes`) | Both systems: 52,350 stays vs 23,381 with the old item 220739 |
| stress_score | `mimiciii_clinical.chartevents` items 223791, 1044, 524 | The only pain scores in MIMIC-III; the old query used only 223791 |
| sofa_score | `mimiciii_derived.pivoted_sofa` (`SOFA_24hours`) | Hourly SOFA, so each row's label matches its own time |
| clinical_notes | `mimiciii_notes.noteevents` | Has `CHARTTIME` and `ISERROR` for exact timing and error filtering |

---

## Command 1 — Create the working dataset

Creates the empty dataset that will hold all the new tables. It does not touch anything that already exists.

```sql
CREATE SCHEMA IF NOT EXISTS `mimic-project-2.dataset_v2`
OPTIONS (location = "US");
```

**Expected result:** "This statement created a new dataset named mimic-project-2.dataset_v2" (or a message that it already exists). In the left panel, `dataset_v2` appears under `mimic-project-2`.

---

## Command 2 — Stage 1: the patient cohort

Chooses which ICU stays the whole dataset is built from. Every later table is limited to these stays.

**Status:** tested by running it in a scratch dataset and checking the result (2 iterations, see the end of this section).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.cohort` AS
WITH stays AS (
  SELECT
    d.subject_id,
    d.hadm_id,
    d.icustay_id,
    TIMESTAMP(d.intime)  AS intime,
    TIMESTAMP(d.outtime) AS outtime,
    DATETIME_DIFF(d.outtime, d.intime, MINUTE) / 60 AS los_icu_hours,
    d.admission_age,
    d.icustay_seq
  FROM `physionet-data.mimiciii_derived.icustay_detail` AS d
  WHERE d.admission_age >= 16                                   -- adults only
    AND d.intime  IS NOT NULL
    AND d.outtime IS NOT NULL                                   -- complete ICU in/out times
    AND DATETIME_DIFF(d.outtime, d.intime, MINUTE) >= 6 * 60    -- ICU stay of at least 6 hours
    AND d.icustay_id IN (                                       -- hourly SOFA must exist (the label)
          SELECT DISTINCT icustay_id
          FROM `physionet-data.mimiciii_derived.pivoted_sofa`
        )
),
charted AS (                                                    -- at least one heart-rate reading
  SELECT DISTINCT s.icustay_id                                  -- recorded DURING the stay
  FROM stays AS s
  JOIN `physionet-data.mimiciii_derived.pivoted_vital` AS v
    ON v.icustay_id = s.icustay_id
   AND v.HeartRate IS NOT NULL
   AND TIMESTAMP(v.charttime) BETWEEN s.intime AND s.outtime
)
SELECT s.*
FROM stays AS s
JOIN charted USING (icustay_id);
```

**Expected result:** **52,248** stays (38,304 patients, 48,863 admissions).

### Why each part, with the evidence

**Source: `mimiciii_derived.icustay_detail`, not the raw `icustays` table**
- It already joins the stay to the patient and admission, and computes `admission_age`, `icustay_seq` and the in/out times, so we don't redo (and risk getting wrong) the age calculation from date of birth.
- It has 61,051 stays; the raw `icustays` table has 481 more. Those 481 are stays MIMIC itself excludes from `icustay_detail` (inconsistent or missing patient records), so leaving them out is intended.
- Checked: its `intime`/`outtime` match the raw `icustays` table exactly for every cohort stay (0 differences).

**`admission_age >= 16` — why 16, and not 18 or "all ages"**

| Age band | ICU stays | Unit |
|---|---|---|
| Under 1 (newborns) | 8,058 | Neonatal ICU (NICU) |
| 1–15 | 37 | |
| 16–17 | 62 | Adult ICUs (trauma, medical, cardiac) |
| 18–89 | 50,191 | Adult ICUs |
| Over 89 | 2,703 | Adult ICUs (age shown as about 300 for privacy) |

- **Why remove under-16s:** 8,087 of the 8,095 under-16 stays are newborns in the neonatal ICU. SOFA is defined for adults, and a newborn's normal vitals look severely abnormal on adult scales (a normal newborn heart rate of 140 is tachycardia for an adult; a normal newborn MAP of 40 is shock for an adult). Mixing them in teaches the model wrong relationships. This was a real problem in the old dataset.
- **Why 16 and not 18:** the patients aged 16–17 were all treated in **adult** ICUs with adult protocols, and 16 is the age cut-off used in the standard MIMIC-III research cohorts. Including them follows convention and loses nothing.
- **Why keep over-89s:** they are real adult ICU patients (a common group in ICUs). Their age is only *displayed* as about 300; age is not one of our model's inputs, so it has no effect.

**At least 6 hours in the ICU — and why it is calculated from the times, not `los_icu`**
- `los_icu` in this table is a **whole number of days** (an INTEGER), rounded down. A stay of 17.6 hours shows as 1, a stay of 23 hours as 0. Filtering on `los_icu >= 0.25` therefore silently meant "at least 1 full day" and removed every stay under 24 hours (the first version of this command had this bug).
- Real stay lengths for adults, calculated from `intime`/`outtime`:

  | Stay length | Stays |
  |---|---|
  | under 4 h | 249 |
  | 4–6 h | 135 |
  | 6–12 h | 605 |
  | 12–24 h | **6,969** |
  | 24 h or more | 44,995 |

  Using the real times keeps the 7,574 stays of 6–24 hours, which the rounded column would have thrown away.
- **Why 6 hours:** the trend columns (Stage 4) summarise the previous 6 hours of readings, so a stay needs at least 6 hours to produce even one complete trend window. The 384 shorter stays are mostly brief observations or transfers with very few readings.

**Complete in/out times** — 3 adult stays have no `outtime`, so their length (and their hourly timeline) cannot be built. Removing 3 of 53,000 is negligible.

**Must have hourly SOFA in `pivoted_sofa`** — the label (Stage 8) comes from this table; a stay without it cannot have a label. Checked: every adult stay of 6+ hours has it, so this filter removes 0 stays today. It is kept as a safety guard so the cohort can never contain a stay without a label.

**At least one heart-rate reading during the stay (added in iteration 2)**
- **Why:** iteration 1 found **321 stays with no heart-rate reading at all** between ICU admission and discharge, and 240 with no vital signs of any kind. A stay with no vitals can never produce a single training row (every row needs vitals), so it would only add empty work to every later stage.
- **Why heart rate specifically:** it is the most frequently charted vital (present in 87% of `pivoted_vital` rows); a stay without even one heart-rate reading has effectively no bedside monitoring recorded.
- **Why only readings *inside* the stay:** 6 of the 321 had heart-rate readings, but outside the ICU stay times. Readings before admission or after discharge don't describe the ICU stay, so they don't count.
- **Checked for bias:** the 321 removed stays are not a hidden group of very sick patients being thrown away. Most had no chart data at all (only 130 of 321 had any chart entry), and their average first-day SOFA was 1.9 against 4.2 for the kept stays; they are mostly records with missing documentation. Their higher death rate (21% vs 12%) is consistent with comfort-care or rapidly fatal admissions where monitoring was not charted, which are not suitable for a monitoring model.

**Repeat ICU stays are kept** — 3,486 stays are a patient's second or later stay. They are real, separate episodes, so they are kept. Checked: no two stays of the same patient overlap in time (0 overlaps), so no reading can be counted twice. To stop the same person appearing in both training and test data, the split will be done **by patient** (`subject_id`), not by stay.

**Both MIMIC systems are included** — 28,863 CareVue stays, 23,250 MetaVision stays and 135 that span both. All three groups are kept; the later stages use tables that cover both systems.

**Columns kept and why**

| Column | Used for |
|---|---|
| subject_id | Patient-level train/test split; part of the final table |
| hadm_id | Linking notes; part of the final table |
| icustay_id | Joining every other table; part of the final table |
| intime, outtime | Building the hourly timeline and keeping only readings inside the stay |
| los_icu_hours | The real stay length (checks) |
| admission_age, icustay_seq | Checks only; not model inputs |

`intime`/`outtime` are converted to TIMESTAMP so they compare directly with the TIMESTAMP columns in the later tables.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | Age ≥ 16, real stay length ≥ 6 h, times present, hourly SOFA present | 52,569 stays | 321 stays have no heart-rate reading during the stay → can never produce a row |
| 2 | **+ at least one heart-rate reading inside the stay** | **52,248 stays** | All checks pass (below) |

Checks run on the final version:
- 52,248 rows, 52,248 unique stays (no duplicates), no empty IDs
- Shortest stay 6.0 h, youngest patient 16
- ICU times identical to the raw `icustays` table
- 0 overlapping stays for the same patient
- 0 stays without heart-rate data
- Hourly SOFA covers each stay's full length (median 1.02 SOFA hours per stay hour)
- Every cohort stay also existed in the old `ml_dataset_final`, so nothing new or unexpected was added; the old data had 7,940 more stays, almost all newborns and the stays removed above

**Check it (optional):**
```sql
SELECT COUNT(*)                   AS stays,
       COUNT(DISTINCT subject_id) AS patients,
       COUNT(DISTINCT hadm_id)    AS admissions,
       MIN(los_icu_hours)         AS shortest_hours,
       MIN(admission_age)         AS youngest
FROM `mimic-project-2.dataset_v2.cohort`;
```
Expected: 52,248 stays · 38,304 patients · 48,863 admissions · shortest 6.0 hours · youngest 16.

---

## Command 3 — Stage 2: clean vital-sign readings

Takes every vital-sign reading recorded during a cohort stay and removes values that are physically impossible or recording errors. Output: one row per (stay, charttime) with the 7 vitals our final dataset uses.

**Status:** tested by running it in a scratch dataset and checking the result (4 iterations, see the end of this section).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.vitals_clean` AS
WITH raw AS (                                     -- 1. readings inside each cohort stay, first range check
  SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    TIMESTAMP(v.charttime) AS charttime,
    IF(v.HeartRate BETWEEN 20 AND 250, v.HeartRate, NULL) AS HR,
    IF(v.SysBP     BETWEEN 40 AND 250, v.SysBP,     NULL) AS SBP0,
    IF(v.DiasBP    BETWEEN 10 AND 180, v.DiasBP,    NULL) AS DBP0,
    IF(v.MeanBP    BETWEEN 20 AND 200, v.MeanBP,    NULL) AS MAP0,
    IF(v.RespRate  BETWEEN  2 AND  70, v.RespRate,  NULL) AS RR,
    IF(v.TempC     BETWEEN 30 AND  43, v.TempC,     NULL) AS Temp,
    IF(v.SpO2      BETWEEN 50 AND 100, v.SpO2,      NULL) AS SpO2_0
  FROM `physionet-data.mimiciii_derived.pivoted_vital` AS v
  JOIN `mimic-project-2.dataset_v2.cohort`             AS c
    ON v.icustay_id = c.icustay_id
   AND TIMESTAMP(v.charttime) BETWEEN c.intime AND c.outtime
),
checked AS (                                      -- 2. information needed for the consistency checks
  SELECT
    *,
    (SBP0 IS NOT NULL AND DBP0 IS NOT NULL AND SBP0 <= DBP0) AS bp_impossible,
    LAG(SpO2_0)  OVER w AS spo2_prev,
    LEAD(SpO2_0) OVER w AS spo2_next,
    TIMESTAMP_DIFF(charttime, LAG(charttime)  OVER w, MINUTE) AS gap_prev,
    TIMESTAMP_DIFF(LEAD(charttime) OVER w, charttime, MINUTE) AS gap_next
  FROM raw
  WINDOW w AS (PARTITION BY icustay_id, (SpO2_0 IS NULL) ORDER BY charttime)
),
cleaned AS (                                      -- 3. apply the consistency checks
  SELECT
    subject_id,
    hadm_id,
    icustay_id,
    charttime,
    HR,
    IF(bp_impossible, NULL, SBP0) AS SBP,
    IF(bp_impossible, NULL, DBP0) AS DBP,
    CASE
      WHEN bp_impossible                        THEN NULL
      WHEN SBP0 IS NOT NULL AND MAP0 > SBP0 + 5 THEN NULL
      WHEN DBP0 IS NOT NULL AND MAP0 < DBP0 - 5 THEN NULL
      ELSE MAP0
    END AS MAP,
    RR,
    Temp,
    IF(SpO2_0 < 85 AND spo2_prev >= 94 AND spo2_next >= 94
       AND gap_prev <= 60 AND gap_next <= 60, NULL, SpO2_0) AS SpO2
  FROM checked
)
SELECT *                                          -- 4. keep only rows with at least one valid vital
FROM cleaned
WHERE COALESCE(HR, SBP, DBP, MAP, RR, Temp, SpO2) IS NOT NULL;
```

**Expected result:** **6,893,304** rows covering all **52,248** cohort stays.

### Why each part, with the evidence

**Source: `mimiciii_derived.pivoted_vital`, not raw `chartevents`**
- It already combines both MIMIC systems (CareVue and MetaVision) and both blood-pressure methods (arterial line and cuff), converts Fahrenheit to Celsius, and removes the most extreme typing errors.
- On the same 52,248 stays it has **5,746,586 systolic BP readings vs 3,366,848** in the old pipeline (+71%), because the old query used only arterial-line item IDs. The old pipeline also kept values such as heart rate 9,999,999, SpO₂ 6,363,333 and temperature 5,336 °C.

**Only readings between ICU admission and discharge** — `pivoted_vital` also contains 48,115 readings timed before admission or after discharge for these stays. They do not describe the ICU stay (and after-discharge readings would leak future information), so they are dropped by the join.

**The plausible ranges — chosen from the data, with clinical limits**

Percentiles of the raw readings inside cohort stays, and how many readings each limit removes:

| Vital | 0.1st percentile | Median | 99.9th percentile | Kept range | Readings removed |
|---|---|---|---|---|---|
| Heart rate | 39 | 86 | 158 | **20–250** | 285 (272 below, 13 above) |
| Systolic BP | 50 | 119 | 210 | **40–250** | 2,378 |
| Diastolic BP | 20 | 59 | 127 | **10–180** | 508 |
| MAP | 29 | 77 | 170 | **20–200** | 5,409 |
| Respiratory rate | 5 | 20 | 47 | **2–70** | 289 |
| Temperature (°C) | 33.6 | 37.0 | 39.9 | **30–43** | 79 |
| SpO₂ | 62 | 98 | 100 | **50–100** | 2,989 |

How the limits were set:
- They sit **well outside** the 0.1st–99.9th percentile of real readings, so real but extreme patients are kept: profound bradycardia (HR 30), severe shock (MAP 25), severe hypothermia (31 °C) and severe hypoxaemia (SpO₂ 55) all survive.
- Values beyond them are not survivable or not measurable, so they are recording errors: a MAP of 4,606, a systolic of 12, an SpO₂ above 100%, a respiratory rate of 0.17.
- Less than 0.1% of readings are removed for any vital.
- A removed value becomes **missing (NULL)**, never 0, and only that one vital is blanked; the other vitals recorded at the same time are kept.

**Systolic must be above diastolic** — 350 readings had systolic ≤ diastolic (for example SBP 4 / DBP 44), which is impossible. Usually the two fields were swapped or one was mistyped, and we cannot tell which, so **SBP, DBP and MAP** of that reading are all blanked.

**MAP must lie between diastolic and systolic (±5 mmHg)**
- Mean arterial pressure is by definition between diastolic and systolic. 16,451 readings broke this; the far cases (9,527) are clear errors.
- The ±5 mmHg allowance accepts small differences from rounding and from the three numbers not being captured at exactly the same moment.
- Only MAP is blanked, because SBP and DBP passed their own checks.
- The check is applied whenever SBP or DBP is present (iteration 3 fixed a version that only checked when *both* were present, which let 99 bad MAPs through, e.g. DBP 68 with MAP 45).

**Isolated SpO₂ dips are removed** — a single SpO₂ reading below 85% with readings of 94% or more both within the hour before and the hour after.
- Why: a real desaturation lasts more than a moment; a single dip between normal readings is almost always the probe slipping, cold fingers or patient movement. Among readings under 70%, 27% were followed within two hours by a reading of 90% or more.
- 5,198 readings (14% of all SpO₂ readings under 85%) match and are removed. Sustained low SpO₂ — two or more low readings in a row — is kept, because that is real hypoxaemia, which the model must learn.

**Rows with no valid vital are dropped** — 514,345 rows had nothing left after cleaning (for example rows that only held a glucose value, which we do not use). They carry no information. The filter is applied *after* the checks (iteration 4 fixed a version that filtered before them and left 288 empty rows).

**What is not done here, on purpose**
- **No averaging or filling of gaps** — this table keeps the readings exactly as recorded (after cleaning). Hourly averaging and carrying values forward happen in the next stages, so this table can be checked on its own.
- **Glucose is not kept** — it is in `pivoted_vital` but is not one of our final columns.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | Range limits only | 7,407,649 rows | 154 impossible SBP ≤ DBP readings, 15,148 MAPs outside [DBP, SBP], 514,057 rows with no vital |
| 2 | + SBP > DBP rule, + MAP-between rule (when both present), + drop empty rows | 6,893,592 rows | 99 bad MAPs remained (rule only applied when both SBP and DBP were present); 2 empty rows |
| 3 | + MAP rule against SBP and DBP separately, + remove isolated SpO₂ dips | 6,893,592 rows | All MAP checks pass; 5,198 SpO₂ dips removed; but 288 rows were empty after cleaning |
| 4 | **Drop empty rows after cleaning, not before** | **6,893,304 rows** | All checks pass (below) |

Checks run on the final version:
- 6,893,304 rows, all 52,248 stays present, 0 duplicate (stay, time) rows, 0 completely empty rows
- Every value inside its range: HR 20–250, SBP 40–250, DBP 10–180, MAP 20–200, RR 2–69, Temp 30–42.8, SpO₂ 50–100
- 0 readings with SBP ≤ DBP; 0 MAPs outside [DBP − 5, SBP + 5]
- Medians are clinically normal: HR 85, SBP 119, DBP 59, MAP 77, RR 20, Temp 37.0, SpO₂ 98
- Readings available: HR 6.21 M, SBP 5.75 M, DBP 5.75 M, MAP 5.75 M, RR 6.24 M, Temp 1.72 M, SpO₂ 6.04 M
- Stays that never have a given vital: 0 (HR), 2 (BP), 35 (RR), 172 (Temp), 46 (SpO₂); these become missing and are handled in later stages

**Check it (optional):**
```sql
SELECT COUNT(*)                   AS readings,
       COUNT(DISTINCT icustay_id) AS stays,
       COUNT(HR)   AS hr,   COUNT(SBP) AS sbp, COUNT(MAP) AS map,
       COUNT(RR)   AS rr,   COUNT(Temp) AS temp, COUNT(SpO2) AS spo2,
       COUNTIF(SBP <= DBP) AS impossible_bp
FROM `mimic-project-2.dataset_v2.vitals_clean`;
```
Expected: 6,893,304 readings · 52,248 stays · HR 6,206,905 · SBP 5,746,586 · MAP 5,748,672 · RR 6,240,011 · Temp 1,719,519 · SpO₂ 6,042,814 · impossible_bp 0.

---

## Command 4 — Stage 3: hourly vitals on the SOFA timeline

Puts the cleaned readings onto a fixed **hourly timeline** for every stay. The timeline is the one used by the hourly SOFA table, so in the final dataset every row's vitals and its SOFA label describe exactly the same hour.

**Status:** tested by running it in a scratch dataset and checking the result (2 iterations, see the end of this section).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.vitals_hourly` AS
WITH grid AS (                                        -- 1. the hourly timeline of each stay
  SELECT DISTINCT
    s.icustay_id,
    s.hr                   AS hr_idx,
    TIMESTAMP(s.starttime) AS starttime,
    TIMESTAMP(s.endtime)   AS endtime
  FROM `physionet-data.mimiciii_derived.pivoted_sofa` AS s
  JOIN `mimic-project-2.dataset_v2.cohort`             AS c USING (icustay_id)
  WHERE TIMESTAMP(s.endtime) > c.intime                -- drop hours that end before ICU admission
),
bounds AS (                                           -- 2. last hour of each stay
  SELECT icustay_id, MAX(hr_idx) AS last_hr
  FROM grid
  GROUP BY icustay_id
)
SELECT                                                -- 3. one row per stay-hour with that hour's readings
  c.subject_id,
  c.hadm_id,
  g.icustay_id,
  g.hr_idx,
  g.starttime,
  g.endtime,
  AVG(v.HR)          AS HR,
  AVG(v.SBP)         AS SBP,
  AVG(v.DBP)         AS DBP,
  AVG(v.MAP)         AS MAP,
  AVG(v.RR)          AS RR,
  AVG(v.Temp)        AS Temp,
  AVG(v.SpO2)        AS SpO2,
  MIN(v.SpO2)        AS SpO2_min,
  COUNT(v.HR)        AS HR_n,
  SUM(v.HR)          AS HR_sum,
  SUM(v.HR * v.HR)   AS HR_sumsq,
  COUNT(v.charttime) AS n_readings
FROM grid AS g
JOIN `mimic-project-2.dataset_v2.cohort` AS c ON c.icustay_id = g.icustay_id
JOIN bounds                               AS b ON b.icustay_id = g.icustay_id
LEFT JOIN `mimic-project-2.dataset_v2.vitals_clean` AS v
  ON v.icustay_id = g.icustay_id
 AND v.charttime >= g.starttime
 AND (v.charttime < g.endtime                                       -- reading falls inside the hour
      OR (g.hr_idx = b.last_hr AND v.charttime <= c.outtime))       -- last hour also takes readings up to discharge
GROUP BY c.subject_id, c.hadm_id, g.icustay_id, g.hr_idx, g.starttime, g.endtime;
```

**Expected result:** **5,253,941** rows (one per stay-hour) covering all **52,248** stays.

### Why each part, with the evidence

**Why an hourly timeline, and why this one**
- The old pipeline used 10-minute windows built from whenever readings happened to exist, and then attached one SOFA value for the whole stay. The new label (`SOFA_24hours`, Command 9) changes **every hour**, so the features must be built on the same hours.
- `pivoted_sofa` defines that timeline: each row is one exact hour (`starttime`–`endtime`, always 60 minutes, always starting on the hour), numbered by `hr`. Taking the grid straight from it guarantees every row of the final dataset has a matching label.
- `SOFA_24hours` at a given hour summarises the **24 hours ending at that hour's end** (checked: it is never lower than the organ scores of the hour itself across all 5.38 M rows). So vitals averaged *within* that hour come from the same period as the label, without looking into the future.

**`SELECT DISTINCT` on the grid** — `pivoted_sofa` contains **12,388 duplicated (stay, hour) rows** inside our cohort (the same hour listed 2–3 times). Without removing them, every reading in those hours would be counted twice or three times. In 852 of them the duplicates even disagree on the SOFA value; this table only needs the hour boundaries, which are identical, and Command 9 handles the label duplicates. Checked: no hours are *missing* from the grid (0 gaps in any cohort stay).

**Dropping hours that end before ICU admission** — the first `pivoted_sofa` hour (`hr = 0`) always ends *before* the patient arrives (it starts 60–119 minutes before admission). In iteration 1, 100% of these hours were empty, because our cleaned readings only start at admission. They are removed, so the timeline starts with the hour in which the patient arrives.

**How readings are assigned to an hour**
- A reading belongs to the hour where `starttime ≤ charttime < endtime`. The end is exclusive, so a reading at exactly 14:00 goes to the 14:00–15:00 hour, never to both. Checked: the hourly table holds exactly **6,893,304** readings, the same number as `vitals_clean` (0 lost, 0 counted twice).
- **The last hour also takes readings up to discharge.** The final `pivoted_sofa` hour ends up to 59 minutes before the patient leaves. In iteration 1, **13,870 readings** recorded in those last minutes were lost; the rule above keeps them.

**What is calculated for each hour**

| Column | How | Why |
|---|---|---|
| HR, SBP, DBP, MAP, RR, Temp, SpO2 | Mean of the hour's readings | The hourly value of each vital; this is what `latest_*` will use |
| SpO2_min | Lowest SpO₂ in the hour | Needed for the final `SpO2_min` column (lowest value over the trend window) |
| HR_n, HR_sum, HR_sumsq | Count, sum and sum of squares of heart-rate readings | Lets Command 5 compute the exact heart-rate mean and standard deviation over the trend window from the *individual readings*, not from hourly averages (which would hide beat-to-beat variability) |
| n_readings | All readings in the hour | Checks: how well each hour is documented |

**Empty hours are kept, not removed** — 365,396 hours (7%) have no reading. Most are single hours between two documented hours (90% of gaps are 1–2 hours), which is normal charting rhythm. Keeping the row keeps the timeline continuous; Command 6 fills such gaps by carrying the last value forward for a limited time.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | Grid from `pivoted_sofa` (DISTINCT), hourly means | 5,306,189 rows | First hour (`hr = 0`) empty in 100% of stays because it ends before admission; 13,870 readings after the last grid hour were lost |
| 2 | **Drop hours ending before admission; last hour takes readings up to discharge; add HR sum/sum-of-squares** | **5,253,941 rows** | All checks pass (below) |

Checks run on the final version:
- 5,253,941 rows, all 52,248 stays, 0 duplicate (stay, hour) rows
- Readings in the hourly table = readings in `vitals_clean` = 6,893,304 (none lost or double-counted)
- Share of hours with a value: HR 92.5%, SBP/MAP 89.8%, RR 91.0%, SpO₂ 89.0%, Temp 30.2% (temperature is normally charted every 4 hours)
- `HR_sum / HR_n` equals the hourly HR mean in every row
- Remaining empty hours are short charting gaps (90% are 1–2 hours long)

**Check it (optional):**
```sql
SELECT COUNT(*)                   AS hours,
       COUNT(DISTINCT icustay_id) AS stays,
       SUM(n_readings)            AS readings,
       COUNTIF(n_readings = 0)    AS empty_hours,
       ROUND(COUNT(HR) / COUNT(*), 3)  AS hr_share,
       ROUND(COUNT(MAP) / COUNT(*), 3) AS map_share
FROM `mimic-project-2.dataset_v2.vitals_hourly`;
```
Expected: 5,253,941 hours · 52,248 stays · 6,893,304 readings · 365,396 empty hours · HR 0.925 · MAP 0.898.

---

## Command 5 — Stage 4: the 9 trend columns

Builds the 9 trend columns of the final dataset (`HR_mean`, `HR_std`, `RR_mean`, `SpO2_mean`, `SpO2_min`, `Temp_mean`, `SBP_mean`, `DBP_mean`, `MAP_mean`) for every stay-hour, from the readings of the recent past.

**Status:** tested by running it in a scratch dataset and checking the result (2 iterations, see the end of this section).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.vital_trends` AS
WITH windowed AS (                                    -- 1. sums and averages over the trend window
  SELECT
    subject_id,
    hadm_id,
    icustay_id,
    hr_idx,
    starttime,
    endtime,
    SUM(HR_n)     OVER w4 AS hr_n_win,
    SUM(HR_sum)   OVER w4 AS hr_sum_win,
    SUM(HR_sumsq) OVER w4 AS hr_sumsq_win,
    AVG(RR)       OVER w4 AS RR_mean,
    AVG(SpO2)     OVER w4 AS SpO2_mean,
    MIN(SpO2_min) OVER w4 AS SpO2_min,
    AVG(SBP)      OVER w4 AS SBP_mean,
    AVG(DBP)      OVER w4 AS DBP_mean,
    AVG(MAP)      OVER w4 AS MAP_mean,
    AVG(Temp)     OVER w8 AS Temp_mean
  FROM `mimic-project-2.dataset_v2.vitals_hourly`
  WINDOW
    w4 AS (PARTITION BY icustay_id ORDER BY hr_idx RANGE BETWEEN 3 PRECEDING AND CURRENT ROW),  -- this hour + 3 before = 4 h
    w8 AS (PARTITION BY icustay_id ORDER BY hr_idx RANGE BETWEEN 7 PRECEDING AND CURRENT ROW)   -- 8 h, for temperature
)
SELECT                                                -- 2. heart-rate mean and standard deviation from the raw readings
  subject_id,
  hadm_id,
  icustay_id,
  hr_idx,
  starttime,
  endtime,
  hr_sum_win / NULLIF(hr_n_win, 0) AS HR_mean,
  IF(hr_n_win >= 3,
     SQRT(GREATEST((hr_sumsq_win - hr_sum_win * hr_sum_win / hr_n_win) / (hr_n_win - 1), 0)),
     NULL)                         AS HR_std,
  RR_mean,
  SpO2_mean,
  SpO2_min,
  Temp_mean,
  SBP_mean,
  DBP_mean,
  MAP_mean,
  hr_n_win AS hr_readings_in_window
FROM windowed;
```

**Expected result:** **5,253,941** rows (one per stay-hour), all **52,248** stays.

### Why each part, with the evidence

**A window measured in time, not in rows**
- The old query averaged "the previous 20 rows" (`ROWS BETWEEN 19 PRECEDING`). Rows are created only when something is charted, so 20 rows could be 3 hours in a busy hour or several days in a quiet one — the same column meant different things for different patients.
- Here the window is `RANGE BETWEEN 3 PRECEDING AND CURRENT ROW` on the hour number: always **the current hour plus the 3 before it**, regardless of how often readings were taken. Missing hours simply contribute nothing.
- The window only looks **backwards**, so no trend ever uses a reading from after the hour it describes.

**Why 4 hours for most vitals — matching the dashboard and checking against the label**
- **Training must match how the dashboard computes these columns.** `ml_service.py` builds the trend columns from the last 20 readings of the patient, and the demo patients' readings are 10 minutes apart: 20 × 10 min ≈ **3–4 hours**. A 4-hour window gives the model training values that mean the same thing as the values it sees live.
- **Checked against the label** (`SOFA_24hours`, on a 5% sample of 254,470 stay-hours), for different window lengths:

  | Window | Correlation of MAP_mean with SOFA | HR_mean | SpO2_min |
  |---|---|---|---|
  | 3 h | −0.148 | 0.071 | −0.019 |
  | **4 h** | ≈ −0.15 | ≈ 0.07 | ≈ −0.02 |
  | 6 h | −0.157 | 0.075 | −0.034 |
  | 12 h | −0.163 | 0.076 | −0.056 |
  | 24 h | −0.169 | 0.072 | −0.083 |

  Longer windows correlate slightly more with SOFA, but only because `SOFA_24hours` itself covers 24 hours. The gains are small (≤ 0.02 for MAP and HR), and a 24-hour window would not match the dashboard at all, so the model would see different numbers live than in training. 4 hours is the honest choice.

**Why 8 hours for temperature**
- Temperature is charted about every 4 hours (the median gap between temperature readings is 3–4 hours, and only 30% of hours have one). In a 4-hour window, **12.5%** of rows had no temperature at all (iteration 1).
- An 8-hour window always contains at least one routine temperature check: missing values fall to **5.3%**. Temperature changes slowly, so an 8-hour average is still current.

**Heart-rate mean and standard deviation from the individual readings**
- `HR_mean` and `HR_std` are computed from the count, sum and sum of squares of **every heart-rate reading** in the window (prepared in Command 4), not from hourly averages. Averaging first would smooth away the beat-to-beat changes that `HR_std` is meant to capture.
- Formula: mean = Σx / n; standard deviation = √[(Σx² − (Σx)²/n) / (n − 1)] — the usual sample standard deviation, identical to what the dashboard's `pandas .std()` computes.
- `GREATEST(..., 0)` guards against tiny negative values caused by floating-point rounding when all readings are equal.

**`HR_std` needs at least 3 readings (added in iteration 2)**
- With 1 reading the standard deviation is undefined; with 2 it is extremely unstable. In iteration 1, windows with only 1–2 readings produced 97,133 zero values and many meaningless ones.
- Requiring 3 readings leaves it missing where it cannot be estimated (8.1% of rows), and most windows have 4 or more readings (median 4).

**Very high heart-rate variability is kept** — 2,389 rows (0.045%) have `HR_std` above 40. Checked: these are real events, not errors — the example shows a heart rate of 51 jumping to 135–149 within minutes (a sudden arrhythmia). Each reading already passed the 20–250 range check, and this kind of instability is exactly what the column exists to capture.

**`SpO2_min` is the lowest single reading** — the minimum of the hourly minimums, i.e. the lowest actual SpO₂ in the window, not the lowest hourly average. Isolated probe dips were already removed in Command 3, so this reflects real desaturation.

**`hr_readings_in_window`** is kept only for checking; it is not one of the final columns.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | 4-hour time window for all vitals; HR mean/std from raw readings | 5,253,941 rows | Temperature missing in 12.5% of rows (charted every ~4 h); `HR_std` computed from 1–2 readings gave 97,133 zeros |
| 2 | **8-hour window for temperature; `HR_std` only with ≥ 3 readings** | **5,253,941 rows** | All checks pass (below) |

Checks run on the final version:
- 5,253,941 rows, all 52,248 stays, 0 duplicate (stay, hour) rows
- Missing share: HR_mean 2.7%, HR_std 8.1%, RR_mean 3.3%, SpO2_mean/min 3.9%, Temp_mean 5.3%, SBP/DBP_mean 3.6%, MAP_mean 3.2% (the old final table had 13–45% missing in these columns)
- All values inside the cleaned ranges (e.g. HR_mean 20–232, MAP_mean 20–198, SpO2_min 50–100); the old table had values like HR_std 7,070,999
- `HR_std` median 4.0 (quartiles 2.3–6.4), in line with the dashboard demo patients (3.0–6.9)
- Correlations with SOFA have the clinically expected direction: lower MAP → higher SOFA (−0.15), higher HR → higher SOFA (+0.07)

**Check it (optional):**
```sql
SELECT COUNT(*)                             AS rows_,
       COUNT(DISTINCT icustay_id)           AS stays,
       ROUND(1 - COUNT(HR_mean)  / COUNT(*), 3) AS hr_mean_missing,
       ROUND(1 - COUNT(HR_std)   / COUNT(*), 3) AS hr_std_missing,
       ROUND(1 - COUNT(MAP_mean) / COUNT(*), 3) AS map_mean_missing,
       ROUND(1 - COUNT(Temp_mean)/ COUNT(*), 3) AS temp_mean_missing
FROM `mimic-project-2.dataset_v2.vital_trends`;
```
Expected: 5,253,941 · 52,248 · 0.027 · 0.081 · 0.032 · 0.053.

---

## Command 6 — Stage 5: the 7 latest-vital columns

Builds `latest_HR`, `latest_RR`, `latest_SpO2`, `latest_Temp`, `latest_SBP`, `latest_DBP` and `latest_MAP` for every stay-hour: the **most recent reading** of each vital, carried forward for a limited time if nothing new was charted.

**Status:** tested by running it and checking the result (3 iterations, see the end of this section).

> **Storage note:** while testing this command, the project reached BigQuery's free storage limit (10 GB). The scratch copies of earlier tables were deleted to make room (the real tables in `dataset_v2` were checked first and are identical). If you ever see *"Quota exceeded: Your project exceeded quota for free storage"*, delete tables you no longer need; the old `Dataset` tables will be replaced at the end (Command 12).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.latest_vitals` AS
WITH last_in_hour AS (                       -- 1. newest single reading of each vital within each hour
  SELECT
    h.icustay_id,
    h.hr_idx,
    ARRAY_AGG(v.HR   IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS HR,
    ARRAY_AGG(v.RR   IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS RR,
    ARRAY_AGG(v.SpO2 IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS SpO2,
    ARRAY_AGG(v.Temp IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS Temp,
    ARRAY_AGG(IF(COALESCE(v.SBP, v.DBP, v.MAP) IS NULL, NULL,          -- SBP, DBP and MAP are taken
                 STRUCT(v.SBP AS SBP, v.DBP AS DBP, v.MAP AS MAP))     -- together from ONE reading
              IGNORE NULLS ORDER BY v.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS BP
  FROM `mimic-project-2.dataset_v2.vitals_hourly` AS h
  JOIN `mimic-project-2.dataset_v2.vitals_clean`  AS v
    ON v.icustay_id = h.icustay_id
   AND v.charttime >= h.starttime
   AND v.charttime <  h.endtime
  GROUP BY h.icustay_id, h.hr_idx
),
timeline AS (                                -- 2. every stay-hour, with that hour's newest readings (if any)
  SELECT
    h.subject_id, h.hadm_id, h.icustay_id, h.hr_idx, h.starttime, h.endtime,
    l.HR, l.RR, l.SpO2, l.Temp, l.BP
  FROM `mimic-project-2.dataset_v2.vitals_hourly` AS h
  LEFT JOIN last_in_hour AS l
    ON l.icustay_id = h.icustay_id
   AND l.hr_idx     = h.hr_idx
),
carried AS (                                 -- 3. most recent value so far, and the hour it was recorded
  SELECT
    *,
    LAST_VALUE(HR   IGNORE NULLS) OVER w AS HR_c,   LAST_VALUE(IF(HR   IS NULL, NULL, hr_idx) IGNORE NULLS) OVER w AS HR_at,
    LAST_VALUE(RR   IGNORE NULLS) OVER w AS RR_c,   LAST_VALUE(IF(RR   IS NULL, NULL, hr_idx) IGNORE NULLS) OVER w AS RR_at,
    LAST_VALUE(SpO2 IGNORE NULLS) OVER w AS SpO2_c, LAST_VALUE(IF(SpO2 IS NULL, NULL, hr_idx) IGNORE NULLS) OVER w AS SpO2_at,
    LAST_VALUE(Temp IGNORE NULLS) OVER w AS Temp_c, LAST_VALUE(IF(Temp IS NULL, NULL, hr_idx) IGNORE NULLS) OVER w AS Temp_at,
    LAST_VALUE(BP   IGNORE NULLS) OVER w AS BP_c,   LAST_VALUE(IF(BP   IS NULL, NULL, hr_idx) IGNORE NULLS) OVER w AS BP_at
  FROM timeline
  WINDOW w AS (PARTITION BY icustay_id ORDER BY hr_idx ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT                                       -- 4. keep a carried value only while it is still recent
  subject_id, hadm_id, icustay_id, hr_idx, starttime, endtime,
  IF(hr_idx - HR_at   <= 2, HR_c,     NULL) AS latest_HR,
  IF(hr_idx - RR_at   <= 2, RR_c,     NULL) AS latest_RR,
  IF(hr_idx - SpO2_at <= 2, SpO2_c,   NULL) AS latest_SpO2,
  IF(hr_idx - Temp_at <= 6, Temp_c,   NULL) AS latest_Temp,
  IF(hr_idx - BP_at   <= 2, BP_c.SBP, NULL) AS latest_SBP,
  IF(hr_idx - BP_at   <= 2, BP_c.DBP, NULL) AS latest_DBP,
  IF(hr_idx - BP_at   <= 2, BP_c.MAP, NULL) AS latest_MAP
FROM carried;
```

**Expected result:** **5,253,941** rows (one per stay-hour), all **52,248** stays.

### Why each part, with the evidence

**"Latest" means the newest single reading, not the hourly average**
- The dashboard's `latest_*` columns are the patient's **newest reading** (`ml_service.py`), so training uses the same meaning.
- The old pipeline used the 10-minute average and called it "latest", which mostly duplicated the trend columns.
- When an hour has more than one reading (9.3% of hours for MAP), the newest one is chosen (`ORDER BY charttime DESC LIMIT 1`). It differs from the hourly mean by a median of 2.5 mmHg and by 12 mmHg or more in 10% of those hours, so this choice matters: a patient whose MAP fell from 85 to 60 within the hour should have `latest_MAP = 60`, not 72.

**Carrying values forward — and why only for a limited time**
- A vital that wasn't charted this hour hasn't disappeared; its most recent reading is still the best estimate. Without carrying forward, `latest_*` would be empty in 7.5–11% of hours (70% for temperature) and would become a misleading median or flag later.
- But an old reading can be out of date. Checked by comparing a carried MAP with the next real MAP reading:

  | Age of the carried MAP | Average error vs the next real reading |
  |---|---|
  | 1 hour | 8.9 mmHg |
  | 2 hours | 10.0 mmHg |
  | 3–4 hours | 10.4 mmHg |
  | 6–12 hours | 12.4 mmHg |

- The error grows with age, so the carry limit is set from how often each vital is normally charted:

  | Vital | Normal charting interval (median / 95th pct) | Carry limit | Why |
  |---|---|---|---|
  | HR, RR, SpO₂, BP | 60 / 60–90 minutes | **2 hours** | Covers one or two missed hourly checks; missing falls from about 10% to about 4% |
  | Temperature | 194 / 330 minutes | **6 hours** | Covers the normal 4-hourly check plus a short delay; missing falls from 70% to 6% |

  Longer limits gain little coverage (MAP missing 3.7% at 2 h vs 3.0% at 4 h) while making the value staler.

**Blood pressure is carried as one reading (added in iteration 3)** — in iteration 2, SBP, DBP and MAP were carried forward separately, so a row could combine an SBP from one hour with a DBP from another. That produced 58 rows with SBP ≤ DBP and 10,768 with MAP outside the DBP–SBP range — the same impossible combinations Command 3 removed. Now all three come from the **same** newest BP reading (a STRUCT) and are carried together: 0 impossible combinations.

**No future information** — every value comes from a reading at or before the end of its own hour (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`). A value is never filled backwards from a later reading.

**Readings must lie inside the hour** — the newest reading is taken only from readings with `starttime ≤ charttime < endtime`, the same rule as Command 4, so every reading counts for exactly one hour.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | Hourly *mean*, carried forward (2 h; 6 h for temperature) | 5,253,941 rows | Coverage good (3–6% missing), but "latest" was an average, not the newest reading the dashboard uses |
| 2 | **Newest single reading in the hour**, carried forward | 5,253,941 rows | SBP, DBP, MAP carried separately → 58 rows with SBP ≤ DBP, 10,768 with MAP outside DBP–SBP |
| 3 | **SBP, DBP, MAP taken and carried together from one reading** | **5,253,941 rows** | All checks pass (below) |

Checks run on the final version:
- 5,253,941 rows, all 52,248 stays, 0 duplicate (stay, hour) rows
- Missing share: latest_HR 3.0%, latest_RR 3.8%, latest_SpO2 4.7%, latest_Temp 6.0%, latest_SBP/DBP 4.4%, latest_MAP 5.2% (the old final table had 5.7–80.8% missing in these columns)
- 0 rows with SBP ≤ DBP; 0 rows with MAP outside [DBP − 5, SBP + 5]
- All values inside the cleaned ranges; medians are clinically normal (HR 85, RR 20, SpO₂ 98, Temp 36.9, SBP 119, DBP 59, MAP 77)

**Check it (optional):**
```sql
SELECT COUNT(*)                                  AS rows_,
       COUNT(DISTINCT icustay_id)                AS stays,
       ROUND(1 - COUNT(latest_HR)   / COUNT(*), 3) AS hr_missing,
       ROUND(1 - COUNT(latest_MAP)  / COUNT(*), 3) AS map_missing,
       ROUND(1 - COUNT(latest_Temp) / COUNT(*), 3) AS temp_missing,
       COUNTIF(latest_SBP <= latest_DBP)         AS impossible_bp
FROM `mimic-project-2.dataset_v2.latest_vitals`;
```
Expected: 5,253,941 · 52,248 · 0.030 · 0.052 · 0.060 · 0.

---

## Command 7 — Stage 6: GCS eye opening

Builds `GCS_eye_opening` (1–4) for every stay-hour: the most recent eye-opening score charted by the nurses, carried forward for a limited time.

**Status:** tested by running it and checking the result; the carry-forward limit was chosen by comparing four alternatives (see the end of this section).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.gcs_hourly` AS
WITH eye_in_hour AS (                        -- 1. newest valid eye-opening score within each hour
  SELECT
    h.icustay_id,
    h.hr_idx,
    ARRAY_AGG(g.GCSEyes IGNORE NULLS ORDER BY TIMESTAMP(g.charttime) DESC LIMIT 1)[SAFE_OFFSET(0)] AS eye
  FROM `mimic-project-2.dataset_v2.vitals_hourly`     AS h
  JOIN `physionet-data.mimiciii_derived.pivoted_gcs`  AS g
    ON g.icustay_id = h.icustay_id
   AND TIMESTAMP(g.charttime) >= h.starttime
   AND TIMESTAMP(g.charttime) <  h.endtime
  WHERE g.GCSEyes BETWEEN 1 AND 4
  GROUP BY h.icustay_id, h.hr_idx
),
timeline AS (                                -- 2. every stay-hour, with that hour's score (if any)
  SELECT
    h.subject_id, h.hadm_id, h.icustay_id, h.hr_idx, h.starttime, h.endtime,
    e.eye
  FROM `mimic-project-2.dataset_v2.vitals_hourly` AS h
  LEFT JOIN eye_in_hour AS e
    ON e.icustay_id = h.icustay_id
   AND e.hr_idx     = h.hr_idx
),
carried AS (                                 -- 3. most recent score so far, and the hour it was charted
  SELECT
    *,
    LAST_VALUE(eye IGNORE NULLS) OVER w                          AS eye_c,
    LAST_VALUE(IF(eye IS NULL, NULL, hr_idx) IGNORE NULLS) OVER w AS eye_at
  FROM timeline
  WINDOW w AS (PARTITION BY icustay_id ORDER BY hr_idx ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT                                       -- 4. keep a carried score for up to 8 hours
  subject_id, hadm_id, icustay_id, hr_idx, starttime, endtime,
  IF(hr_idx - eye_at <= 8, eye_c, NULL) AS GCS_eye_opening
FROM carried;
```

**Expected result:** **5,253,941** rows (one per stay-hour), all **52,248** stays.

### Why each part, with the evidence

**Source: `mimiciii_derived.pivoted_gcs`, not item 220739 only**
- The old query used only item **220739**, MetaVision's GCS-eye item. CareVue (the older system, more than half of our stays) records it as item **184**, which was never collected — the main reason the old `GCS_eye_opening` was **77% empty**.
- `pivoted_gcs` combines both systems: inside our cohort it has **1,509,958** eye-opening scores from **52,161** of 52,248 stays.

**Only whole scores 1–4** — the GCS eye scale has exactly four values (1 = no eye opening, 2 = to pain, 3 = to voice, 4 = spontaneously). `pivoted_gcs` contains only these values plus 1,421 empty ones, which the filter removes. The old table had 71 impossible fractional values (1.5, 2.5, 3.67…) because it *averaged* scores within 10-minute windows; this version takes the newest single score instead, so every value is a real score.

**Newest score in the hour, not an average** — the same reasoning as Command 6: the score that applies at the end of the hour is the most recently charted one. This also matches the dashboard, where the camera's live eye-opening score is the current value.

**Carried forward for up to 8 hours — chosen by comparing alternatives**
- GCS is charted much less often than vitals: the time between eye-opening scores has a median of about **2–4 hours** (nurses usually assess neurology every 2–4 hours; 90% of intervals are under about 5½ hours).
- Without carrying forward, **71%** of hours would have no score. Checked for each carry limit — how much is still missing, and how often the carried score equals the *next* real score (i.e. is still correct):

  | Carry limit | Hours still missing | Carried score still correct, by age of the score |
  |---|---|---|
  | none | 71.3% | — |
  | 4 h | 8.1% | 1–2 h old: 82% · 3–4 h old: 83% |
  | 6 h | 5.4% | |
  | **8 h** | **4.0%** | 5–8 h old: 79% |
  | 12 h | 3.0% | 9–12 h old: 77% |
  | (24 h) | | 13–24 h old: 67% |

- **8 hours** covers two normal assessment intervals while keeping the carried score correct about 79% of the time. Going to 12 hours gains only 1 point of coverage while accuracy keeps falling; beyond that, a score is wrong a third of the time.

**Sedated and intubated patients are kept as charted** — intubated patients have a lower average eye score (2.6 vs 3.8), largely because of sedation. The score still reflects how responsive the patient is, which is what the SOFA brain component measures, so it is not altered.

**Scores outside the stay are ignored** — 12,221 scores were charted before admission or after discharge; only scores inside the stay's hours are used.

### Test result

| Iteration | Change | Result |
|---|---|---|
| 1 | Newest score per hour from `pivoted_gcs`, scores 1–4 only, carried up to 8 h (limit chosen from the comparison above) | 5,253,941 rows; all checks pass |

Checks run on the final version:
- 5,253,941 rows, all 52,248 stays, 0 duplicate (stay, hour) rows
- Missing share **4.0%** (the old final table: **77%** for the same stays)
- Every value is exactly 1, 2, 3 or 4: E4 3,237,722 · E3 908,047 · E2 324,039 · E1 576,299
- Clinically meaningful: correlation with `SOFA_24hours` −0.47 and with the SOFA brain sub-score −0.64 (lower eye score → higher SOFA), much stronger than any single vital sign

**Check it (optional):**
```sql
SELECT COUNT(*)                                       AS rows_,
       COUNT(DISTINCT icustay_id)                     AS stays,
       ROUND(1 - COUNT(GCS_eye_opening) / COUNT(*), 3) AS missing,
       COUNTIF(GCS_eye_opening NOT IN (1, 2, 3, 4))   AS invalid
FROM `mimic-project-2.dataset_v2.gcs_hourly`;
```
Expected: 5,253,941 · 52,248 · 0.040 · 0.

---

## Command 8 — Stage 7: stress score (nurse-recorded pain)

Builds `stress_score` (0–10) for every stay-hour from the pain scores nurses chart, carried forward for a limited time. It uses the same kind of measurement as the old column (nurse pain level), but from both MIMIC systems and without the "Unable to Score" error.

**Status:** tested by running it and checking the result; the carry-forward limit was chosen by comparing alternatives (see the end of this section).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.stress_hourly` AS
WITH pain AS (                               -- 1. pain scores from both systems, converted to 0-10
  SELECT
    ce.ICUSTAY_ID           AS icustay_id,
    TIMESTAMP(ce.CHARTTIME) AS charttime,
    CASE
      WHEN ce.ITEMID = 223791 AND ce.VALUE != 'Unable to Score'                 -- MetaVision: number stored in VALUENUM
        THEN ce.VALUENUM
      WHEN ce.ITEMID = 1044                                                     -- CareVue: text such as "5-Moderate"
        THEN SAFE_CAST(REGEXP_EXTRACT(ce.VALUE, r'^(\d+)-') AS FLOAT64)
    END AS pain_value
  FROM `physionet-data.mimiciii_clinical.chartevents` AS ce
  JOIN `mimic-project-2.dataset_v2.cohort`            AS c
    ON ce.ICUSTAY_ID = c.icustay_id
   AND TIMESTAMP(ce.CHARTTIME) BETWEEN c.intime AND c.outtime
  WHERE ce.ITEMID IN (223791, 1044)
    AND (ce.ERROR IS NULL OR ce.ERROR = 0)                                      -- drop entries marked as errors
),
valid AS (
  SELECT * FROM pain WHERE pain_value BETWEEN 0 AND 10
),
pain_in_hour AS (                            -- 2. newest valid score within each hour
  SELECT
    h.icustay_id,
    h.hr_idx,
    ARRAY_AGG(p.pain_value ORDER BY p.charttime DESC LIMIT 1)[SAFE_OFFSET(0)] AS pain_value
  FROM `mimic-project-2.dataset_v2.vitals_hourly` AS h
  JOIN valid AS p
    ON p.icustay_id = h.icustay_id
   AND p.charttime >= h.starttime
   AND p.charttime <  h.endtime
  GROUP BY h.icustay_id, h.hr_idx
),
timeline AS (                                -- 3. every stay-hour, with that hour's score (if any)
  SELECT
    h.subject_id, h.hadm_id, h.icustay_id, h.hr_idx, h.starttime, h.endtime,
    p.pain_value
  FROM `mimic-project-2.dataset_v2.vitals_hourly` AS h
  LEFT JOIN pain_in_hour AS p
    ON p.icustay_id = h.icustay_id
   AND p.hr_idx     = h.hr_idx
),
carried AS (                                 -- 4. most recent score so far, and the hour it was charted
  SELECT
    *,
    LAST_VALUE(pain_value IGNORE NULLS) OVER w                          AS pain_c,
    LAST_VALUE(IF(pain_value IS NULL, NULL, hr_idx) IGNORE NULLS) OVER w AS pain_at
  FROM timeline
  WINDOW w AS (PARTITION BY icustay_id ORDER BY hr_idx ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT                                       -- 5. keep a carried score for up to 8 hours
  subject_id, hadm_id, icustay_id, hr_idx, starttime, endtime,
  IF(hr_idx - pain_at <= 8, pain_c, NULL) AS stress_score
FROM carried;
```

**Expected result:** **5,253,941** rows (one per stay-hour), all **52,248** stays.

### Why each part, with the evidence

**What this column is** — in MIMIC-III the only bedside measurement close to "stress" is the nurse-recorded **pain level** (0 = no pain, 10 = worst pain). The old column was built from the same measurement (item 223791, saved as `pain_score` and renamed `stress_score`), so the meaning of the column does not change; only its quality does. *(On the dashboard, the camera's PSPI facial stress score is fed into this column; that is a related but different measurement and should be stated as such.)*

**Both MIMIC systems** — the old query used only MetaVision item **223791** (21,595 of our stays). CareVue records pain as item **1044** "Pain Level" (18,008 stays) as text such as `"5-Moderate"` or `"10-Worst"`; the number before the dash is the score, so it is extracted with a regular expression (`'^(\d+)-'`). Together: **37,133 stays** with at least one pain score, against about 21,600 before.

**Why items 524, 1045 and 523 are not added** — checked:
- **524** "Pain Level (Rest)" is charted at **exactly the same time** as 1044 in all 140,352 cases (same values), so it is a duplicate copy of 1044.
- **1045** "Pain Level/Response" is a re-assessment after a treatment (for example after a pain drug), not the patient's routine pain level.
- **523** "Pain Level (Movement)" has only 693 entries.

**"Unable to Score" is treated as missing, not as 0 (the most important fix)**
- In MetaVision, "Unable to Score" is stored with **VALUENUM = 0** — exactly the same number as "None" (no pain). It is the **most common** pain entry: 118,250 of 337,148 (35%).
- The old pipeline used VALUENUM directly, so every unresponsive or deeply sedated patient was recorded as "pain 0" — the same as a comfortable patient. In the old `ml_dataset_final`, 148,551 of the 243,344 non-empty stress values were 0, a large share of them actually "Unable to Score".
- Checked: at the hours when "Unable to Score" was charted, patients' GCS eye score averaged **2.87**, against **3.52** when "None" was charted — these are less responsive patients, not pain-free ones.
- "Unable to Score" now becomes missing; CareVue's equivalent `"Unable to score"` has no number before a dash, so it becomes missing automatically.

**Entries marked as errors are removed** — 339 MetaVision pain entries have `ERROR = 1` (charted by mistake).

**Newest score in the hour, carried forward up to 8 hours**
- Pain is assessed about every 2–4 hours (median interval about 4 hours).
- Compared carry limits (missing share of all hours, and the average error of a carried score against the next real score):

  | Carry limit | Hours still missing | Error by age of the carried score |
  |---|---|---|
  | 4 h | 78.8% | 1–4 h old: 1.3 points |
  | **8 h** | **73.4%** | 5–8 h old: 1.7 points |
  | 12 h | 69.6% | 9–12 h old: 2.2 points |

- 8 hours matches the GCS limit, covers two assessment intervals, and keeps the carried score within about 1.7 points on average. Longer limits add little coverage while the error keeps growing.

**Why the column is still often empty — and why that is correct**
- Even in stays where pain *is* charted, a valid score appears in only 6.3% of hours (9% including "Unable to Score"), and 15,115 stays never have a valid score. These gaps are real: pain is not charted every hour, and patients who cannot report pain have no score.
- Filling them would invent data. They are left missing and handled in Python (training median plus the model's handling of missing values), as for every other column.
- The pattern itself is informative: hours with no pain score have an average SOFA of 5.98, hours with one 4.72 (sicker patients are more often unable to report pain).

### Test iterations

| Iteration | Change | Result |
|---|---|---|
| 1 | Both systems (223791 + 1044), "Unable to Score" and error entries excluded, newest score per hour, carried up to 8 h | 5,253,941 rows; all checks pass |

Checks run on the final version:
- 5,253,941 rows, all 52,248 stays, 0 duplicate (stay, hour) rows
- Every value a whole number from 0 to 10 (0: 437,826 · 5: 196,445 · 10: 25,086)
- Missing share **73.4%**, against **95.9%** in the old final table for the same stays; 37,133 stays have at least one score
- No "Unable to Score" entry counted as 0

**Check it (optional):**
```sql
SELECT COUNT(*)                                    AS rows_,
       COUNT(DISTINCT icustay_id)                  AS stays,
       ROUND(1 - COUNT(stress_score) / COUNT(*), 3) AS missing,
       MIN(stress_score)                           AS min_score,
       MAX(stress_score)                           AS max_score
FROM `mimic-project-2.dataset_v2.stress_hourly`;
```
Expected: 5,253,941 · 52,248 · 0.734 · 0 · 10.

---

## Command 9a — Free storage before continuing

The project has reached BigQuery's **free storage limit (10 GB)**: the old `Dataset` uses 6.7 GB and `dataset_v2` 3.0 GB. Creating another table fails with *"Quota exceeded: Your project exceeded quota for free storage"*.

The old pipeline's **intermediate** tables are not read by any code. `train_federated.py` reads only `Dataset.ml_dataset_final` and `Dataset.clinical_notes` (checked in the code), and those two are kept until the new tables replace them in Command 12. The intermediate tables were only steps used to build `ml_dataset_final`, and every query that made them is recorded in `sql_queries.pdf`, so they can be rebuilt at any time.

```sql
DROP TABLE IF EXISTS `mimic-project-2.Dataset.vitals_raw`;        -- 1.51 GB
DROP TABLE IF EXISTS `mimic-project-2.Dataset.vitals_pivoted`;    -- 0.53 GB
DROP TABLE IF EXISTS `mimic-project-2.Dataset.vitals_10min`;      -- 0.48 GB
DROP TABLE IF EXISTS `mimic-project-2.Dataset.vital_trends_20`;   -- 0.65 GB
DROP TABLE IF EXISTS `mimic-project-2.Dataset.latest_vitals`;     -- 0.48 GB
DROP TABLE IF EXISTS `mimic-project-2.Dataset.ml_dataset`;        -- 0.95 GB
```

**Frees about 4.6 GB** (from 9.65 GB down to about 5.0 GB), enough for all remaining stages.

**Not deleted, and why:**

| Table | Why it stays |
|---|---|
| `ml_dataset_final`, `clinical_notes` | Read by `train_federated.py`; replaced only in Command 12, after the new tables are checked |
| `gcs_eye`, `stress_score`, `cv_features_10min`, `sofa_scores` | Very small (0.07 GB together); kept for comparison |

Each `DROP` can be run on its own; BigQuery runs them one after another when pasted together.

---

## Command 9 — Stage 8: the SOFA label (time-matched)

Builds `sofa_score` for every stay-hour: the SOFA score **for that hour**, taken from the hourly SOFA table. This is the most important change in the whole rebuild.

**Status:** tested by running it and checking the result (2 iterations, see the end of this section).

```sql
CREATE OR REPLACE TABLE `mimic-project-2.dataset_v2.labels` AS
WITH sofa AS (                               -- 1. one SOFA row per stay-hour (removes duplicate hours)
  SELECT
    icustay_id,
    hr                     AS hr_idx,
    SOFA_24hours           AS sofa_score,
    respiration_24hours    AS sofa_respiration,
    coagulation_24hours    AS sofa_coagulation,
    liver_24hours          AS sofa_liver,
    cardiovascular_24hours AS sofa_cardiovascular,
    cns_24hours            AS sofa_cns,
    renal_24hours          AS sofa_renal
  FROM `physionet-data.mimiciii_derived.pivoted_sofa`
  WHERE TRUE
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY icustay_id, hr
    ORDER BY SOFA_24hours DESC, respiration_24hours DESC, cardiovascular_24hours DESC,
             renal_24hours DESC, cns_24hours DESC, coagulation_24hours DESC, liver_24hours DESC
  ) = 1
)
SELECT                                       -- 2. attach the label to our hourly timeline
  h.subject_id,
  h.hadm_id,
  h.icustay_id,
  h.hr_idx,
  h.starttime,
  h.endtime,
  s.sofa_score,
  s.sofa_respiration,
  s.sofa_coagulation,
  s.sofa_liver,
  s.sofa_cardiovascular,
  s.sofa_cns,
  s.sofa_renal
FROM `mimic-project-2.dataset_v2.vitals_hourly` AS h
JOIN sofa AS s
  ON s.icustay_id = h.icustay_id
 AND s.hr_idx     = h.hr_idx;
```

**Expected result:** **5,253,941** rows (one per stay-hour), all **52,248** stays, every row with a SOFA score.

### Why each part, with the evidence

**Why the label had to change — the old label described a different time**
- The old label came from `mimiciii_derived.sofa`: **one score per ICU stay**, the worst SOFA of the **first 24 hours**. That single number was copied onto every window of the stay — including windows days or weeks later.
- The old training data used the *last* window of each stay, which falls a median of **48 hours** after admission (only 23% inside the first 24 hours). So the model was mostly shown day-2-to-day-100 vital signs and asked to predict a day-1 score. A patient who arrived very sick and recovered showed calm vitals labelled "SOFA 12".
- Checked: within a stay, the new hourly label changes in **96.4%** of stays (median range 5 points). The old label was constant by design, so it hid all of this change.

**The new label: `SOFA_24hours` from `pivoted_sofa`**
- For each hour, the SOFA score over the **24 hours ending at that hour** — the standard way SOFA is defined (the worst value of each organ component over a 24-hour period).
- It only looks **backwards**, so it never uses information from after the hour. The vital-sign features (Commands 4–8) also only look backwards, so features and label describe the same past period.
- Joined on `(icustay_id, hr_idx)` — exactly the same hourly timeline the features use (Command 4), so every row's features and label match.

**The label lines up much better with the patient's condition** (correlation with the features at the same hour, 10% sample):

| Feature | Old label (first-day SOFA) | New label (hourly) |
|---|---|---|
| GCS eye opening | −0.14 | **−0.47** |
| MAP_mean | −0.12 | **−0.15** |
| HR_mean | +0.20 | +0.08 |

GCS — a direct part of SOFA — is now three times more strongly linked to the label. (The old HR link was higher because, with a fixed first-day label, early tachycardia stood in for overall severity.)

**Duplicate hours — one complete row kept (fixed in iteration 2)**
- `pivoted_sofa` lists 12,388 of our stay-hours more than once; in 852 the copies disagree (by 1–6 points), for example 8 vs 9 when two different PaO₂/FiO₂ measurements were recorded in the same hour.
- Iteration 1 took the maximum of each column *separately*. That produced 15 rows where the total no longer equalled the sum of its six organ scores, because the maximum total and the maximum of each organ came from different copies.
- Iteration 2 keeps **one whole row** per hour (`QUALIFY ROW_NUMBER() ... = 1`): the copy with the **highest total**, and among equal totals, the highest organ scores. Taking the higher value follows how SOFA is defined (the worst value in the period). Now the total always equals the sum of its parts.

**Organ sub-scores are kept for checking only** — `sofa_respiration`, `sofa_coagulation`, `sofa_liver`, `sofa_cardiovascular`, `sofa_cns` and `sofa_renal` (each 0–4) are stored to verify the label and to analyse results. They are **not** model inputs and are not copied into the final dataset: they are parts of the answer, so using them as inputs would leak it.

**The first 24 hours of a stay** — for the first day, the 24-hour window reaches back before admission, so the score covers less than a full day of ICU data. These rows are 22% of the table; their average SOFA (5.8) is close to later hours (5.7), so they are kept. This matches how SOFA is used clinically on day 1.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | Hourly `SOFA_24hours`; duplicate hours combined by taking the maximum of each column | 5,253,941 rows | 15 rows where the total ≠ sum of the six organ scores (values from different copies mixed) |
| 2 | **Keep one complete row per duplicate hour (highest total)** | **5,253,941 rows** | All checks pass (below) |

*(Iteration 2 was checked with a read-only query because the storage limit blocked creating test tables; the checks are identical.)*

Checks run on the final version:
- 5,253,941 rows, all 52,248 stays, 0 duplicate (stay, hour) rows, **0 missing labels**
- SOFA 0–24; each organ score 0–4; total = sum of organ scores in **every** row
- Average 5.70 (old first-day label: 4.17 for the same stays)
- Risk bands: Low 2,329,842 (44%) · Moderate 2,129,834 (41%) · High 794,265 (**15%**)
- The label changes over time in 96.4% of stays

**Check it (optional):**
```sql
SELECT COUNT(*)                   AS rows_,
       COUNT(DISTINCT icustay_id) AS stays,
       COUNTIF(sofa_score IS NULL) AS missing,
       MIN(sofa_score) AS min_sofa, MAX(sofa_score) AS max_sofa,
       ROUND(AVG(sofa_score), 2)  AS mean_sofa,
       COUNTIF(sofa_score != sofa_respiration + sofa_coagulation + sofa_liver
                           + sofa_cardiovascular + sofa_cns + sofa_renal) AS sum_mismatch
FROM `mimic-project-2.dataset_v2.labels`;
```
Expected: 5,253,941 · 52,248 · 0 · 0 · 24 · 5.70 · 0.

---

## Command 10 — Stage 9: the final `ml_dataset_final` table

Joins the label and all feature tables into **one table with exactly the same 23 columns, names and types** as the old `ml_dataset_final`, so `train_federated.py` reads it without any code change.

**Status:** tested by running it and checking the result, including the exact query `train_federated.py` runs on it (2 iterations, see the end of this section).

```sql
CREATE OR REPLACE VIEW `mimic-project-2.dataset_v2.ml_dataset_final` AS
SELECT
  l.subject_id,
  l.hadm_id,
  l.icustay_id,
  l.endtime          AS window_time,      -- end of the hour the row describes
  -- trend columns (Command 5)
  t.HR_mean,
  t.HR_std,
  t.RR_mean,
  t.SpO2_mean,
  t.SpO2_min,
  t.Temp_mean,
  t.SBP_mean,
  t.DBP_mean,
  t.MAP_mean,
  -- latest-vital columns (Command 6)
  v.latest_HR,
  v.latest_RR,
  v.latest_SpO2,
  v.latest_Temp,
  v.latest_SBP,
  v.latest_DBP,
  v.latest_MAP,
  -- label (Command 9)
  l.sofa_score,
  -- observation columns (Commands 7 and 8)
  g.GCS_eye_opening,
  s.stress_score
FROM `mimic-project-2.dataset_v2.labels`        AS l
JOIN `mimic-project-2.dataset_v2.vital_trends`  AS t ON t.icustay_id = l.icustay_id AND t.hr_idx = l.hr_idx
JOIN `mimic-project-2.dataset_v2.latest_vitals` AS v ON v.icustay_id = l.icustay_id AND v.hr_idx = l.hr_idx
JOIN `mimic-project-2.dataset_v2.gcs_hourly`    AS g ON g.icustay_id = l.icustay_id AND g.hr_idx = l.hr_idx
JOIN `mimic-project-2.dataset_v2.stress_hourly` AS s ON s.icustay_id = l.icustay_id AND s.hr_idx = l.hr_idx
WHERE v.latest_HR IS NOT NULL                                          -- a current heart rate, and
  AND COALESCE(v.latest_MAP, v.latest_SBP, v.latest_SpO2) IS NOT NULL; -- a current blood pressure or SpO2
```

**Expected result:** **5,089,857** rows · **52,248** stays · **38,304** patients.

> **Why this is a VIEW, not a TABLE.** Creating it as a table failed with *"Quota exceeded: Your project exceeded quota for free storage"*, even though the tables listed in the project add up to only about 3.5 GB. BigQuery keeps deleted data recoverable for a period after it is dropped (time travel, 7 days by default), so the space freed by deleting the old `Dataset` may not count as free straight away.
>
> A view stores **no data**: it is a saved query that BigQuery runs whenever the view is read. It has exactly the same 23 columns, names and types, and `train_federated.py` reads it with its normal query (`SELECT * ... FROM ml_dataset_final`) without any change. This was checked: the exact Phase 0 query was run on the view and returned the expected 52,248 rows.
>
> The only differences: each read re-runs the joins (it scans about 2.7 GB, well inside BigQuery's free 1 TB of queries per month), and the view needs the five tables it reads (`labels`, `vital_trends`, `latest_vitals`, `gcs_hourly`, `stress_hourly`) to stay in place. If storage frees up later, the same query can be run with `CREATE OR REPLACE TABLE` instead.

### Why each part, with the evidence

**Exactly the same columns as before** — 23 columns in the same order with the same names and types (`subject_id`, `hadm_id`, `icustay_id` INTEGER; `window_time` TIMESTAMP; 16 vital columns FLOAT; `sofa_score` INTEGER; `GCS_eye_opening`, `stress_score` FLOAT). Checked against the old table's schema. `train_federated.py` selects `*`, keeps the newest row per stay, and drops the four ID/time columns, so nothing in the Python code needs to change.

**`window_time` = the end of the hour** — each row describes one hour, and every feature and the label summarise readings *up to the end* of that hour, so the end time is the honest timestamp. It is also what `train_federated.py` orders by when it picks the newest row per stay.

**Joining on `(icustay_id, hr_idx)`** — every stage was built on the same hourly timeline (Command 4), so each table has exactly one row per stay-hour and the joins are one-to-one. Checked: 0 duplicate (stay, time) rows after the join, and no rows are lost by the joins (every stage has all 5,253,941 stay-hours).

**Rows without a current heart rate and a current blood pressure or SpO₂ are removed (added in iteration 2)**
- **Why:** a row only describes the patient if the basic vital signs were actually measured around that hour. In iteration 1, **137,622 rows** had no vital value at all, and many more had none of the key vitals — typically hours in long charting gaps, such as when the patient was off the unit for a procedure. Such rows have nothing for the model to learn from; Phase 0 would fill every empty value with 0, which the model reads as "heart rate 0".
- **Why heart rate plus BP or SpO₂:** heart rate is the most continuously charted vital; blood pressure or oxygen saturation is the second core measurement of every bedside check. Requiring both still keeps **97%** of rows (5,089,857 of 5,253,941).
- **Checked, on the rows `train_federated.py` actually uses** (the newest row per stay), comparing the filters:

  | Rows kept | Newest-row latest_HR missing | MAP_mean missing | latest_MAP missing | GCS missing | Corr. latest_MAP vs SOFA |
  |---|---|---|---|---|---|
  | all hours | 9.5% | 8.0% | 14.8% | 5.6% | −0.31 |
  | at least one vital | 9.5% | 3.5% | 14.8% | 4.0% | −0.31 |
  | **current HR + (BP or SpO₂)** | **0.0%** | **2.2%** | **6.0%** | **3.6%** | **−0.38** |

  Every stay still has a row (52,248), and the link between the vitals and the label becomes stronger, because the newest row is now a moment when the patient was actually being monitored.

**All usable hours are stored, not just the last one** — the table keeps every stay-hour (5.09 M rows), like the old table (7.6 M rows). `train_federated.py` currently keeps only the newest row per stay; storing all hours means later work can use more of each stay without rebuilding the dataset. Using more rows per stay was compared (one row every 4 or 6 hours gives 600–840 thousand rows and 15–16% High Risk rows, against 5.6% with only the last row); it would need a small change in `train_federated.py`, so it is noted as a future improvement, not done here.

**Nothing is filled in here** — empty values stay NULL (empty), never 0. The final table is a faithful record; how the model handles missing values is decided in Python, where the training data's medians are known.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | Join all stages into the 23 columns | 5,253,941 rows | 137,622 rows with no vital at all; in the rows Phase 0 uses, latest_HR 9.5% and latest_MAP 14.8% missing |
| 2 | **Keep only rows with a current heart rate and a current BP or SpO₂** | **5,089,857 rows** | All checks pass (below) |

*(While testing this command the storage freed by deleting the old `Dataset` had not yet registered, so iteration 1 was checked with read-only queries; iteration 2 was created as a test table and checked in full.)*

Checks run on the final version:
- 23 columns, same names, order and types as the old `ml_dataset_final`
- 5,089,857 rows, 52,248 stays, 38,304 patients, 0 duplicate (stay, time) rows, 0 missing labels
- Missing share in **all rows**: 0–5.3% for every vital column (HR_std 5.3%, latest_Temp 3.5%, everything else under 3%), GCS 1.7%, stress 72.9%
- All rows: average SOFA 5.79 · Low 2,205,622 · Moderate 2,097,412 · High 786,823 (15.5%)
- **The exact query `train_federated.py` runs** (newest row per stay) returns **52,248 rows**: latest_HR 0% missing, MAP_mean 2.2%, GCS 3.6%, stress 60.8% (old table: 45%, 83% and 97% missing); GCS–SOFA correlation −0.54, MAP–SOFA −0.35

**Comparison with the old table** (rows `train_federated.py` uses — newest row per stay):

| | Old | New |
|---|---|---|
| Stays | 60,188 (incl. 8,000 newborns) | 52,248 adults |
| Label | first-day SOFA, copied to every hour | SOFA of that same hour |
| Impossible values (HR 9,999,999, SpO₂ 152,040…) | present | none |
| MAP_mean missing | 45% | 2.2% |
| latest_MAP missing | 64% | 6.0% |
| latest_Temp missing | 81% | ≈ 4% |
| GCS_eye_opening missing | 83% | 3.6% |
| stress_score missing | 97% ("Unable to Score" counted as 0) | 61% (not counted as 0) |
| GCS ↔ SOFA correlation | −0.14 | −0.54 |

**Check it (optional) — runs the same selection as `train_federated.py`:**
```sql
SELECT COUNT(*)                                  AS stays,
       ROUND(AVG(sofa_score), 2)                 AS mean_sofa,
       COUNTIF(sofa_score >= 10)                 AS high_risk,
       ROUND(AVG(IF(MAP_mean IS NULL, 1, 0)), 3)        AS map_mean_missing,
       ROUND(AVG(IF(GCS_eye_opening IS NULL, 1, 0)), 3) AS gcs_missing
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY window_time DESC) AS rn
  FROM `mimic-project-2.dataset_v2.ml_dataset_final`
  WHERE sofa_score IS NOT NULL
)
WHERE rn = 1;
```
Expected: 52,248 · 3.89 · 3,008 · 0.022 · 0.036.

---

## Command 11 — Stage 10: the final `clinical_notes` table

Builds the notes table `train_federated.py` reads, with **exactly the same 5 columns, names and types** as the old `clinical_notes` (`subject_id` INTEGER, `hadm_id` INTEGER, `chartdate` DATE, `category` STRING, `text` STRING). It keeps only bedside notes written in the **24 hours before** the moment each stay is scored, newest first.

**Status:** tested by running it and checking the result (2 iterations, see the end of this section). Created as a **view** for the same storage reason as Command 10.

```sql
CREATE OR REPLACE VIEW `mimic-project-2.dataset_v2.clinical_notes` AS
WITH last_row AS (                           -- 1. the moment each stay is scored (the row Phase 0 uses)
  SELECT icustay_id, hadm_id, window_time
  FROM `mimic-project-2.dataset_v2.ml_dataset_final`
  WHERE TRUE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY window_time DESC) = 1
),
picked AS (                                  -- 2. bedside notes from the 24 hours before that moment
  SELECT DISTINCT
    n.ROW_ID, n.SUBJECT_ID, n.HADM_ID, n.CHARTDATE, n.CHARTTIME, n.CATEGORY, n.TEXT
  FROM last_row AS l
  JOIN `mimic-project-2.dataset_v2.cohort`        AS c USING (icustay_id)
  JOIN `physionet-data.mimiciii_notes.noteevents` AS n ON n.HADM_ID = l.hadm_id
  WHERE n.CATEGORY IN ('Nursing', 'Nursing/other', 'Physician ', 'Respiratory ')  -- exact MIMIC spellings
    AND n.ISERROR IS NULL                                     -- not marked as written in error
    AND n.CHARTTIME IS NOT NULL                               -- exact time known
    AND LENGTH(n.TEXT) >= 50                                  -- not an empty or one-word note
    AND TIMESTAMP(n.CHARTTIME) >= c.intime                    -- written during the ICU stay
    AND TIMESTAMP(n.CHARTTIME) <= l.window_time               -- nothing from after the scored moment
    AND TIMESTAMP(n.CHARTTIME) >  TIMESTAMP_SUB(l.window_time, INTERVAL 24 HOUR)
),
deduped AS (                                 -- 3. drop copy-pasted duplicate notes
  SELECT *
  FROM picked
  WHERE TRUE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY HADM_ID, TEXT ORDER BY CHARTTIME, ROW_ID) = 1
)
SELECT                                       -- 4. one row per admission, newest note first
  ANY_VALUE(SUBJECT_ID)                                          AS subject_id,
  HADM_ID                                                        AS hadm_id,
  MAX(CHARTDATE)                                                 AS chartdate,
  STRING_AGG(DISTINCT CATEGORY, ', ' ORDER BY CATEGORY)          AS category,
  STRING_AGG(TEXT, '\n\n' ORDER BY CHARTTIME DESC, ROW_ID DESC)  AS text
FROM deduped
GROUP BY HADM_ID;
```

**Expected result:** **35,799** rows (one per admission) covering **38,322** of the 52,248 stays.

### Why each part, with the evidence

**The old notes table had three problems**
1. **Notes from after the outcome.** It contained discharge summaries, written at the end of the stay, and 49% of its text was written after the moment being scored. The model could partly read the answer from the notes, and the dashboard never has such notes at the bedside.
2. **Most bedside notes were missing.** The old query asked for the categories `'Physician'` and `'Nursing/Other'`, but MIMIC stores them as `'Physician '` (with a trailing space) and `'Nursing/other'` (lower-case "other"), so 141,624 physician notes and 822,497 nursing notes were silently excluded.
3. **Random text was kept.** `train_federated.py` joins an admission's notes in random order and keeps the first 10,000 characters, and 55% of admissions are longer than that, so which part survived was chance.

**Only bedside categories** — inside our cohort's ICU stays, by category:

| Category | Notes | Kept? | Why |
|---|---|---|---|
| Nursing/other | 415,729 | **Yes** | Bedside nursing notes (CareVue era) |
| Nursing | 220,399 | **Yes** | Bedside nursing notes (MetaVision era) |
| Physician | 139,759 | **Yes** | Daily physician progress notes |
| Respiratory | 31,602 | **Yes** | Ventilator and breathing notes; directly about the lung SOFA component |
| Radiology | 208,813 | No | Imaging reports, not bedside notes like the dashboard's |
| Nutrition, rehab, social work, case management, pharmacy, consult, general | about 27,000 | No | Not about organ failure |
| Discharge summary | — | No | Written after the stay; leaks the outcome |

Checked: adding radiology *lowers* the link between note words and SOFA (0.43 vs 0.51), so it is left out.

**Only the 24 hours before the scored moment — chosen by comparing windows** (link between SOFA-related words in the notes and the SOFA label, on the rows Phase 0 uses):

| Notes used | Stays with notes | Correlation with SOFA |
|---|---|---|
| whole ICU stay | 38,409 | 0.39 |
| **24 h before the scored moment** | **38,281** | **0.51** |
| 12 h before | 34,029 | 0.45 |

- The 24-hour window matches what notes are meant to tell the model — the patient's *current* state — and it matches the SOFA label, which also covers the 24 hours ending at that moment.
- 12 hours loses 4,000 stays (many get only one note per shift) and is weaker; the whole stay mixes in days-old notes.
- **Nothing after the scored moment is included** (`CHARTTIME <= window_time`), so the notes cannot reveal the future.

**The scored moment is the newest row of each stay in `ml_dataset_final`** — exactly the row `train_federated.py` keeps (`ROW_NUMBER() ... ORDER BY window_time DESC = 1`). So for every stay, the notes cover the 24 hours leading up to the same hour as the vital signs and the label.

**Removed: error notes, empty notes and copy-paste duplicates**
- `ISERROR` marks notes written in error (about 700 in these categories).
- Notes under 50 characters are empty templates such as "FULL CODE" or "TITLE:".
- Nurses often copy a note forward; exact duplicate texts within an admission are kept only once (the earliest), so a pasted template does not count twice.
- Notes without an exact `CHARTTIME` are excluded, because their timing cannot be checked.

**One row per admission, newest note first (added in iteration 2)**
- `train_federated.py` groups notes by `hadm_id`, joins them, and keeps the first 10,000 characters. In iteration 1 the table kept one row per note, so the script's random shuffle again decided which text survived.
- Now the table already holds **one row per admission**, with its notes joined **newest first**. Whatever the script does, the first 10,000 characters are the most recent notes, closest to the scored moment.
- 8,784 admissions (25%) are longer than 10,000 characters; for them only the oldest notes of the 24 hours are cut off.
- The five columns keep their meaning: `chartdate` is the date of the newest note, and `category` lists the note types included (for example `"Nursing, Physician "`).

**About 13,900 stays have no notes** — they had no bedside note in the 24 hours before the scored moment. They are not removed: `train_federated.py` joins notes with a left join and gives these stays all-zero word scores, the same as before.

**Notes signal checked the way Phase 0 uses it** — counting SOFA-related words in the first 10,000 characters of each admission's text:

| | Correlation with SOFA |
|---|---|
| Old-style notes (nursing + discharge summaries, random order) | 0.31 |
| **New notes (bedside, last 24 h, newest first)** | **0.50** |

The new notes carry about 60% more SOFA-related signal, without any leakage from after the outcome.

### Test iterations

| Iteration | Change | Result | Finding |
|---|---|---|---|
| 1 | Bedside categories, last 24 h before the scored moment, no errors/empty/duplicates; one row per note | 169,901 notes, 35,799 admissions | Correct notes, but one row per note lets the script's random shuffle decide which text survives the 10,000-character cut |
| 2 | **One row per admission, notes joined newest first** | **35,799 rows** | All checks pass (below) |

Checks run on the final version:
- 5 columns, same names and types as the old `clinical_notes`
- 35,799 rows = 35,799 distinct admissions; 28,614 patients; each admission belongs to exactly one patient
- 0 discharge summaries; 0 notes after the scored moment; 0 duplicate texts within an admission
- 38,322 of 52,248 stays have notes
- SOFA-word signal 0.50 vs 0.31 for old-style notes

**Check it (optional):**
```sql
SELECT COUNT(*)                   AS admissions,
       COUNT(DISTINCT subject_id) AS patients,
       APPROX_QUANTILES(LENGTH(text), 2)[OFFSET(1)] AS median_chars,
       COUNTIF(category LIKE '%Discharge%')          AS discharge_summaries
FROM `mimic-project-2.dataset_v2.clinical_notes`;
```
Expected: 35,799 · 28,614 · about 2,800 · 0.
