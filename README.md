Matching Children and Parents by SQL in Python

This repository reproduces the PSID parent–child income analysis described in the accompanying tutorial. It uses the 1968–2023 Individual Data File (IND23) to construct income histories, the Parent Identification File (PID23) to identify parent–child links, SQL in Python's built-in `sqlite3` module to aggregate and join records, and log–log OLS regressions to estimate intergenerational income elasticities (IGEs).
The analysis addresses two measurement problems. A parent's income in one year can contain temporary shocks and may be observed at a particular point in the life cycle. Averaging income over several positive-income years provides a more stable proxy for longer-run economic status, although the average is not a perfect measure of permanent income.

## Repository organization

The current project uses the following layout:

```text
PSID Mapping/
├── README.md
├── code/
│   └── psid_ige_analysis.py
├── data/
│   └── data/
│       ├── individual/
│       │   ├── IND2023ER.txt
│       │   └── IND2023ER.do
│       └── parents/
│           └── pid23/
│               ├── PID23.txt
│               └── PID23.do
└── results/
    └── results/
        ├── ige_estimates_by_survey_year.csv
        └── ige_age_profile_estimates.csv
```

The duplicated `data/data/` and `results/results/` levels reflect the current project and script settings. The run command below passes the actual input paths explicitly. With the script's default output setting, `results/` is appended to the default `<project>/results` path, so the CSV files are written to `<project>/results/results/`. To write directly to `<project>/results/`, add `--outdir .` to the command.
The `.do` files are required: the script reads their fixed-width field positions and variable labels. The other PSID codebook and format files in the data directories are reference materials; the Python script does not execute Stata, SAS, or SPSS code.

## Requirements and setup

Use Python 3.10 or newer. The script requires `numpy`, `pandas`, and `statsmodels`. `argparse`, `re`, `sqlite3`, and `pathlib` are part of Python's standard library; SQLite data tables are created in memory.
From the repository root, create an environment and install the packages:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas statsmodels
```
Use the same activated environment to run the script. If using Anaconda, an environment with Python 3.10+ and those three packages is sufficient.

## Replication instructions

Run the command from the repository root. These paths match the current `data/data/` layout and avoid the script defaults, which look for files one level higher under `data/`:
```powershell
python .\code\psid_ige_analysis.py --ind-data .\data\data\individual\IND2023ER.txt --ind-format .\data\data\individual\IND2023ER.do --pid-data .\data\data\parents\pid23\PID23.txt --pid-format .\data\data\parents\pid23\PID23.do
```
The command uses the script's default analysis settings:

- Survey years: 1988–1993, corresponding to income years 1987–1992.
- Parent and child long-run income window: interview ages 30–50.
- Minimum observations for an age-30–50 average: 3 positive-income years.
- Real-dollar base year: 1992.
- Output directory: `results/results/` as explained above.

The script includes annual-average CPI-U values for 1967–1992. An optional CPI CSV can replace them with `--cpi-csv PATH`; it must contain numeric `year` and `cpi` columns, including the selected base year and all income years used. Other settings can be changed with `--base-year`, `--age-min`, `--age-max`, `--min-average-years`, `--survey-year-start`, `--survey-year-end`, and `--outdir`. Run `python .\code\psid_ige_analysis.py --help` to see the command-line options.

## Data preparation and parent–child matching

### 1. Read person and income records

The IND23 file contains repeated observations for individuals. The script uses `ER30001` as the 1968 interview identifier and `ER30002` as the person number, then reads the needed income, age, and record-type fields from the fixed-width file. It keeps valid individual records with both identifiers, interview age from 1 to 98, and individual record type 0.

The code maps survey waves 1968–1993 to calendar income years 1967–1992. Thus, a survey in year `t + 1` reports income for year `t`. Income components vary by year: the script uses the IND23 components mapped in `INCOME_COMPONENTS`, sums multiple components where specified, and applies the early-wave all-nines missing-value cleaning described in the tutorial. It then converts nominal income to 1992 dollars using:

```text
real income = nominal income × CPI(base year) / CPI(income year)
```

### 2. Construct parent–child links from PID23

PID23 supplies the child's identifiers (`PID2`, `PID3`), mother's identifiers (`PID4`, `PID5`), and father's identifiers (`PID23`, `PID24`). The script reshapes these into one link per child–parent relationship, labels each link `mother` or `father`, validates identifier ranges, and removes duplicate links.
The match uses both the 1968 interview identifier and the person number. The person number alone is not a unique identifier across the full panel. PID23 records the parent identifiers used here; the script does not independently verify biological relationships.

### 3. Aggregate income with SQL

The script writes the cleaned person-year panel and the parent links to temporary SQLite tables held in memory. The core SQL operations solve different parts of the matching and measurement problem:

- **`SELECT`** chooses the identifiers and measures to return. It can also create derived fields, such as `survey_year = income_year + 1`, and rename columns with `AS`.
- **`WHERE`** filters individual rows before aggregation. For the long-run measure, it keeps positive real-income observations within the specified age window.
- **`GROUP BY id68, person_number`** gathers a person's repeated yearly rows into one group. Grouping by both identifiers preserves the correct person identity.
- **Aggregation (`AVG`, `COUNT`)** calculates each person's mean real income and the number of observed positive-income years contributing to that mean.
- **`HAVING COUNT(...) >= 3`** filters the completed person-level groups, retaining only people with at least three qualifying years. `HAVING` is used because this condition applies to the aggregated count; it cannot be applied as a row-level `WHERE` condition.
- **`JOIN`** combines tables using matching identifiers. The parent-year query uses an inner `JOIN` to require an observed parent income record for that survey wave. The query then `LEFT JOIN`s the average-income table once for the child and once for the parent. This preserves the matched parent-year row even when a long-run average is unavailable; those missing or nonpositive measures are excluded from the regression sample later.
- **`ORDER BY`** sorts the joined rows by survey year, parent type, and child identifiers for predictable output.

In simplified form, the long-run query is:
```sql
SELECT id68, person_number,
       AVG(income_real) AS avg_income,
       COUNT(income_real) AS n_income_years
FROM person_year
WHERE income_real > 0 AND age BETWEEN 30 AND 50
GROUP BY id68, person_number
HAVING COUNT(income_real) >= 3;
```
This converts multiple yearly observations into one income measure per person. A separate query preserves each parent's annual income observation for the single-year comparison.

## IGE estimation and the measurement questions

For each survey year and separately for mothers and fathers, the script estimates two log–log models:
```text
log(child's average real income) = intercept + β × log(parent's income) + error
```
The dependent variable is the child's average positive real income at interview ages 30–50, requiring at least three years. The parent measure is either:

1. **Single-year income:** the parent's real income for the income year corresponding to that survey wave; or
2. **Multi-year average:** the parent's average positive real income at interview ages 30–50, also requiring at least three years.

For a fair comparison, both models for a given survey year and parent type use the same observations: child average income, parent single-year income, and parent multi-year average must all be positive. Standard errors are clustered by the parent's interview identifier and person number so that siblings linked to the same parent are treated as a cluster.
The coefficient `β` is the IGE estimate. It describes the association between a one-percent difference in parental income and the percentage difference in child income. A higher estimate indicates stronger intergenerational income persistence in the estimation sample; it is not by itself a causal effect.

### Why compare one year with an average?

One year's income can reflect temporary events such as a bonus, a layoff, or illness. Averaging several positive-income observations can reduce the influence of these short-term fluctuations and provide a more stable proxy for long-run economic status. It does not perfectly measure permanent income, and this implementation restricts the average to observed interview ages 30–50 with at least three positive-income years.
The tutorial reports that multi-year estimates are generally larger and less variable than the single-year estimates. This is consistent with attenuation from transitory income noise, but the comparison does not isolate that mechanism: the single-year parent observations occur when parents' mean interview ages are roughly 57–65, while the multi-year average uses ages 30–50. Age-related income patterns may therefore also contribute to the difference.

### How the age-profile analysis relates to life-cycle bias

A single observation at a particular age may not represent a person's longer-run income. To examine this issue, the script estimates separate models for child interview-age bands 20–24, 25–29, 30–34, 35–39, 40–44, and 45–49. Each child's band income is compared with the parent's age-30–50 average.
The tutorial finds that older child age bands generally have higher IGE estimates than ages 20–24, a pattern consistent with life-cycle bias in measured income. It is not a definitive causal test: each age band uses the parent–child pairs available in that band, not one balanced panel of the same people. Changing sample composition and the requirement for positive income may also affect the estimates. The number of pairs falls sharply in the oldest band, so those estimates are less precise.

## Generated files

The script creates two CSVs programmatically and writes them under the selected results directory:
- **`ige_estimates_by_survey_year.csv`** — estimates by survey year, parent type, and parent income measure. Includes the IGE coefficient, clustered standard error, p-value, 95% confidence interval, pair and parent-cluster counts, mean parent age, and R-squared.
- **`ige_age_profile_estimates.csv`** — estimates by parent type and child age band, with clustered standard errors, confidence intervals, and sample counts.
