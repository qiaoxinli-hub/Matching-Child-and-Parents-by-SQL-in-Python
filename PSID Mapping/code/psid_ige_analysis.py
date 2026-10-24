#!/usr/bin/env python3
"""PSID parent-child income and IGE analysis using IND23 individual incomes.

IND23 individual income fields are used with the existing codebook cleaning.
No Family-file correction is applied; Head/Wife fields may be incomplete. Only the two final IGE result tables are saved.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IND = PROJECT_ROOT / "data" / "individual" / "IND2023ER.txt"
DEFAULT_PID = PROJECT_ROOT / "data" / "parents" / "pid23" / "PID23.txt"
DEFAULT_OUT = PROJECT_ROOT / "results"

INCOME_COMPONENTS: dict[int, tuple[str, ...]] = {
    1967: ("ER30012",),
    1968: ("ER30033",),
    1969: ("ER30057",),
    1970: ("ER30081",),
    1971: ("ER30106",),
    1972: ("ER30130",),
    1973: ("ER30152",),
    1974: ("ER30173", "ER30175"),
    1975: ("ER30202", "ER30209"),
    1976: ("ER30231", "ER30238"),
    1977: ("ER30268", "ER30275"),
    1978: ("ER30298", "ER30305"),
    1979: ("ER30328", "ER30335"),
    1980: ("ER30358", "ER30365"),
    1981: ("ER30386", "ER30391"),
    1982: ("ER30415", "ER30420"),
    1983: ("ER30445", "ER30455"),
    1984: ("ER30480", "ER30490"),
    1985: ("ER30515", "ER30525"),
    1986: ("ER30551", "ER30561"),
    1987: ("ER30586", "ER30596"),
    1988: ("ER30622", "ER30632"),
    1989: ("ER30659", "ER30669"),
    1990: ("ER30705", "ER30707", "ER30717"),
    1991: ("ER30750", "ER30752", "ER30762"),
    1992: ("ER30821", "ER30822", "ER30825"),
}

# Annual-average CPI-U, U.S. city average, all items, 1982-84=100.
# Values are from the BLS historical CPI-U table. Default dollars are 1992 USD.
CPI_U_1967_1992: dict[int, float] = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5, 1972: 41.8,
    1973: 44.4, 1974: 49.3, 1975: 53.8, 1976: 56.9, 1977: 60.6, 1978: 65.2,
    1979: 72.6, 1980: 82.4, 1981: 90.9, 1982: 96.5, 1983: 99.6, 1984: 103.9,
    1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0,
    1990: 130.7, 1991: 136.2, 1992: 140.3,
}

FIELD_RE = re.compile(r"\b(?:long\s+)?(ER\d{5}|PID\d{1,2})\s+(\d+)\s*-\s*(\d+)", re.I)
LABEL_RE = re.compile(r'^\s*label\s+variable\s+([A-Z0-9]+)\s+"([^"]*)"', re.I | re.M)


def parse_stata_layout(path: Path) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    """Parse fixed-width positions and labels from a PSID Stata .do file."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    positions: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        for match in FIELD_RE.finditer(line):
            name = match.group(1).upper()
            positions[name] = (int(match.group(2)) - 1, int(match.group(3)))
    labels = {name.upper(): label.strip() for name, label in LABEL_RE.findall(text)}
    if not positions:
        raise ValueError(f"No fixed-width fields found in {path}")
    return positions, labels


def decode_wave_year(two_digits: str) -> int:
    year = int(two_digits)
    return 1900 + year if year >= 68 else 2000 + year


def get_wave_fields(labels: dict[str, str]) -> tuple[dict[int, str], dict[int, str]]:
    """Map interview years to their age and individual-record-type fields."""
    age_fields: dict[int, str] = {}
    type_fields: dict[int, str] = {}
    patterns = (
        (r"AGE OF INDIVIDUAL\s+(\d{2})\s*$", age_fields),
        (r"TYPE OF IND RECORD\s+(\d{2})\s*$", type_fields),
    )
    for variable, label in labels.items():
        for pattern, target in patterns:
            match = re.search(pattern, label, re.I)
            if match:
                target[decode_wave_year(match.group(1))] = variable
    return age_fields, type_fields


def read_selected_file(
    data_path: Path,
    variables: Iterable[str],
    format_path: Path | None = None,
    chunk_size: int = 20000,
) -> pd.DataFrame:
    """Read selected columns from PSID CSV or fixed-width text."""
    wanted = [name.upper() for name in variables]
    if data_path.suffix.lower() == ".csv":
        header = pd.read_csv(data_path, nrows=0)
        names = {str(col).strip().upper(): col for col in header.columns}
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise ValueError(f"{data_path.name} is missing columns: {missing}")
        frame = pd.read_csv(data_path, usecols=[names[name] for name in wanted], dtype="string")
        frame.columns = [str(col).strip().upper() for col in frame.columns]
        frame = frame[wanted]
        for col in wanted:
            frame[col] = pd.to_numeric(frame[col].str.strip(), errors="coerce")
        return frame

    if format_path is None or not format_path.exists():
        raise FileNotFoundError(f"Need the matching Stata .do file for {data_path}")
    positions, _ = parse_stata_layout(format_path)
    missing = sorted(set(wanted) - set(positions))
    if missing:
        raise ValueError(f"{format_path.name} has no positions for: {missing}")
    colspecs = [positions[name] for name in wanted]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_fwf(
        data_path,
        colspecs=colspecs,
        names=wanted,
        dtype="string",
        chunksize=chunk_size,
        encoding="ascii",
    ):
        for col in wanted:
            chunk[col] = pd.to_numeric(chunk[col].str.strip(), errors="coerce")
        pieces.append(chunk)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=wanted)


def build_individual_income(
    ind_data: Path,
    ind_format: Path,
) -> pd.DataFrame:
    """Build person-year income; apply codebook-specific cleaning for 1967-73."""
    positions, labels = parse_stata_layout(ind_format)
    age_fields, type_fields = get_wave_fields(labels)
    required = {"ER30001", "ER30002"}
    for income_year, components in INCOME_COMPONENTS.items():
        interview_year = income_year + 1
        for field_map, description in (
            (age_fields, "AGE OF INDIVIDUAL"),
            (type_fields, "TYPE OF IND RECORD"),
        ):
            if interview_year not in field_map:
                raise ValueError(f"No {description} field for interview year {interview_year}")
            required.add(field_map[interview_year])
        required.update(components)
    missing = sorted(required - set(positions))
    if missing:
        raise ValueError(f"IND format is missing required variables: {missing}")

    wanted = ["ER30001", "ER30002"] + sorted(required - {"ER30001", "ER30002"})
    wide = read_selected_file(ind_data, wanted, ind_format)
    records: list[pd.DataFrame] = []

    for income_year, components in INCOME_COMPONENTS.items():
        interview_year = income_year + 1
        age_field = age_fields[interview_year]
        type_field = type_fields[interview_year]
        values = wide[list(components)].apply(pd.to_numeric, errors="coerce").copy()
        if income_year <= 1973:
            for variable in components:
                width = positions[variable][1] - positions[variable][0]
                all_nines = 10**width - 1
                values[variable] = values[variable].mask(values[variable].eq(all_nines))

        income = (
            values.iloc[:, 0]
            if len(components) == 1
            else values.sum(axis=1, min_count=len(components))
        )
        age = pd.to_numeric(wide[age_field], errors="coerce")
        record_type = pd.to_numeric(wide[type_field], errors="coerce")
        ids_present = wide["ER30001"].notna() & wide["ER30002"].notna()
        valid_record_type = record_type.eq(0)
        valid_age = age.between(1, 98)
        keep = ids_present & valid_record_type & valid_age
        records.append(
            pd.DataFrame(
                {
                    "id68": wide["ER30001"].where(keep),
                    "person_number": wide["ER30002"].where(keep),
                    "income_year": income_year,
                    "age": age.where(keep),
                    "income_nominal": income.where(keep),
                }
            ).loc[keep]
        )
    person_year = pd.concat(records, ignore_index=True)
    person_year["id68"] = person_year["id68"].astype("int64")
    person_year["person_number"] = person_year["person_number"].astype("int64")
    person_year["income_year"] = person_year["income_year"].astype("int64")
    return person_year


def apply_deflator(
    person_year: pd.DataFrame,
    cpi_csv: Path | None,
    base_year: int,
) -> pd.DataFrame:
    result = person_year.copy()
    if cpi_csv is None:
        cpi_by_year = pd.Series(CPI_U_1967_1992, dtype="float64")
    else:
        cpi = pd.read_csv(cpi_csv)
        cpi.columns = [str(c).strip().lower() for c in cpi.columns]
        if not {"year", "cpi"}.issubset(cpi.columns):
            raise ValueError("CPI CSV needs columns year,cpi")
        cpi["year"] = pd.to_numeric(cpi["year"], errors="coerce")
        cpi["cpi"] = pd.to_numeric(cpi["cpi"], errors="coerce")
        cpi = cpi.dropna(subset=["year", "cpi"])
        cpi["year"] = cpi["year"].astype("int64")
        cpi_by_year = cpi.drop_duplicates("year").set_index("year")["cpi"]
    if base_year not in cpi_by_year.index:
        raise ValueError(f"CPI series has no base-year value for {base_year}")
    year_index = result["income_year"].map(cpi_by_year)
    missing = sorted(result.loc[year_index.isna(), "income_year"].unique().tolist())
    if missing:
        raise ValueError(f"CPI series is missing income years: {missing}")
    result["cpi_income_year"] = year_index
    result["income_real"] = (
        result["income_nominal"] * float(cpi_by_year.loc[base_year]) / result["cpi_income_year"]
    )
    result["income_real_base_year"] = base_year
    return result


def build_parent_child_pairs(
    pid_data: Path,
    pid_format: Path | None,
) -> pd.DataFrame:
    names = ["PID2", "PID3", "PID4", "PID5", "PID23", "PID24"]
    pid = read_selected_file(pid_data, names, pid_format)
    pid = pid.rename(
        columns={
            "PID2": "child_id68", "PID3": "child_pn",
            "PID4": "mother_id68", "PID5": "mother_pn",
            "PID23": "father_id68", "PID24": "father_pn",
        }
    )
    for col in pid.columns:
        pid[col] = pd.to_numeric(pid[col], errors="coerce")

    children = pid.dropna(subset=["child_id68", "child_pn"]).copy()
    children = children[(children["child_id68"] > 0) & (children["child_pn"] > 0) & (children["child_pn"] < 999)]
    links: list[pd.DataFrame] = []
    for parent_type, id_col, pn_col in (
        ("mother", "mother_id68", "mother_pn"),
        ("father", "father_id68", "father_pn"),
    ):
        one = children[["child_id68", "child_pn", id_col, pn_col]].rename(
            columns={id_col: "parent_id68", pn_col: "parent_pn"}
        )
        valid = (
            one["parent_id68"].notna()
            & one["parent_pn"].notna()
            & one["parent_id68"].between(1, 9998)
            & one["parent_pn"].between(1, 998)
        )
        cleaned = one.loc[valid].copy()
        cleaned["parent_type"] = parent_type
        cleaned = cleaned.drop_duplicates(
            ["parent_type", "child_id68", "child_pn", "parent_id68", "parent_pn"]
        )
        for col in ["child_id68", "child_pn", "parent_id68", "parent_pn"]:
            cleaned[col] = cleaned[col].astype("int64")
        links.append(cleaned)
    pairs = pd.concat(links, ignore_index=True) if links else pd.DataFrame()
    return pairs


def create_sql_tables(connection: sqlite3.Connection, person_year: pd.DataFrame) -> None:
    columns = ["id68", "person_number", "income_year", "age", "income_real"]
    person_year[columns].to_sql("person_year", connection, if_exists="replace", index=False, chunksize=50000)


AVERAGE_INCOME_SQL = """
SELECT id68, person_number,
       AVG(income_real) AS avg_income,
       COUNT(income_real) AS n_income_years
FROM person_year
WHERE income_real > 0 AND age BETWEEN ? AND ?
GROUP BY id68, person_number
HAVING COUNT(income_real) >= ?
"""

PARENT_YEAR_INCOME_SQL = """
SELECT id68, person_number, income_year, income_year + 1 AS survey_year,
       age AS parent_age, income_real AS parent_year_income
FROM person_year
WHERE income_year BETWEEN ? AND ?
"""


def build_person_income_measures(
    person_year: pd.DataFrame,
    age_min: int,
    age_max: int,
    min_years: int,
    survey_year_start: int,
    survey_year_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(":memory:") as connection:
        create_sql_tables(connection, person_year)
        average_income = pd.read_sql_query(AVERAGE_INCOME_SQL, connection, params=(age_min, age_max, min_years))
        parent_year_income = pd.read_sql_query(
            PARENT_YEAR_INCOME_SQL,
            connection,
            params=(survey_year_start - 1, survey_year_end - 1),
        )
    average_income["id68"] = average_income["id68"].astype("int64")
    average_income["person_number"] = average_income["person_number"].astype("int64")
    parent_year_income["id68"] = parent_year_income["id68"].astype("int64")
    parent_year_income["person_number"] = parent_year_income["person_number"].astype("int64")
    parent_year_income["survey_year"] = parent_year_income["survey_year"].astype("int64")
    return average_income, parent_year_income


JOIN_SQL = """
SELECT py.survey_year, py.income_year, py.parent_age,
       p.parent_type, p.child_id68, p.child_pn, p.parent_id68, p.parent_pn,
       py.parent_year_income,
       ca.avg_income AS child_average_income,
       pa.avg_income AS parent_average_income,
       ca.n_income_years AS child_n_income_years,
       pa.n_income_years AS parent_n_income_years
FROM parent_child AS p
JOIN parent_year_income AS py
  ON py.id68 = p.parent_id68 AND py.person_number = p.parent_pn
LEFT JOIN average_income AS ca
  ON ca.id68 = p.child_id68 AND ca.person_number = p.child_pn
LEFT JOIN average_income AS pa
  ON pa.id68 = p.parent_id68 AND pa.person_number = p.parent_pn
ORDER BY py.survey_year, p.parent_type, p.child_id68, p.child_pn
"""


def join_parent_child_income(
    pairs: pd.DataFrame,
    average_income: pd.DataFrame,
    parent_year_income: pd.DataFrame,
) -> pd.DataFrame:
    with sqlite3.connect(":memory:") as connection:
        pairs.to_sql("parent_child", connection, index=False)
        average_income.to_sql("average_income", connection, index=False)
        parent_year_income.to_sql("parent_year_income", connection, index=False)
        joined = pd.read_sql_query(JOIN_SQL, connection)
    return joined


def estimate_ige(joined: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except ImportError as error:
        raise ImportError("Install statsmodels: python -m pip install statsmodels") from error

    rows: list[dict[str, object]] = []
    for survey_year in sorted(joined["survey_year"].dropna().unique()):
        wave = joined.loc[joined["survey_year"] == survey_year].copy()
        positive = (
            (wave["parent_year_income"] > 0)
            & (wave["parent_average_income"] > 0)
            & (wave["child_average_income"] > 0)
        )
        sample = wave.loc[positive].copy()
        if sample.empty:
            continue
        sample["ln_parent_year_income"] = np.log(sample["parent_year_income"].astype(float))
        sample["ln_parent_average"] = np.log(sample["parent_average_income"].astype(float))
        sample["ln_child_average"] = np.log(sample["child_average_income"].astype(float))
        for parent_type in ("mother", "father"):
            subset = sample.loc[sample["parent_type"] == parent_type].copy()
            for measure, xcol in (
                ("single_year_parent_income", "ln_parent_year_income"),
                ("parent_age_30_50_average", "ln_parent_average"),
            ):
                if len(subset) < 3:
                    continue
                design = sm.add_constant(subset[xcol].to_numpy(dtype=float), has_constant="add")
                cluster = subset["parent_id68"].astype(str) + ":" + subset["parent_pn"].astype(str)
                model = sm.OLS(subset["ln_child_average"].to_numpy(dtype=float), design).fit(
                    cov_type="cluster", cov_kwds={"groups": cluster}
                )
                ci = model.conf_int(alpha=0.05)[1]
                rows.append(
                    {
                        "survey_year": int(survey_year),
                        "income_year": int(subset["income_year"].iloc[0]),
                        "parent_type": parent_type,
                        "income_measure": measure,
                        "ige_beta": float(model.params[1]),
                        "cluster_robust_se": float(model.bse[1]),
                        "p_value": float(model.pvalues[1]),
                        "ci_95_low": float(ci[0]),
                        "ci_95_high": float(ci[1]),
                        "n_parent_child_pairs": int(model.nobs),
                        "n_parent_clusters": int(cluster.nunique()),
                        "mean_parent_age": float(subset["parent_age"].mean()),
                        "r_squared": float(model.rsquared),
                    }
                )
    estimates = pd.DataFrame(rows)
    return estimates


AGE_BAND_QUERY = """
WITH child_band_income AS (
    SELECT id68, person_number, AVG(income_real) AS band_income,
           COUNT(income_real) AS n_band_years
    FROM person_year
    WHERE income_real > 0 AND age BETWEEN ? AND ?
    GROUP BY id68, person_number
), parent_average AS (
    SELECT id68, person_number, AVG(income_real) AS avg_income,
           COUNT(income_real) AS n_income_years
    FROM person_year
    WHERE income_real > 0 AND age BETWEEN ? AND ?
    GROUP BY id68, person_number
    HAVING COUNT(income_real) >= ?
)
SELECT p.parent_type, p.child_id68, p.child_pn, p.parent_id68, p.parent_pn,
       cb.band_income AS child_band_income,
       cb.n_band_years AS child_band_n_years,
       pa.avg_income AS parent_average_income,
       pa.n_income_years AS parent_n_income_years
FROM parent_child AS p
LEFT JOIN child_band_income AS cb
  ON cb.id68 = p.child_id68 AND cb.person_number = p.child_pn
LEFT JOIN parent_average AS pa
  ON pa.id68 = p.parent_id68 AND pa.person_number = p.parent_pn
ORDER BY p.parent_type, p.child_id68, p.child_pn
"""


def estimate_age_profile(
    person_year: pd.DataFrame,
    pairs: pd.DataFrame,
    age_min: int,
    age_max: int,
    min_years: int,
) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except ImportError as error:
        raise ImportError("Install statsmodels: python -m pip install statsmodels") from error
    bands = [(20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49)]
    band_frames: list[pd.DataFrame] = []
    with sqlite3.connect(":memory:") as connection:
        create_sql_tables(connection, person_year)
        pairs.to_sql("parent_child", connection, index=False)
        for low, high in bands:
            frame = pd.read_sql_query(
                AGE_BAND_QUERY,
                connection,
                params=(low, high, age_min, age_max, min_years),
            )
            frame["child_age_band"] = f"{low}-{high}"
            band_frames.append(frame)
    measures = pd.concat(band_frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for (parent_type, age_band), subset in measures.groupby(["parent_type", "child_age_band"]):
        data = subset.dropna(subset=["parent_average_income", "child_band_income"]).copy()
        data = data[(data["parent_average_income"] > 0) & (data["child_band_income"] > 0)]
        if len(data) < 3:
            continue
        x = np.log(data["parent_average_income"].astype(float).to_numpy())
        y = np.log(data["child_band_income"].astype(float).to_numpy())
        design = sm.add_constant(x, has_constant="add")
        cluster = data["parent_id68"].astype(str) + ":" + data["parent_pn"].astype(str)
        model = sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": cluster})
        ci = model.conf_int(alpha=0.05)[1]
        rows.append(
            {
                "parent_type": parent_type,
                "child_age_band": age_band,
                "ige_beta": float(model.params[1]),
                "cluster_robust_se": float(model.bse[1]),
                "p_value": float(model.pvalues[1]),
                "ci_95_low": float(ci[0]),
                "ci_95_high": float(ci[1]),
                "n_parent_child_pairs": int(model.nobs),
                "n_parent_clusters": int(cluster.nunique()),
            }
        )
    estimates = pd.DataFrame(rows)
    return estimates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ind-data", type=Path, default=DEFAULT_IND)
    parser.add_argument("--ind-format", type=Path, default=None)
    parser.add_argument("--pid-data", type=Path, default=DEFAULT_PID)
    parser.add_argument("--pid-format", type=Path, default=None)
    parser.add_argument("--cpi-csv", type=Path, default=None, help="Optional CPI override CSV with year,cpi")
    parser.add_argument("--base-year", type=int, default=1992)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--age-min", type=int, default=30)
    parser.add_argument("--age-max", type=int, default=50)
    parser.add_argument("--survey-year-start", type=int, default=1988)
    parser.add_argument("--survey-year-end", type=int, default=1993)
    parser.add_argument("--min-average-years", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in ((args.ind_data, "IND data"), (args.pid_data, "PID data")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.age_min > args.age_max:
        raise ValueError("--age-min cannot exceed --age-max")
    if args.survey_year_start > args.survey_year_end:
        raise ValueError("--survey-year-start cannot exceed --survey-year-end")
    if args.survey_year_start < 1968 or args.survey_year_end > 1993:
        raise ValueError("This income-component map supports survey years 1968-1993")
    if args.min_average_years < 1:
        raise ValueError("--min-average-years must be at least 1")

    ind_format = args.ind_format or args.ind_data.with_suffix(".do")
    pid_format = args.pid_format or args.pid_data.with_suffix(".do")
    if not ind_format.exists():
        raise FileNotFoundError(f"IND format file not found: {ind_format}")
    if args.pid_data.suffix.lower() != ".csv" and not pid_format.exists():
        raise FileNotFoundError(f"PID format file not found: {pid_format}")

    results_dir = args.outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("## STEP 1: Read IND23 person-year income")
    person_year = build_individual_income(args.ind_data, ind_format)
    person_year = apply_deflator(person_year, args.cpi_csv, args.base_year)
    missing_income_rows = int(person_year["income_nominal"].isna().sum())
    print(f"IND person-year records: {len(person_year):,}")
    print(f"Records with missing IND income: {missing_income_rows:,}")

    print()
    print("## STEP 2: Build parent-child links from PID23")
    pairs = build_parent_child_pairs(args.pid_data, pid_format)

    print()
    print("## STEP 3: Calculate person income measures")
    averages, parent_year_income = build_person_income_measures(
        person_year, args.age_min, args.age_max, args.min_average_years,
        args.survey_year_start, args.survey_year_end,
    )

    print()
    print("## STEP 4: Join parent and child income measures")
    joined = join_parent_child_income(pairs, averages, parent_year_income)

    print()
    print("## STEP 5: Estimate IGE by survey year")
    estimates = estimate_ige(joined)
    estimates.to_csv(results_dir / "ige_estimates_by_survey_year.csv", index=False, encoding="utf-8-sig")

    print()
    print("## STEP 6: Estimate IGE by child's age band")
    profile = estimate_age_profile(
        person_year, pairs, args.age_min, args.age_max, args.min_average_years,
    )
    profile.to_csv(results_dir / "ige_age_profile_estimates.csv", index=False, encoding="utf-8-sig")


    print()
    print("## RUN SUMMARY")
    print("Income basis: IND23 individual income after codebook cleaning; no Family-file correction")
    print(f"Person-year records: {len(person_year):,}; missing IND income: {missing_income_rows:,}")
    print(f"Parent-child links: {len(pairs):,}; joined rows: {len(joined):,}")
    print(f"IGE estimates: {len(estimates):,}; age-profile estimates: {len(profile):,}")
    print(f"Saved the two IGE tables in: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
