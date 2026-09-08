"""
LTPP Excel-to-CSV Converter
============================
Converts the FHWA LTPP Bucket_145337.xlsx file (which contains multiple
spreadsheets/tables) into individual CSV files for each sheet.

Additionally extracts pothole-specific distress records and maps them
to our Shallow/Moderate/Deep severity scale.

Usage:
    python scripts/convert_ltpp_xlsx.py
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

try:
    import openpyxl
except ImportError:
    print("ERROR: openpyxl is required. Install with: pip install openpyxl")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LTPP_RAW_DIR = PROJECT_ROOT / "datasets" / "ltpp" / "raw"
LTPP_PROCESSED_DIR = PROJECT_ROOT / "datasets" / "ltpp" / "processed"
XLSX_PATH = LTPP_RAW_DIR / "Bucket_145337.xlsx"


def safe_value(val):
    """Convert cell values to CSV-safe strings."""
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    return str(val)


def export_all_sheets(wb, output_dir):
    """Export every sheet in the workbook to a separate CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        # Sanitize sheet name for filename
        safe_name = sheet_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        csv_path = output_dir / f"{safe_name}.csv"

        row_count = 0
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in ws.iter_rows(values_only=True):
                writer.writerow([safe_value(v) for v in row])
                row_count += 1

        exported.append((sheet_name, csv_path.name, row_count))
        print(f"  Exported '{sheet_name}' -> {csv_path.name} ({row_count} rows)")

    return exported


def extract_pothole_severity(wb, output_dir):
    """
    Extract pothole distress records from LTPP tables and map to severity.

    Two relevant tables:
    - MON_DIS_PADIAS42_AC (newer format):
        POTHOLES_NO_L/M/H  = count by severity
        POTHOLES_A_L/M/H   = area by severity
    - MON_DIS_PADIAS_AC (older format):
        POTHOLES_NO = total count
        POTHOLES_A  = total area
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    severity_csv = output_dir / "ltpp_pothole_severity.csv"
    records = []

    # --- Process MON_DIS_PADIAS42_AC (newer, has L/M/H breakdown) ---
    sheet_name = "MON_DIS_PADIAS42_AC"
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) > 1:
            headers = [str(h).strip() if h else "" for h in rows[0]]

            # Find column indices
            col_map = {}
            target_cols = [
                "STATE_CODE_EXP", "SHRP_ID", "SURVEY_DATE", "CONSTRUCTION_NO",
                "POTHOLES_NO_L", "POTHOLES_NO_M", "POTHOLES_NO_H",
                "POTHOLES_A_L", "POTHOLES_A_M", "POTHOLES_A_H",
            ]
            for col_name in target_cols:
                if col_name in headers:
                    col_map[col_name] = headers.index(col_name)

            if "POTHOLES_NO_L" in col_map:
                for row in rows[1:]:
                    try:
                        no_l = float(row[col_map.get("POTHOLES_NO_L", -1)] or 0)
                        no_m = float(row[col_map.get("POTHOLES_NO_M", -1)] or 0)
                        no_h = float(row[col_map.get("POTHOLES_NO_H", -1)] or 0)
                        a_l = float(row[col_map.get("POTHOLES_A_L", -1)] or 0)
                        a_m = float(row[col_map.get("POTHOLES_A_M", -1)] or 0)
                        a_h = float(row[col_map.get("POTHOLES_A_H", -1)] or 0)

                        total_count = no_l + no_m + no_h
                        total_area = a_l + a_m + a_h

                        if total_count > 0 or total_area > 0:
                            # Determine dominant severity
                            if no_h > 0 or a_h > 0:
                                severity_int = 2
                                severity_label = "Deep"
                            elif no_m > 0 or a_m > 0:
                                severity_int = 1
                                severity_label = "Moderate"
                            else:
                                severity_int = 0
                                severity_label = "Shallow"

                            state = safe_value(row[col_map.get("STATE_CODE_EXP", 0)])
                            shrp = safe_value(row[col_map.get("SHRP_ID", 0)])
                            survey_date = safe_value(row[col_map.get("SURVEY_DATE", 0)])
                            constr_no = safe_value(row[col_map.get("CONSTRUCTION_NO", 0)])

                            records.append({
                                "source_table": sheet_name,
                                "state": state,
                                "shrp_id": shrp,
                                "survey_date": survey_date,
                                "construction_no": constr_no,
                                "potholes_count_low": no_l,
                                "potholes_count_med": no_m,
                                "potholes_count_high": no_h,
                                "potholes_area_low": a_l,
                                "potholes_area_med": a_m,
                                "potholes_area_high": a_h,
                                "total_count": total_count,
                                "total_area": total_area,
                                "severity_int": severity_int,
                                "severity_label": severity_label,
                            })
                    except (TypeError, ValueError, IndexError):
                        continue

        print(f"  Extracted {len(records)} pothole records from {sheet_name}")

    # --- Process MON_DIS_PADIAS_AC (older, single count/area) ---
    sheet_name2 = "MON_DIS_PADIAS_AC"
    pre_count = len(records)
    if sheet_name2 in wb.sheetnames:
        ws2 = wb[sheet_name2]
        rows2 = list(ws2.iter_rows(values_only=True))
        if len(rows2) > 1:
            headers2 = [str(h).strip() if h else "" for h in rows2[0]]
            col_map2 = {}
            target_cols2 = [
                "STATE_CODE_EXP", "SHRP_ID", "SURVEY_DATE", "CONSTRUCTION_NO",
                "POTHOLES_NO", "POTHOLES_A",
            ]
            for col_name in target_cols2:
                if col_name in headers2:
                    col_map2[col_name] = headers2.index(col_name)

            if "POTHOLES_NO" in col_map2:
                for row in rows2[1:]:
                    try:
                        p_no = float(row[col_map2.get("POTHOLES_NO", -1)] or 0)
                        p_a = float(row[col_map2.get("POTHOLES_A", -1)] or 0)

                        if p_no > 0 or p_a > 0:
                            # Without L/M/H breakdown, classify by count/area heuristic
                            if p_no >= 5 or p_a >= 5.0:
                                severity_int = 2
                                severity_label = "Deep"
                            elif p_no >= 2 or p_a >= 1.0:
                                severity_int = 1
                                severity_label = "Moderate"
                            else:
                                severity_int = 0
                                severity_label = "Shallow"

                            state = safe_value(row[col_map2.get("STATE_CODE_EXP", 0)])
                            shrp = safe_value(row[col_map2.get("SHRP_ID", 0)])
                            survey_date = safe_value(row[col_map2.get("SURVEY_DATE", 0)])
                            constr_no = safe_value(row[col_map2.get("CONSTRUCTION_NO", 0)])

                            records.append({
                                "source_table": sheet_name2,
                                "state": state,
                                "shrp_id": shrp,
                                "survey_date": survey_date,
                                "construction_no": constr_no,
                                "potholes_count_low": p_no,
                                "potholes_count_med": 0,
                                "potholes_count_high": 0,
                                "potholes_area_low": p_a,
                                "potholes_area_med": 0,
                                "potholes_area_high": 0,
                                "total_count": p_no,
                                "total_area": p_a,
                                "severity_int": severity_int,
                                "severity_label": severity_label,
                            })
                    except (TypeError, ValueError, IndexError):
                        continue

        print(f"  Extracted {len(records) - pre_count} pothole records from {sheet_name2}")

    # --- Write combined severity CSV ---
    if records:
        fieldnames = [
            "source_table", "state", "shrp_id", "survey_date", "construction_no",
            "potholes_count_low", "potholes_count_med", "potholes_count_high",
            "potholes_area_low", "potholes_area_med", "potholes_area_high",
            "total_count", "total_area", "severity_int", "severity_label",
        ]
        with open(severity_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\n  Saved {len(records)} total pothole severity records -> {severity_csv.name}")
    else:
        print("\n  No pothole records found in LTPP data.")

    return records


def main():
    print("=" * 60)
    print("LTPP Excel -> CSV Converter")
    print("=" * 60)

    if not XLSX_PATH.exists():
        print(f"\nERROR: Excel file not found at: {XLSX_PATH}")
        print("Please ensure the LTPP data is in datasets/ltpp/raw/")
        sys.exit(1)

    print(f"\nLoading workbook: {XLSX_PATH.name} ...")
    wb = openpyxl.load_workbook(str(XLSX_PATH), read_only=True)
    print(f"Found {len(wb.sheetnames)} sheets: {wb.sheetnames}\n")

    # Step 1: Export all sheets to CSV
    print("--- Exporting all sheets to CSV ---")
    exported = export_all_sheets(wb, LTPP_PROCESSED_DIR)

    # Step 2: Extract pothole severity
    print("\n--- Extracting pothole severity records ---")
    # Need to reload with read_only=False for random access (read_only has limitations)
    wb.close()
    wb = openpyxl.load_workbook(str(XLSX_PATH), read_only=True)
    records = extract_pothole_severity(wb, LTPP_PROCESSED_DIR)
    wb.close()

    # Summary
    print("\n" + "=" * 60)
    print("LTPP Conversion Summary")
    print("=" * 60)
    print(f"  Sheets exported: {len(exported)}")
    for name, fname, rows in exported:
        print(f"    {fname}: {rows} rows")
    print(f"  Pothole severity records: {len(records)}")
    sev_dist = {}
    for r in records:
        lbl = r["severity_label"]
        sev_dist[lbl] = sev_dist.get(lbl, 0) + 1
    for lbl, cnt in sorted(sev_dist.items()):
        print(f"    {lbl}: {cnt}")
    print(f"  Output directory: {LTPP_PROCESSED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
