#!/usr/bin/env python3
"""
Download and process real ASSISTments 2009-2010 skill builder data.

Source: ASSISTments 2009-2010 (corrected, no duplicates)
  Google Drive file ID: 1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE
  Citation: Feng, Heffernan & Koedinger (2009)

Output: data/processed/assistments/interactions.csv
  Columns: user_id, skill_name, correct, hint_count, attempt_count, start_time
"""

import csv
import io
import os
import sys
import zipfile
from pathlib import Path

# Google Drive direct download trick
GDRIVE_FILE_ID = "0B2X0QD6q79ZJUFU1cjYtdGhVNjg"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# Corrected version (no duplicates)
CORRECTED_FILE_ID = "1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE"
CORRECTED_URL = f"https://drive.google.com/uc?export=download&id={CORRECTED_FILE_ID}"

RAW_DIR = Path("data/raw/assistments")
PROCESSED_DIR = Path("data/processed/assistments")
OUTPUT_CSV = PROCESSED_DIR / "interactions.csv"


def download_from_gdrive(file_id: str, output_path: Path) -> bool:
    """Download a file from Google Drive by file ID."""
    import urllib.request
    import urllib.error

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from Google Drive ({file_id})...")
    print(f"  URL: {url}")

    try:
        # For large files, Google Drive may require confirmation
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        # Check if we got a confirmation page instead of the file
        if b'confirm=' in data and b'virus scan' in data.lower():
            # Extract confirmation token
            import re
            match = re.search(rb'confirm=([a-zA-Z0-9_-]+)', data)
            if match:
                confirm = match.group(1).decode()
                url = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}"
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Mozilla/5.0')
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()

        with open(output_path, 'wb') as f:
            f.write(data)

        size_mb = len(data) / (1024 * 1024)
        print(f"  Downloaded {size_mb:.1f} MB to {output_path}")
        return True

    except urllib.error.URLError as e:
        print(f"  Download failed: {e}")
        return False


def try_gdown(file_id: str, output_path: Path) -> bool:
    """Try using gdown library for Google Drive download."""
    try:
        import gdown
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading via gdown: {url}")
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists() and output_path.stat().st_size > 1000
    except ImportError:
        print("gdown not installed. Trying urllib...")
        return False
    except Exception as e:
        print(f"gdown failed: {e}")
        return False


def process_raw_csv(raw_path: Path, output_path: Path,
                    max_students: int = 500, max_rows: int = 100000) -> dict:
    """
    Process raw ASSISTments CSV into our standardized format.

    Raw columns (ASSISTments 2009-2010):
      order_id, assignment_id, user_id, assistment_id, problem_id,
      original, correct, attempt_count, ms_first_response, tutor_mode,
      answer_type, sequence_id, student_class_id, position, type,
      base_sequence_id, skill_id, skill_name, teacher_id, school_id,
      hint_count, hint_total, overlap_time, template_id

    Output columns:
      user_id, skill_name, correct, hint_count, attempt_count, start_time
    """
    print(f"Processing {raw_path}...")

    # Detect encoding and delimiter
    with open(raw_path, 'rb') as f:
        sample = f.read(2000)

    # Check if it's a zip
    if sample[:2] == b'PK':
        print("  File is a ZIP archive, extracting...")
        with zipfile.ZipFile(raw_path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_names:
                print("  No CSV found in ZIP!")
                return {}
            # Extract first CSV
            extracted = zf.extract(csv_names[0], raw_path.parent)
            raw_path = Path(extracted)
            print(f"  Extracted: {raw_path}")

    # Read and process
    students = {}  # user_id -> list of rows
    total_rows = 0
    skipped = 0

    with open(raw_path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        available_cols = reader.fieldnames or []
        print(f"  Columns: {available_cols[:10]}...")

        for row in reader:
            total_rows += 1
            if total_rows > max_rows * 3:  # Read extra to get diverse students
                break

            uid = row.get('user_id', '').strip()
            skill = row.get('skill_name', '').strip()
            correct_raw = row.get('correct', '').strip()

            if not uid or not skill or correct_raw == '':
                skipped += 1
                continue

            # Skip multi-skill rows (duplicated)
            if ',' in skill:
                skill = skill.split(',')[0].strip()

            if len(students) >= max_students and uid not in students:
                continue

            try:
                correct = int(float(correct_raw))
            except (ValueError, TypeError):
                skipped += 1
                continue

            hint_count = int(float(row.get('hint_count', '0') or '0'))
            attempt_count = int(float(row.get('attempt_count', '1') or '1'))
            ms_first = row.get('ms_first_response', '').strip()

            students.setdefault(uid, []).append({
                'user_id': uid,
                'skill_name': skill,
                'correct': correct,
                'hint_count': hint_count,
                'attempt_count': attempt_count,
                'start_time': ms_first,  # milliseconds since epoch
            })

    # Write processed CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_interactions = 0

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'user_id', 'skill_name', 'correct',
            'hint_count', 'attempt_count', 'start_time'
        ])
        writer.writeheader()

        for uid, rows in sorted(students.items()):
            for row in rows[:max_rows // max_students * 2]:  # Cap per student
                writer.writerow(row)
                n_interactions += 1

    stats = {
        'total_raw_rows': total_rows,
        'skipped': skipped,
        'n_students': len(students),
        'n_interactions': n_interactions,
        'unique_skills': len(set(
            r['skill_name'] for rows in students.values() for r in rows
        )),
    }

    print(f"\n  Processed ASSISTments 2009-2010:")
    print(f"    Students:     {stats['n_students']}")
    print(f"    Interactions: {stats['n_interactions']}")
    print(f"    Unique skills: {stats['unique_skills']}")
    print(f"    Skipped rows:  {stats['skipped']}")
    print(f"    Output: {output_path}")

    return stats


def main():
    raw_path = RAW_DIR / "skill_builder_data.csv"

    # Step 1: Download if not already present
    if not raw_path.exists() or raw_path.stat().st_size < 10000:
        print("=== Downloading ASSISTments 2009-2010 ===")
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Try corrected version first
        success = try_gdown(CORRECTED_FILE_ID, raw_path)
        if not success:
            success = download_from_gdrive(CORRECTED_FILE_ID, raw_path)
        if not success:
            # Try original version
            success = try_gdown(GDRIVE_FILE_ID, raw_path)
        if not success:
            success = download_from_gdrive(GDRIVE_FILE_ID, raw_path)

        if not success:
            print("\n❌ Automatic download failed.")
            print("Please download manually:")
            print(f"  1. Go to https://drive.google.com/file/d/{CORRECTED_FILE_ID}/view")
            print(f"  2. Click 'Download'")
            print(f"  3. Save as: {raw_path}")
            print(f"\n  OR download from Figshare:")
            print(f"  https://figshare.com/articles/dataset/ASSISTments2009/24743120")
            sys.exit(1)
    else:
        size_mb = raw_path.stat().st_size / (1024 * 1024)
        print(f"=== ASSISTments raw data already exists ({size_mb:.1f} MB) ===")

    # Step 2: Process into our format
    print("\n=== Processing ASSISTments data ===")
    stats = process_raw_csv(
        raw_path, OUTPUT_CSV,
        max_students=500,
        max_rows=100000,
    )

    if stats.get('n_students', 0) > 0:
        print(f"\n✅ Real ASSISTments data ready at {OUTPUT_CSV}")
        print(f"   {stats['n_students']} students, {stats['n_interactions']} interactions")
    else:
        print("\n❌ Processing failed. Check raw file format.")


if __name__ == "__main__":
    main()
