import os
import time
import csv
import re
import pandas as pd
from google import genai
from google.genai import types

GEMINI_KEY  = "AIzaSyDvSlM5GQNtPKXInx_rME23N1Eneiwud2I"
MODEL_ID    = "gemini-3.1-flash-lite-preview"

PATCHES_DIR = 'C:/Users/Dell/Downloads/Independent-Study-Project---Mitotic-vqa/notebooks/patches'
ANNO_CSV    = f'{PATCHES_DIR}/annotations.csv'
OUTPUT_FILE = 'C:/Users/Dell/Downloads/Independent-Study-Project---Mitotic-vqa/notebooks/results_exp4.csv'

client = genai.Client(api_key=GEMINI_KEY)

# PROMPTS 
prompts = {
    'raw': """You are an expert pathology assistant analyzing H&E stained breast cancer tissue at 40x magnification.

A cell of interest is located at the CENTER of this 256x256 pixel image.

MITOTIC FIGURES (answer YES) show:
- Condensed chromosomes in star-shaped, V-shaped, or scattered patterns
- No visible nuclear membrane (it has dissolved)
- Dark irregular chromatin arranged as distinct chromosome arms
- Chaotic, asymmetric appearance of chromosomal material

HARD NEGATIVES (answer NO) include:
- Apoptotic cells: dark shrunken nucleus, round and smooth, nuclear membrane still visible
- Hyperchromatic nuclei: very dark but round/oval shape with intact membrane
- Normal dark nuclei: uniform dark staining without chromosome structure

Is the cell at the center of this image undergoing mitosis?
Answer in this exact format:
Yes/No, X% confidence. One sentence explaining your reasoning.
Example: "Yes, 85% confidence. The cell shows condensed chromosomal material consistent with active mitosis." """,

    'bbox': """You are an expert pathology assistant analyzing H&E stained breast cancer tissue at 40x magnification.

The RED BOUNDING BOX marks the specific cell to evaluate. It is located at the CENTER of this image.

MITOTIC FIGURES (answer YES) show:
- Condensed chromosomes in star-shaped, V-shaped, or scattered patterns
- No visible nuclear membrane (it has dissolved)
- Dark irregular chromatin arranged as distinct chromosome arms
- Chaotic, asymmetric appearance of chromosomal material

HARD NEGATIVES (answer NO) include:
- Apoptotic cells: dark shrunken nucleus, round and smooth, nuclear membrane still visible
- Hyperchromatic nuclei: very dark but round/oval shape with intact membrane
- Normal dark nuclei: uniform dark staining without chromosome structure

Is the cell inside the red bounding box undergoing mitosis?
Answer in this exact format:
Yes/No, X% confidence. One sentence explaining your reasoning.
Example: "Yes, 85% confidence. The cell shows condensed chromosomal material consistent with active mitosis." """,

    'mask': """You are an expert pathology assistant analyzing H&E stained breast cancer tissue at 40x magnification.

The GREEN HIGHLIGHTED REGION marks the nucleus of the specific cell to evaluate. It is located at the CENTER of this image.

MITOTIC FIGURES (answer YES) show:
- Condensed chromosomes in star-shaped, V-shaped, or scattered patterns
- No visible nuclear membrane (it has dissolved)
- Dark irregular chromatin arranged as distinct chromosome arms
- Chaotic, asymmetric appearance of chromosomal material

HARD NEGATIVES (answer NO) include:
- Apoptotic cells: dark shrunken nucleus, round and smooth, nuclear membrane still visible
- Hyperchromatic nuclei: very dark but round/oval shape with intact membrane
- Normal dark nuclei: uniform dark staining without chromosome structure

Is the cell in the green highlighted region undergoing mitosis?
Answer in this exact format:
Yes/No, X% confidence. One sentence explaining your reasoning.
Example: "Yes, 85% confidence. The cell shows condensed chromosomal material consistent with active mitosis." """
}

def parse_response(text):
    text = text.strip()
    yes_no = -1
    if text.lower().startswith('yes'):
        yes_no = 1
    elif text.lower().startswith('no'):
        yes_no = 0
    confidence = None
    match = re.search(r'(\d+)\s*%', text)
    if match:
        confidence = int(match.group(1))
    explanation = ''
    dot_idx = text.find('.')
    if dot_idx != -1 and dot_idx < len(text) - 1:
        explanation = text[dot_idx + 1:].strip()
    return yes_no, confidence, explanation

#  API CALL 
def call_api(image_path, prompt):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt
        ]
    )
    return response.text.strip()

def fix_path(old_path, patches_dir):
    """
    CSV stores paths like:
    C:/Users/Dell/Downloads/Independent-Study-Project---Mitotic-vqa/patches/raw/...
    But files are actually at:
    C:/Users/Dell/Downloads/Independent-Study-Project---Mitotic-vqa/notebooks/patches/raw/...
    Extract condition/label/filename and rebuild correct path.
    """
    # Find the patches/ part and take everything after it
    marker = '/patches/'
    idx = old_path.find(marker)
    if idx == -1:
        return old_path  # already correct or unknown format
    relative = old_path[idx + len(marker):]  # e.g. raw/hard_negative/ann_1.png
    return f'{patches_dir}/{relative}'

def save_results(results):
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'ann_id', 'image_id', 'filename', 'label', 'category_id',
            'condition', 'ground_truth', 'prediction',
            'confidence', 'response', 'explanation'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def clean_and_load():
    if not os.path.exists(OUTPUT_FILE):
        print("No existing results. Starting fresh.")
        return [], set()
    rows = []
    with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    seen = {}
    for row in rows:
        key = (row['ann_id'], row['condition'])
        if key not in seen:
            seen[key] = row
        else:
            if seen[key]['prediction'] == '-1' and row['prediction'] != '-1':
                seen[key] = row
    results = list(seen.values())
    processed_set = {
        (r['ann_id'], r['condition'])
        for r in results
        if str(r['prediction']) != '-1'
    }
    dupes = len(rows) - len(results)
    print(f"Loaded {len(results)} unique results ({len(processed_set)} successful)")
    if dupes > 0:
        print(f"Removed {dupes} duplicate rows automatically")
    print(f"Errors will be retried automatically.")
    return results, processed_set

def verify_paths(anno_df):
    print("=== PATH VERIFICATION ===")
    print(f"PATCHES_DIR : {PATCHES_DIR} — {'EXISTS' if os.path.exists(PATCHES_DIR) else 'MISSING!!!'}")
    print(f"ANNO_CSV    : {ANNO_CSV} — {'EXISTS' if os.path.exists(ANNO_CSV) else 'MISSING!!!'}")
    print(f"OUTPUT_FILE : {OUTPUT_FILE}")
    for condition in ['raw', 'bbox', 'mask']:
        for label in ['mitotic', 'hard_negative']:
            folder = f'{PATCHES_DIR}/{condition}/{label}'
            n = len(os.listdir(folder)) if os.path.exists(folder) else 0
            print(f"  {condition}/{label}: {n} files")

    # Test path fix on first row
    sample = anno_df.iloc[0]
    fixed = fix_path(sample['raw_path'], PATCHES_DIR)
    print(f"\nPath fix test:")
    print(f"  Original : {sample['raw_path']}")
    print(f"  Fixed    : {fixed}")
    print(f"  Exists   : {os.path.exists(fixed)}")
    print()

# MAIN 
def main():
    anno_df = pd.read_csv(ANNO_CSV)
    verify_paths(anno_df)

    print(f"Annotations loaded : {len(anno_df)}")
    print(f"Mitotic            : {len(anno_df[anno_df['label']=='mitotic'])}")
    print(f"Hard negative      : {len(anno_df[anno_df['label']=='hard_negative'])}")

    results, processed_set = clean_and_load()

    # Build full task list with fixed paths
    all_tasks = []
    for _, row in anno_df.iterrows():
        for condition in ['raw', 'bbox', 'mask']:
            path_col   = f'{condition}_path'
            image_path = fix_path(row[path_col], PATCHES_DIR)
            all_tasks.append((row, condition, image_path))

    remaining = [
        (row, cond, path) for row, cond, path in all_tasks
        if (str(int(row['ann_id'])), cond) not in processed_set
    ]

    print(f"\nTotal tasks     : {len(all_tasks)}")
    print(f"Already done    : {len(processed_set)}")
    print(f"Remaining       : {len(remaining)}")
    print(f"Model           : {MODEL_ID}")
    print(f"Output file     : {OUTPUT_FILE}")
    print(f"\nStarting...\n")

    results_lookup = {
        (r['ann_id'], r['condition']): i
        for i, r in enumerate(results)
    }

    processed_count = 0
    total_remaining = len(remaining)
    quota_exceeded  = False

    for row, condition, image_path in remaining:
        if quota_exceeded:
            break

        ann_id       = int(row['ann_id'])
        label        = row['label']
        ground_truth = 1 if label == 'mitotic' else 0
        prompt       = prompts[condition]
        processed_count += 1

        success    = False
        last_error = ""

        for attempt in range(4):
            try:
                response_text      = call_api(image_path, prompt)
                yes_no, conf, expl = parse_response(response_text)

                new_row = {
                    'ann_id'      : ann_id,
                    'image_id'    : row['image_id'],
                    'filename'    : row['filename'],
                    'label'       : label,
                    'category_id' : row['category_id'],
                    'condition'   : condition,
                    'ground_truth': ground_truth,
                    'prediction'  : yes_no,
                    'confidence'  : conf,
                    'response'    : response_text,
                    'explanation' : expl
                }

                key = (str(ann_id), condition)
                if key in results_lookup:
                    results[results_lookup[key]] = new_row
                else:
                    results_lookup[key] = len(results)
                    results.append(new_row)

                print(f"[{processed_count}/{total_remaining}] "
                      f"ann_{ann_id} | {condition} | "
                      f"{label} | pred={yes_no} | conf={conf}%")
                success = True
                break

            except Exception as e:
                last_error = str(e)

                if "RESOURCE_EXHAUSTED" in last_error or "429" in last_error:
                    print(f"Quota exceeded. Saving and stopping.")
                    quota_exceeded = True
                    break
                elif "503" in last_error or "UNAVAILABLE" in last_error:
                    wait = (attempt + 1) * 60
                    print(f"503 error. Waiting {wait}s before retry {attempt+1}/3...")
                    time.sleep(wait)
                elif "getaddrinfo" in last_error or "connection" in last_error.lower():
                    wait = (attempt + 1) * 30
                    print(f"Network error. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Error for ann_{ann_id}/{condition}: {last_error[:100]}")
                    break

        if quota_exceeded:
            break

        if not success:
            error_row = {
                'ann_id'      : ann_id,
                'image_id'    : row['image_id'],
                'filename'    : row['filename'],
                'label'       : label,
                'category_id' : row['category_id'],
                'condition'   : condition,
                'ground_truth': ground_truth,
                'prediction'  : -1,
                'confidence'  : None,
                'response'    : f"Error: {last_error}",
                'explanation' : ''
            }
            key = (str(ann_id), condition)
            if key in results_lookup:
                results[results_lookup[key]] = error_row
            else:
                results_lookup[key] = len(results)
                results.append(error_row)

        if success:
            if processed_count % 15 == 0:
                print(f"--- 15 requests done, pausing 60s ---")
                time.sleep(60)
            else:
                time.sleep(10)

        if processed_count % 10 == 0:
            save_results(results)
            done   = len([r for r in results if str(r['prediction']) != '-1'])
            errors = len(results) - done
            print(f"--- Checkpoint: {done} done, {errors} errors ---")

    save_results(results)
    done   = len([r for r in results if str(r['prediction']) != '-1'])
    errors = len(results) - done

    print(f"\n{'='*50}")
    print(f"Run complete.")
    print(f"Total rows  : {len(results)}")
    print(f"Successful  : {done}")
    print(f"Errors      : {errors}")
    print(f"Saved to    : {OUTPUT_FILE}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()