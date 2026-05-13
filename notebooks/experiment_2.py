import os
import time
import csv
import random
from google import genai
from google.genai import types

# 1. Setup
API_KEY = "AIzaSyBOummRoVqe6H4zx-ZclIE9KGHikDI6Ovk"
client = genai.Client(api_key=API_KEY)

MODEL_ID = "gemini-3.1-flash-lite-preview"

# 2. Improved prompts
prompts = {
    'raw': """You are an expert pathology assistant analyzing H&E stained breast cancer tissue at 40x magnification.

Classify the cell at the CENTER of this 256x256 pixel image.

MITOTIC FIGURES (answer YES) show:
- Condensed chromosomes in star-shaped, V-shaped, or scattered patterns
- No visible nuclear membrane (it has dissolved)
- Dark irregular chromatin arranged as distinct chromosome arms
- Chaotic, asymmetric appearance of chromosomal material

HARD NEGATIVES (answer NO) include:
- Apoptotic cells: dark shrunken nucleus, round and smooth, nuclear membrane still visible
- Hyperchromatic nuclei: very dark but round/oval shape with intact membrane
- Normal dark nuclei: uniform dark staining without chromosome structure

Be conservative. Only answer YES if you clearly see chromosome condensation patterns of active mitosis. If in doubt, answer NO.

Is the cell at the center undergoing mitosis? Answer only yes or no, followed by one sentence explaining your reasoning.""",

    'bbox': """You are an expert pathology assistant analyzing H&E stained breast cancer tissue at 40x magnification.

The RED BOUNDING BOX marks the specific cell to evaluate.

MITOTIC FIGURES (answer YES) show:
- Condensed chromosomes in star-shaped, V-shaped, or scattered patterns
- No visible nuclear membrane (it has dissolved)
- Dark irregular chromatin arranged as distinct chromosome arms
- Chaotic, asymmetric appearance of chromosomal material

HARD NEGATIVES (answer NO) include:
- Apoptotic cells: dark shrunken nucleus, round and smooth, nuclear membrane still visible
- Hyperchromatic nuclei: very dark but round/oval shape with intact membrane
- Normal dark nuclei: uniform dark staining without chromosome structure

Be conservative. Only answer YES if you clearly see chromosome condensation patterns of active mitosis. If in doubt, answer NO.

Is the cell inside the red box undergoing mitosis? Answer only yes or no, followed by one sentence explaining your reasoning.""",

    'mask': """You are an expert pathology assistant analyzing H&E stained breast cancer tissue at 40x magnification.

The GREEN HIGHLIGHTED REGION marks the nucleus of the specific cell to evaluate.

MITOTIC FIGURES (answer YES) show:
- Condensed chromosomes in star-shaped, V-shaped, or scattered patterns
- No visible nuclear membrane (it has dissolved)
- Dark irregular chromatin arranged as distinct chromosome arms
- Chaotic, asymmetric appearance of chromosomal material

HARD NEGATIVES (answer NO) include:
- Apoptotic cells: dark shrunken nucleus, round and smooth, nuclear membrane still visible
- Hyperchromatic nuclei: very dark but round/oval shape with intact membrane
- Normal dark nuclei: uniform dark staining without chromosome structure

Be conservative. Only answer YES if you clearly see chromosome condensation patterns of active mitosis. If in doubt, answer NO.

Is the cell in the green highlighted region undergoing mitosis? Answer only yes or no, followed by one sentence explaining your reasoning."""
}

def get_mime_type(filename):
    ext = filename.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg']:
        return "image/jpeg"
    elif ext == 'png':
        return "image/png"
    else:
        return "image/jpeg"

def sample_files(folder_path, n=50, seed=42):
    """Sample n files from folder with fixed seed for reproducibility."""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    random.seed(seed)
    return random.sample(all_files, min(n, len(all_files)))

def process_images():
    results = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    quota_exceeded = False
    
    # Load existing results to resume
    processed_set = set()
    if os.path.exists('results_improved.csv'):
        with open('results_improved.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['yes_no'] != 'error':
                    key = (row['folder_type'], row['subfolder'], row['filename'])
                    processed_set.add(key)
                results.append(row)
        print(f"Loaded {len(results)} existing results. Resuming...")
    
    # Build sample list 
    sample_plan = {}
    total_images = 0
    for folder_type in ['raw', 'bbox', 'mask']:
        sample_plan[folder_type] = {}
        for subfolder in ['hard_negative', 'mitotic']:
            folder_path = os.path.join(folder_type, subfolder)
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist, skipping.")
                sample_plan[folder_type][subfolder] = []
                continue
            sampled = sample_files(folder_path, n=10000, seed=42)  # Use large n to get all files
            sample_plan[folder_type][subfolder] = sampled
            # Count only unprocessed
            unprocessed = [f for f in sampled if (folder_type, subfolder, f) not in processed_set]
            total_images += len(unprocessed)

    print(f"Processing all available images: {total_images} remaining to process")
    print(f"Already processed: {len(processed_set)}")
    print(f"Remaining to process: {total_images}")
    print("Starting...\n")

    processed_count = 0
    for folder_type in ['raw', 'bbox', 'mask']:
        if quota_exceeded:
            break
        prompt = prompts[folder_type]
        for subfolder in ['hard_negative', 'mitotic']:
            if quota_exceeded:
                break
            folder_path = os.path.join(folder_type, subfolder)
            sampled_files = sample_plan[folder_type][subfolder]

            for filename in sampled_files:
                if quota_exceeded:
                    break
                if (folder_type, subfolder, filename) in processed_set:
                    continue

                file_path = os.path.join(folder_path, filename)
                processed_count += 1

                max_retries = 3
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        with open(file_path, "rb") as f:
                            image_bytes = f.read()
                            

                        mime_type = get_mime_type(filename)

                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=[
                                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                prompt
                            ]
                        )

                        response_text = response.text.strip()
                        parts = response_text.split(' ', 1)
                        yes_no = parts[0].lower() if parts else ''
                        explanation = parts[1] if len(parts) > 1 else ''

                        results.append({
                            'folder_type': folder_type,
                            'subfolder': subfolder,
                            'filename': filename,
                            'response': response_text,
                            'yes_no': yes_no,
                            'explanation': explanation
                        })

                        print(f"[{processed_count}/{total_images}] {folder_type}/{subfolder}/{filename}: {yes_no}")
                        success = True

                    except Exception as e:
                        retry_count += 1
                        error_str = str(e)
                        print(f"Error (attempt {retry_count}/{max_retries}): {folder_type}/{subfolder}/{filename}: {error_str}")

                        if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                            quota_exceeded = True
                            print("Quota exceeded..")
                            break
                        elif "503" in error_str or "UNAVAILABLE" in error_str:
                            if retry_count < max_retries:
                                wait_time = 120 * retry_count  # 120s, 240s
                                print(f"Service unavailable. Retrying in {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                results.append({
                                    'folder_type': folder_type,
                                    'subfolder': subfolder,
                                    'filename': filename,
                                    'response': f"Error after {max_retries} retries: {e}",
                                    'yes_no': 'error',
                                    'explanation': ''
                                })
                        elif "rate" in error_str.lower():
                            if retry_count < max_retries:
                                print("Rate limit. Retrying")
                                time.sleep(60)
                            else:
                                results.append({
                                    'folder_type': folder_type,
                                    'subfolder': subfolder,
                                    'filename': filename,
                                    'response': f"Error after {max_retries} retries: {e}",
                                    'yes_no': 'error',
                                    'explanation': ''
                                })
                        else:
                            # Other errors, don't retry
                            results.append({
                                'folder_type': folder_type,
                                'subfolder': subfolder,
                                'filename': filename,
                                'response': f"Error: {e}",
                                'yes_no': 'error',
                                'explanation': ''
                            })
                            break

                # Rate limiting between successful requests
                if success:
                    if processed_count % 15 == 0:
                        time.sleep(120)
                    else:
                        time.sleep(10)

                # Save every 10 predictions
                if processed_count % 10 == 0:
                    with open('results_improved.csv', 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = ['folder_type', 'subfolder', 'filename', 'response', 'yes_no', 'explanation']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
                    print(f"--- Checkpoint saved ({processed_count} done) ---")

    # Final save
    with open('results_improved.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['folder_type', 'subfolder', 'filename', 'response', 'yes_no', 'explanation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Total predictions: {len(results)}")

if __name__ == "__main__":
    process_images()