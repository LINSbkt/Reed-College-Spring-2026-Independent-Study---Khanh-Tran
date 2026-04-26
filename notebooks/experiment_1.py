import os
import time
import csv
from google import genai
from google.genai import types

# 1. Setup 
API_KEY = "AIzaSyCy9dtWH624tFx1nZtzEum6f9QAlwl9BRI"
client = genai.Client(api_key=API_KEY)

# 2. Configuration
MODEL_ID = "gemini-3.1-flash-lite-preview"

# Prompts for each folder type
prompts = {
    'raw': "You are a pathology assistant. The image shows a 256x256 pixel crop from an H&E stained breast cancer slide at 40x magnification. Is the cell at the center of this image undergoing mitosis? Answer only yes or no, followed by one sentence explaining your reasoning.",
    'bbox': "You are a pathology assistant. The image shows a pixel crop from an H&E stained breast cancer slide at 40x magnification. The red bounding box marks a specific cell. Is this cell undergoing mitosis? Answer only yes or no, followed by one sentence explaining your reasoning.",
    'mask': "You are a pathology assistant. The image shows a 256x256 pixel crop from an H&E stained breast cancer slide at 40x magnification. The green highlighted region marks the nucleus of a specific cell. Is this cell undergoing mitosis? Answer only yes or no, followed by one sentence explaining your reasoning."
}

def get_mime_type(filename):
    ext = filename.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg']:
        return "image/jpeg"
    elif ext == 'png':
        return "image/png"
    elif ext == 'webp':
        return "image/webp"
    else:
        return "image/jpeg"  # default

def process_images():
    results = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    total_images = 0
    quota_exceeded = False
    
    # Load existing results to resume processing
    processed_set = set()
    if os.path.exists('results.csv'):
        with open('results.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Only skip successful processing; retry errors
                if row['yes_no'] != 'error':
                    key = (row['folder_type'], row['subfolder'], row['filename'])
                    processed_set.add(key)
                results.append(row)
        print(f"Loaded {len(results)} existing results. Resuming processing (retrying errors)...")
    
    for folder_type in ['raw', 'bbox', 'mask']:
        prompt = prompts[folder_type]
        for subfolder in ['hard_negative', 'mitotic']:
            folder_path = os.path.join(folder_type, subfolder)
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist, skipping.")
                continue
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
            for filename in image_files:
                if (folder_type, subfolder, filename) not in processed_set:
                    total_images += 1
    
    print(f"Found {total_images} images to process. Starting...")
    
    processed_count = len(results)
    for folder_type in ['raw', 'bbox', 'mask']:
        if quota_exceeded:
            break
        prompt = prompts[folder_type]
        for subfolder in ['hard_negative', 'mitotic']:
            if quota_exceeded:
                break
            folder_path = os.path.join(folder_type, subfolder)
            if not os.path.exists(folder_path):
                continue
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
            
            for filename in image_files:
                if quota_exceeded:
                    break
                file_path = os.path.join(folder_path, filename)
                
                # Skip if already processed
                if (folder_type, subfolder, filename) in processed_set:
                    continue
                
                processed_count += 1
                
                try:
                    # Load and send the image
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
                    
                    # Parse response: first word is yes/no, rest is explanation
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
                    
                    print(f"[{processed_count}/{total_images}] Processed {folder_type}/{subfolder}/{filename}: {yes_no}")
                    
                    # 3. Rate Limiting Logic (Crucial for Free Tier)
                    # 15 RPM means 1 request every 4 seconds. We use 5 to be safe.
                    if processed_count % 15 == 0:
                        print("--- Reached minute limit, pausing for 60 seconds ---")
                        time.sleep(60)
                    else:
                        time.sleep(5) 
                        
                except Exception as e:
                    print(f"Error processing {folder_type}/{subfolder}/{filename}: {e}")
                    results.append({
                        'folder_type': folder_type,
                        'subfolder': subfolder,
                        'filename': filename,
                        'response': f"Error: {e}",
                        'yes_no': 'error',
                        'explanation': ''
                    })
                    # Check for quota exceeded
                    if "RESOURCE_EXHAUSTED" in str(e) or "quota exceeded" in str(e).lower():
                        quota_exceeded = True
                        print("Daily quota exceeded. Stopping processing to save partial results.")
                        break
                    # If we hit a rate limit error, wait longer
                    elif "rate limit" in str(e).lower():
                        print("Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
    
    # Save results to CSV
    with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['folder_type', 'subfolder', 'filename', 'response', 'yes_no', 'explanation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Processing complete. Results saved to results.csv")
    
    # Save results to CSV
    with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['folder_type', 'subfolder', 'filename', 'response', 'yes_no', 'explanation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Processing complete. Results saved to results.csv")

if __name__ == "__main__":
    process_images()