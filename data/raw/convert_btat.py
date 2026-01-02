import os
import json
import pandas as pd
import time

# Use raw string (r"...") or forward slashes to avoid SyntaxWarning
REPO_ROOT = r".\BTAT" 

TARGET_FOLDERS = [
    "Normal",
    "DoS", 
    "Delegatecall", 
    "FoT", 
    "OaU", 
    "Reentrancy", 
    "FDV"
]

def load_btat_optimized(root_dir):
    all_data = []
    total_files = 0
    start_time = time.time()

    print(f"Scanning directory: {os.path.abspath(root_dir)}")

    for target in TARGET_FOLDERS:
        start_path = os.path.join(root_dir, target)
        
        if not os.path.exists(start_path):
            print(f"Skipping {target} (Not found)")
            continue

        for current_root, dirs, files in os.walk(start_path):
            # Generate Flattened Label (e.g., Reentrancy_CR)
            relative_path = os.path.relpath(current_root, root_dir)
            label = relative_path.replace(os.sep, '_')
            
            print(f"--> Processing Folder: {label} | Found {len(files)} files...")

            for i, filename in enumerate(files):
                # Simple progress indicator every 100 files
                if i > 0 and i % 100 == 0:
                    print(f"    Processed {i} files in {label}...", end='\r')

                if filename.startswith('.') or filename.lower().endswith('.md'):
                    continue
                
                file_path = os.path.join(current_root, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # OPTIMIZATION: Try to load directly without f.read()
                        # This saves RAM if it's a standard JSON file.
                        try:
                            data = json.load(f)
                            # Standardize to list
                            items = data if isinstance(data, list) else [data]
                            
                            for item in items:
                                if isinstance(item, dict):
                                    item['label'] = label
                                    all_data.append(item)
                                    
                        except json.JSONDecodeError:
                            # Fallback: Reset file pointer and try Line-by-Line (NDJSON)
                            # This handles files that have one JSON object per line
                            f.seek(0) 
                            for line in f:
                                line = line.strip()
                                if not line: continue
                                try:
                                    item = json.loads(line)
                                    if isinstance(item, dict):
                                        item['label'] = label
                                        all_data.append(item)
                                except:
                                    pass # Skip broken lines
                                    
                    total_files += 1

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

            print(f"    Finished folder: {label}                    ")

    print("="*40)
    print(f"Done! Processed {total_files} files in {time.time() - start_time:.2f}s")
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    # Ensure this path matches your folder structure
    # Based on your ls command, the script is running in 'raw', and BTAT is in 'raw/BTAT'
    # So using current directory "." might be safer if you cd into it, 
    # but based on your traceback, this explicit path is best:
    df = load_btat_optimized(REPO_ROOT)
    
    if not df.empty:
        output_csv = "BTAT_dataset_final.csv"
        print(f"Saving {len(df)} rows to {output_csv}...")
        df.to_csv(output_csv, index=False)
        print("Success.")
    else:
        print("No data extracted. Please check the path.")