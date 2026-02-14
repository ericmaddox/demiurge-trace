import requests
import tarfile
import io
import os
import re

ZENODO_RECORD_ID = "8423265"
DATA_DIR = "data/real"
TARGET_PULSARS = [
    "J1713+0747", "J1909-3744", "J0437-4715", "J1614-2230", "J1744-1134",
    "B1937+21", "B1855+09", "J1600-3053", "J2145-0750", "J1857+0943"
]

# Alias map: some pulsars appear in NANOGrav data under a different name
# Key = our target name, Value = alternative prefix used in tarball filenames
PULSAR_ALIASES = {
    "J1857+0943": "B1857+09",
}

def patch_par_file(filepath):
    """Patches PAR files to fix binary model compatibility."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Patch T2 -> DDK
        new_content = re.sub(r"BINARY\s+T2", "BINARY         DDK", content)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"  [PATCHED] {os.path.basename(filepath)}")
    except Exception as e:
        print(f"  [ERROR PATCHING] {filepath}: {e}")

def setup_array():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Fetching metadata for Zenodo record {ZENODO_RECORD_ID}...")
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()

    # Find the main tarball
    tarball_url = None
    for file_info in data['files']:
        if file_info['key'].endswith('.tar.gz') and 'dataset' in file_info['key']:
            tarball_url = file_info['links']['self']
            break
    
    if not tarball_url:
        # Fallback to any large tarball
        for file_info in data['files']:
            if file_info['key'].endswith('.tar.gz'):
                tarball_url = file_info['links']['self']
                break

    if not tarball_url:
        print("Could not find data tarball.")
        return

    print(f"Downloading dataset from {tarball_url}...")
    print("This may take a few minutes (~600MB)...")
    r = requests.get(tarball_url, stream=True)
    r.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        print("Extracting target pulsars...")
        for member in tar.getmembers():
            if not member.isfile():
                continue
                
            path_parts = member.name.split('/')
            filename = path_parts[-1]
            
            # We want files in narrowband/par or wideband/par
            # But mostly we want the .par and .tim files
            if not (filename.endswith(".par") or filename.endswith(".tim")):
                continue

            # Check if this file belongs to one of our target pulsars
            for psr in TARGET_PULSARS:
                # Check both the canonical name and any known alias
                alias = PULSAR_ALIASES.get(psr)
                if filename.startswith(psr) or (alias and filename.startswith(alias)):
                    # We prefer naming them PSR.par and PSR.tim locally for convenience
                    # Always save under the canonical name (psr), not the alias
                    ext = ".par" if filename.endswith(".par") else ".tim"
                    target_name = f"{psr}{ext}"
                    
                    # Avoid overwriting with 'pred' or 'alternate' files if we already have the main one
                    if "pred" in filename or "alternate" in member.name:
                        continue
                        
                    member.name = target_name 
                    tar.extract(member, path=DATA_DIR)
                    matched_via = f" (matched via alias {alias})" if alias and filename.startswith(alias) else ""
                    print(f"  [EXTRACTED] {filename} -> {target_name}{matched_via}")
                    
                    if target_name.endswith(".par"):
                        patch_par_file(os.path.join(DATA_DIR, target_name))
                    break

    print("\nEnsemble data setup complete.")
    print(f"Files saved to: {DATA_DIR}")

if __name__ == "__main__":
    setup_array()
