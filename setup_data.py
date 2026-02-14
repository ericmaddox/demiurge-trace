import requests
import tarfile
import io
import os

def setup_data():
    record_id = "8423265"
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    print(f"Fetching metadata from {api_url}...")
    response = requests.get(api_url)
    data = response.json()
    files = data.get('files', [])
    
    tar_url = None
    for f in files:
        if f['key'].endswith(".tar.gz"):
            tar_url = f['links']['self']
            print(f"Found tarball: {f['key']}")
            break
            
    if not tar_url:
        print("No tarball found!")
        return

    print(f"Downloading tarball from {tar_url}...")
    # Stream download so we don't load into memory if huge
    r = requests.get(tar_url, stream=True)
    
    # Check size
    total_size = int(r.headers.get('content-length', 0))
    print(f"File size: {total_size / 1024 / 1024:.2f} MB")
    
    # We will load it into a BytesIO wrapper if it's small enough, or write to temp file.
    # 15yr data is likely < 500MB.
    # Actually, let's just use tarfile.open with fileobj=r.raw (if it supports seek? probably not)
    # Better to write to a temp file.
    
    temp_tar = "nanograv_data.tar.gz"
    with open(temp_tar, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("Download complete. Extracting J1713+0747 files...")
    
    target_dir = "data/real"
    os.makedirs(target_dir, exist_ok=True)
    
    with tarfile.open(temp_tar, "r:gz") as tar:
        # List all members to find the right path
        members = tar.getmembers()
        print(f"Found {len(members)} files in tar.")
        
        # We need narrow band par and tim
        # Try to find them dynamically
        par_file = None
        tim_file = None
        
        for m in members:
            # Heuristic for the "best" J1713 file
            # usually narrow band is standard for these analyses
            if "J1713+0747" in m.name:
                if m.name.endswith(".par") and "narrow" in m.name.lower():
                     par_file = m
                elif m.name.endswith(".tim") and "narrow" in m.name.lower():
                     tim_file = m
        
        # Fallback if "narrow" not in name (check structure)
        if not par_file:
             for m in members:
                if "J1713+0747" in m.name and m.name.endswith(".par"):
                    par_file = m
                    break
        if not tim_file:
             for m in members:
                if "J1713+0747" in m.name and m.name.endswith(".tim"):
                    tim_file = m
                    break
                    
        if par_file:
            print(f"Extracting {par_file.name}...")
            tar.extract(par_file, path=target_dir)
            # Flatten structure? The extract keeps the folder structure. 
            # We might want to move it.
            extracted_path = os.path.join(target_dir, par_file.name)
            final_path = os.path.join(target_dir, "J1713+0747.par")
            # Create subdirs if extract didn't (tar.extract does)
            
            # Actually tar.extract extracts full path.
            # let's just find where it went.
            # actually better to read and write content to our clean path.
            f_in = tar.extractfile(par_file)
            with open(final_path, 'wb') as f_out:
                f_out.write(f_in.read())
            print(f"Saved to {final_path}")
            
        if tim_file:
            print(f"Extracting {tim_file.name}...")
            f_in = tar.extractfile(tim_file)
            final_path = os.path.join(target_dir, "J1713+0747.tim")
            with open(final_path, 'wb') as f_out:
                f_out.write(f_in.read())
            print(f"Saved to {final_path}")
            
    # Cleanup
    os.remove(temp_tar)
    print("Done.")

if __name__ == "__main__":
    setup_data()
