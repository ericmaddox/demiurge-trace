import requests
import json

def get_urls():
    record_id = "8423265"
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url)
    data = response.json()
    
    files = data.get('files', [])
    
    # We want J1713+0747 par and tim files.
    # From NANOGrav releases, they usually have a narrow band and wide band version.
    # Let's look for "J1713+0747_NANOGrav_15yv1.gls.par" or similar.
    # Or just print all matches.
    
    print(f"Total files found: {len(files)}")
    for i, f in enumerate(files):
        if i < 20: # Start with first 20
            print(f"File: {f['key']}")
        
        fname = f['key']
        if "J1713+0747" in fname:
            print(f"--> MATCH FOUND: {fname}")
            print(f"    URL: {f['links']['self']}")

if __name__ == "__main__":
    get_urls()
