import requests
from tqdm import tqdm
import argparse
from pathlib import Path
import os

def download_zenodo_file(record_id, filename, destination, access_token=None):
    """
    Downloads a specific file from a Zenodo record with a progress bar.

    Parameters:
    - record_id (str): The Zenodo record ID.
    - filename (str): The exact name of the file to download.
    - destination (str): The local path where the file will be saved.
    - access_token (str, optional): Your Zenodo access token for restricted records.

    Returns:
    - None
    """
    base_url = f"https://zenodo.org/api/records/{record_id}"
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

    # Fetch the record metadata
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    record = response.json()

    # Find the file in the record's files
    file_info = next((f for f in record["files"] if f["key"] == filename), None)
    if not file_info:
        raise FileNotFoundError(f"File '{filename}' not found in record '{record_id}'.")

    # Download the file with a progress bar
    download_url = file_info["links"]["self"]
    with requests.get(download_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 1024 * 1024
        with open(destination, "wb") as f, tqdm(
            total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress_bar.update(len(chunk))

    print(f"\nFile '{filename}' has been downloaded to '{destination}'.")
    
import zipfile

def unzip_file(zip_path, extract_to):
    """
    Unzips a .zip file to the specified directory.

    :param zip_path: Path to the .zip file.
    :param extract_to: Directory where files should be extracted.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")
    

def add_to_pythonpath():
    shell_rc_file = Path.home() / (".zshrc" if os.environ.get("SHELL", "").endswith("zsh") else ".bashrc")
    line_to_add = 'export PYTHONPATH="$PYTHONPATH:$HOME/src"'

    # Check if the line already exists
    if not shell_rc_file.exists() or line_to_add not in shell_rc_file.read_text():
        with open(shell_rc_file, "a") as rc_file:
            rc_file.write(f"\n{line_to_add}\n")
        print(f"✅ Added {line_to_add} to {shell_rc_file}")

    # Source the updated file
    os.system(f"source {shell_rc_file}")
    print(f"✅ Sourced {shell_rc_file}")

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o','--out_path', type=str)
    
    args = parser.parse_args()
    
    out_path = Path(args.out_path).resolve()
    
    record_id = "15040813"  # Replace with your Zenodo record ID
    filename = 'data.zip'  # Replace with the exact filename you wish to download
    
    destination = out_path / filename  # Replace with your desired save path
    download_zenodo_file(record_id, filename, destination)
    
    unzip_file(destination, out_path)
    
    print(out_path.joinpath('data/config/last_checkpoint'))
    print(f'{out_path}/data/config/checkpoint.pth')
    
    with open(out_path.joinpath('data/config/last_checkpoint'), "w") as file:
        file.write(f'{out_path}/data/config/checkpoint.pth')
        
    add_to_pythonpath()
    
        
    
    