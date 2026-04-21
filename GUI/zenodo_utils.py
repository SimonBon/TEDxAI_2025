import requests
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path

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
    

EXPECTED_ARTIFACTS = ['model_new.pth', 'small_data.h5', 'CP_TU_MORE']


def flatten_into(out_path):
    """The Zenodo archive extracts to out_path/data/*. Move those into out_path/ directly."""
    data_dir = out_path / 'data'
    if not data_dir.is_dir():
        return

    for item in data_dir.iterdir():
        if item.name.startswith('.'):
            continue
        target = out_path / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))

    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(out_path / '__MACOSX', ignore_errors=True)


def report_missing(out_path):
    missing = [name for name in EXPECTED_ARTIFACTS if not (out_path / name).exists()]
    if missing:
        print(f"\n⚠️  Expected files missing after extraction: {missing}")
    if not (out_path / 'real_images').is_dir():
        print("⚠️  'real_images/' is not in the Zenodo bundle — the 'Load Image' combo box will be empty until TIFFs are placed there manually.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o','--out_path', type=str)

    args = parser.parse_args()

    out_path = Path(args.out_path).resolve()

    record_id = "15040813"  # Zenodo record ID
    filename = 'data.zip'

    destination = out_path / filename
    download_zenodo_file(record_id, filename, destination)

    unzip_file(destination, out_path)
    flatten_into(out_path)
    destination.unlink(missing_ok=True)
    report_missing(out_path)

    