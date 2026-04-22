import requests
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path

def resolve_latest_archive(record_id, access_token=None):
    """Return (latest_version_id, single_archive_file_entry) for a Zenodo concept record."""
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

    r = requests.get(
        f"https://zenodo.org/api/records/{record_id}/versions",
        headers=headers,
    )
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    published = [h for h in hits if h.get("status") == "published" and h.get("files")]
    if not published:
        raise RuntimeError(f"No published versions with files found under record {record_id}")

    latest = max(published, key=lambda h: h.get("created", ""))
    files = latest["files"]
    if len(files) != 1:
        names = [f["key"] for f in files]
        raise RuntimeError(
            f"Expected a single archive file in latest version, found {len(files)}: {names}"
        )
    return latest["id"], files[0]


def download_to(file_entry, destination, access_token=None):
    """Download a Zenodo file entry to a local path with a progress bar."""
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    name = file_entry["key"]
    url = file_entry["links"]["self"]

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(destination, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"\nFile '{name}' has been downloaded to '{destination}'.")
    
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
    parser.add_argument('-o', '--out_path', type=str, required=True)
    args = parser.parse_args()

    out_path = Path(args.out_path).resolve()
    record_id = "15040813"  # any version id in the concept — we follow to latest

    version_id, file_entry = resolve_latest_archive(record_id)
    print(f"Latest Zenodo version: {version_id}  (file: {file_entry['key']})")

    destination = out_path / file_entry['key']
    download_to(file_entry, destination)

    unzip_file(destination, out_path)
    flatten_into(out_path)
    destination.unlink(missing_ok=True)
    report_missing(out_path)
