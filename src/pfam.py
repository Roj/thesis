import requests
from progressbar import ProgressBar
from pathlib import Path

PFAM_URL="https://www.ebi.ac.uk/pdbe/api/mappings/pfam/{}"
files = list(Path("data/").glob("*.pdb"))
failed = []
for file in ProgressBar()(files, max_value=len(files)):
    try:
        out_filename = Path("if/pfam/") / f"{file.stem}.json"
        if not out_filename.exists():
            response = requests.get(PFAM_URL.format(file.stem))
            response.raise_for_status()

            with open(out_filename, "w") as out:
                out.write(response.text)

    except requests.HTTPError:
        failed.append(file)

print(f"Total failed: {len(failed)}, list: \n\t{failed}")