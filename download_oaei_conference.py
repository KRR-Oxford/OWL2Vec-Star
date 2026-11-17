"""Download the OAEI Conference track ontologies and reference alignments.

  ontologies             -> ./ontologies/                       (16 .owl)
  reference alignments   -> ./reference_alignments/conference/  (21 .rdf)

Usage:
  python download_oaei_conference.py
  python download_oaei_conference.py --year 2023 --ontology_dir ./ontologies
"""
import argparse
import io
import os
import urllib.request
import zipfile

BASE_URL = 'https://oaei.ontologymatching.org/{year}/conference/data/{name}'


def download_and_extract(url: str, target_dir: str, extension: str) -> int:
    """Extract files with the given extension, flattened, into target_dir."""
    print(f'Downloading {url}')
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            # keep only the bare file name so archive paths cannot escape
            name = os.path.basename(member.filename)
            if member.is_dir() or not name.endswith(extension):
                continue
            with open(os.path.join(target_dir, name), 'wb') as f:
                f.write(archive.read(member))
            count += 1
    print(f'  {count} {extension} files -> {target_dir}')
    return count


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--year', default='2023')
    parser.add_argument('--ontology_dir', default='./ontologies')
    parser.add_argument('--alignment_dir',
                        default='./reference_alignments/conference')
    args = parser.parse_args()

    download_and_extract(
        BASE_URL.format(year=args.year, name='conference.zip'),
        args.ontology_dir, '.owl')
    download_and_extract(
        BASE_URL.format(year=args.year, name='reference-alignment.zip'),
        args.alignment_dir, '.rdf')


if __name__ == '__main__':
    main()
