from pathlib import Path
from datetime import datetime
import shutil
import subprocess
import sys


# Toggle here to switch between local and OBS modes.
# USE_OBS_IO = False
USE_OBS_IO = True

# Change this to your actual OBS bucket name.
BUCKET_NAME = 'mahjong-data'
OBS_DATA_PREFIX = f'obs://{BUCKET_NAME}/SL/data'

# Local mode:
# INPUT_FILE = 'data/data.txt'
# OUTPUT_DIR = 'data'

# OBS mode:
# INPUT_FILE = f'{OBS_DATA_PREFIX}/data.txt'
# OUTPUT_DIR = OBS_DATA_PREFIX

INPUT_FILE = f'{OBS_DATA_PREFIX}/data.txt' if USE_OBS_IO else 'data/data.txt'
OUTPUT_DIR = OBS_DATA_PREFIX if USE_OBS_IO else 'data'


def main():
    base_dir = Path(__file__).resolve().parent
    if not str(OUTPUT_DIR).startswith('obs://'):
        data_dir = base_dir / OUTPUT_DIR
        data_dir.mkdir(exist_ok = True, parents = True)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = base_dir / f'data_backup_{stamp}'
        backup_dir.mkdir(exist_ok = True)

        moved_any = False
        for pattern in ('*.npz', 'count.json'):
            for path in data_dir.glob(pattern):
                shutil.move(str(path), str(backup_dir / path.name))
                moved_any = True

        print('Backup directory:', backup_dir.name)
        if moved_any:
            print('Old generated data moved to backup.')
        else:
            print('No generated data found to back up.')
    else:
        print('OBS output selected, skipping local backup.')

    print('Rebuilding dataset from data.txt ...')
    subprocess.run(
        [sys.executable, 'preprocess.py', '--input-file', INPUT_FILE, '--output-dir', OUTPUT_DIR],
        cwd = base_dir,
        check = True
    )
    print('Done.')


if __name__ == '__main__':
    main()
