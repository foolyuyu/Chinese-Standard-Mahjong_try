from pathlib import Path
from datetime import datetime
import shutil
import subprocess
import sys


OUTPUT_DIR = 'data'


def main():
    base_dir = Path(__file__).resolve().parent
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

    print('Rebuilding dataset from data.txt ...')
    subprocess.run(
        [sys.executable, 'preprocess.py'],
        cwd = base_dir,
        check = True
    )
    print('Done.')


if __name__ == '__main__':
    main()
