from pathlib import Path
import argparse
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def _read_metrics(metrics_file):
    rows = []
    with Path(metrics_file).open('r', newline = '', encoding = 'utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(float(row.get('epoch', 0)))
            except ValueError:
                continue
            try:
                train_loss = float(row.get('train_loss', 'nan'))
            except ValueError:
                train_loss = float('nan')
            try:
                validate_acc = float(row.get('validate_acc', 'nan'))
            except ValueError:
                validate_acc = float('nan')
            rows.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'validate_acc': validate_acc,
                'mode': row.get('mode', ''),
            })
    rows.sort(key = lambda r: r['epoch'])
    return rows


def _plot(rows, output_file, title):
    if not rows:
        raise RuntimeError('No metrics rows found.')

    epochs = [row['epoch'] for row in rows]
    train_loss = [row['train_loss'] for row in rows]
    validate_acc = [row['validate_acc'] for row in rows]

    fig, ax1 = plt.subplots(figsize = (10.5, 5.8))
    ax1.plot(epochs, train_loss, color = '#1f77b4', marker = 'o', linewidth = 2.2, label = 'train_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color = '#1f77b4')
    ax1.tick_params(axis = 'y', labelcolor = '#1f77b4')
    ax1.grid(True, alpha = 0.25, linestyle = '--')

    ax2 = ax1.twinx()
    ax2.plot(epochs, validate_acc, color = '#d62728', marker = 's', linewidth = 2.2, label = 'validate_acc')
    ax2.set_ylabel('Validation Acc', color = '#d62728')
    ax2.tick_params(axis = 'y', labelcolor = '#d62728')
    ax2.set_ylim(0.0, 1.0)

    mode_set = sorted({row['mode'] for row in rows if row['mode']})
    subtitle = ' | '.join(mode_set) if mode_set else 'metrics'
    fig.suptitle(title or f'Training Curves ({subtitle})')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc = 'best')

    fig.tight_layout()
    fig.savefig(output_file, dpi = 180, bbox_inches = 'tight')


def main():
    parser = argparse.ArgumentParser(description = 'Plot SL training metrics')
    parser.add_argument('--input', type = str, default = 'model/metrics.csv', help = 'Path to metrics.csv')
    parser.add_argument('--output', type = str, default = 'model/metrics.png', help = 'Path to output image')
    parser.add_argument('--title', type = str, default = None, help = 'Optional plot title')
    args = parser.parse_args()

    rows = _read_metrics(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents = True, exist_ok = True)
    _plot(rows, output_file, args.title)
    print('Saved plot to %s' % output_file)


if __name__ == '__main__':
    main()
