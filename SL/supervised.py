from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MahjongGBDataset, load_manifest, load_shard_arrays
from model import CNNModel


def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available()


def _resolve_device(requested):
    requested = str(requested).strip().lower()
    if requested == 'auto':
        if _npu_available():
            requested = 'npu'
        elif torch.cuda.is_available():
            requested = 'cuda'
        else:
            requested = 'cpu'
    elif requested.startswith('npu') and not _npu_available():
        print('Warning: NPU is not available, fallback to CPU.')
        requested = 'cpu'
    elif requested.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA is not available, fallback to CPU.')
        requested = 'cpu'
    return torch.device(requested)


def _maybe_set_npu(device):
    if device.type == 'npu':
        import torch_npu  # noqa: F401
        if hasattr(torch, 'npu') and hasattr(torch.npu, 'set_device'):
            torch.npu.set_device(0)


def _resolve_metrics_path(args, base_dir):
    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
    else:
        logdir = Path(args.logdir) if args.logdir else base_dir / 'model'
        metrics_path = logdir / 'metrics.csv'
    metrics_path.parent.mkdir(parents = True, exist_ok = True)
    if metrics_path.exists():
        metrics_path.unlink()
    return metrics_path


def _append_metrics_row(metrics_path, row):
    fieldnames = ['epoch', 'mode', 'train_loss', 'validate_acc', 'train_samples', 'validate_samples']
    write_header = not metrics_path.exists()
    with metrics_path.open('a', newline = '', encoding = 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _to_device(batch, device):
    obs = torch.from_numpy(batch['obs']).to(device)
    glob = torch.from_numpy(batch['glob']).to(device)
    mask = torch.from_numpy(batch['mask']).to(device)
    act = torch.from_numpy(batch['act']).long().to(device)
    return obs, glob, mask, act


def _train_on_indices(model, optimizer, arrays, indices, device, batch_size, epoch, shard_name):
    model.train(True)
    total_loss = 0.0
    total_count = 0
    if len(indices) == 0:
        return total_loss, total_count
    for step in range(0, len(indices), batch_size):
        batch_idx = indices[step: step + batch_size]
        batch = {
            'obs': arrays['obs'][batch_idx],
            'glob': arrays['glob'][batch_idx],
            'mask': arrays['mask'][batch_idx],
            'act': arrays['act'][batch_idx],
        }
        obs, glob, mask, act = _to_device(batch, device)
        input_dict = {'is_training': True, 'obs': {'observation': obs, 'global': glob, 'action_mask': mask}}
        logits = model(input_dict)
        loss = F.cross_entropy(logits, act)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_idx)
        total_count += len(batch_idx)
        if step % max(batch_size * 8, batch_size) == 0:
            print('  train', shard_name, '%d/%d' % (step, len(indices)), 'loss', loss.item())
    return total_loss, total_count


@torch.no_grad()
def _eval_on_indices(model, arrays, indices, device, batch_size, shard_name):
    model.train(False)
    total_correct = 0
    total_count = 0
    for step in range(0, len(indices), batch_size):
        batch_idx = indices[step: step + batch_size]
        batch = {
            'obs': arrays['obs'][batch_idx],
            'glob': arrays['glob'][batch_idx],
            'mask': arrays['mask'][batch_idx],
            'act': arrays['act'][batch_idx],
        }
        obs, glob, mask, act = _to_device(batch, device)
        input_dict = {'is_training': False, 'obs': {'observation': obs, 'global': glob, 'action_mask': mask}}
        logits = model(input_dict)
        pred = logits.argmax(dim = 1)
        total_correct += torch.eq(pred, act).sum().item()
        total_count += len(batch_idx)
    if total_count:
        print('  val', shard_name, 'acc', total_correct / total_count)
    return total_correct, total_count


def _run_sharded_training(args, manifest, device, base_dir):
    _maybe_set_npu(device)
    data_dir = args.data_dir
    logdir = Path(args.logdir) if args.logdir else base_dir / 'model'
    (logdir / 'checkpoint').mkdir(parents = True, exist_ok = True)
    metrics_path = _resolve_metrics_path(args, base_dir)

    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    shards = list(manifest.get('shards', []))
    if not shards:
        raise RuntimeError('Shard manifest is empty.')

    for e in range(args.epochs):
        print('Epoch', e)
        torch.save(model.state_dict(), logdir / 'checkpoint' / ('%d.pkl' % e))

        rng = np.random.default_rng(seed = e)
        shard_order = list(range(len(shards)))
        rng.shuffle(shard_order)

        train_loss_sum = 0.0
        train_count = 0
        val_correct = 0
        val_count = 0

        for shard_idx in shard_order:
            shard = shards[shard_idx]
            arrays = load_shard_arrays(data_dir, shard['file'])
            n = len(arrays['act'])
            if n == 0:
                continue

            indices = np.arange(n)
            rng.shuffle(indices)
            split_point = int(n * args.split_ratio)
            split_point = min(max(split_point, 1), n) if n > 1 else n
            train_idx = indices[:split_point]
            val_idx = indices[split_point:]

            if len(train_idx):
                loss_sum, count = _train_on_indices(model, optimizer, arrays, train_idx, device, args.batch_size, e, shard['file'])
                train_loss_sum += loss_sum
                train_count += count

            if len(val_idx):
                correct, count = _eval_on_indices(model, arrays, val_idx, device, args.batch_size, shard['file'])
                val_correct += correct
                val_count += count

            del arrays

        avg_loss = train_loss_sum / train_count if train_count else 0.0
        avg_acc = val_correct / val_count if val_count else 0.0
        print('Epoch', e + 1, 'train_loss:', avg_loss, 'validate_acc:', avg_acc)
        _append_metrics_row(metrics_path, {
            'epoch': e + 1,
            'mode': 'sharded',
            'train_loss': avg_loss,
            'validate_acc': avg_acc,
            'train_samples': train_count,
            'validate_samples': val_count,
        })


def _run_legacy_training(args, manifest, device, base_dir):
    # Legacy one-match-per-file mode kept for local fallback.
    logdir = Path(args.logdir) if args.logdir else base_dir / 'model'
    (logdir / 'checkpoint').mkdir(parents = True, exist_ok = True)
    metrics_path = _resolve_metrics_path(args, base_dir)

    trainDataset = MahjongGBDataset(0, args.split_ratio, True, data_dir = args.data_dir)
    validateDataset = MahjongGBDataset(args.split_ratio, 1, False, data_dir = args.data_dir)
    loader = DataLoader(
        dataset = trainDataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        pin_memory = device.type == 'cuda'
    )
    vloader = DataLoader(
        dataset = validateDataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = device.type == 'cuda'
    )

    print('Using device:', device)
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for e in range(args.epochs):
        print('Epoch', e)
        torch.save(model.state_dict(), logdir / 'checkpoint' / ('%d.pkl' % e))
        train_loss_sum = 0.0
        train_count = 0
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].to(device), 'global': d[1].to(device), 'action_mask': d[2].to(device)}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[3].long().to(device))
            if i % 128 == 0:
                print('Iteration %d/%d' % (i, len(trainDataset) // args.batch_size + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count = len(d[3])
            train_loss_sum += loss.item() * batch_count
            train_count += batch_count
        print('Run validation:')
        correct = 0
        val_count = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].to(device), 'global': d[1].to(device), 'action_mask': d[2].to(device)}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[3].to(device)).sum().item()
                val_count += len(d[3])
        acc = correct / val_count if val_count else 0.0
        avg_loss = train_loss_sum / train_count if train_count else 0.0
        print('Epoch', e + 1, 'Validate acc:', acc)
        _append_metrics_row(metrics_path, {
            'epoch': e + 1,
            'mode': 'legacy',
            'train_loss': avg_loss,
            'validate_acc': acc,
            'train_samples': train_count,
            'validate_samples': val_count,
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Supervised training for MahjongGB')
    # Toggle here to switch default training data source.
    # USE_OBS_IO = False
    USE_OBS_IO = True
    # Change this to your actual OBS bucket name.
    BUCKET_NAME = 'mahjong-data'
    OBS_DATA_PREFIX = f'obs://{BUCKET_NAME}/SL/data'
    # DATA_DIR = 'data'
    # DATA_DIR = OBS_DATA_PREFIX
    DATA_DIR = OBS_DATA_PREFIX if USE_OBS_IO else 'data'
    parser.add_argument('--data-dir', type = str, default = DATA_DIR, help = 'Directory containing data.txt, count.json, and *.npz')
    parser.add_argument('--logdir', type = str, default = None, help = 'Directory for checkpoints and logs')
    parser.add_argument('--split-ratio', type = float, default = 0.9)
    parser.add_argument('--batch-size', type = int, default = 1024)
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--device', type = str, default = 'auto', choices = ['auto', 'cpu', 'cuda', 'npu'])
    parser.add_argument('--num-workers', type = int, default = 0)
    parser.add_argument('--metrics-file', type = str, default = None, help = 'CSV file for per-epoch metrics')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    device = _resolve_device(args.device)
    print('Using device:', device)

    manifest = load_manifest(args.data_dir)
    if isinstance(manifest, dict) and manifest.get('schema') == 'sharded-v1':
        _run_sharded_training(args, manifest, device, base_dir)
    else:
        _run_legacy_training(args, manifest, device, base_dir)
