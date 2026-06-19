from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F

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
    if metrics_path.exists() and not args.append_metrics:
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


def _cpu_state_dict(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _save_cpu_checkpoint(model, path):
    torch.save(_cpu_state_dict(model), path)


def _save_best_checkpoint(model, logdir):
    _save_cpu_checkpoint(model, logdir / 'best.pkl')


def _load_resume_checkpoint(model, resume_path, device):
    if not resume_path:
        return
    state_dict = torch.load(resume_path, map_location = device)
    model.load_state_dict(state_dict)
    print('Loaded resume checkpoint:', resume_path)


def _to_device(batch, device):
    non_blocking = device.type in ['cuda', 'npu']
    obs = torch.from_numpy(batch['obs']).to(device, non_blocking = non_blocking)
    glob = torch.from_numpy(batch['glob']).to(device, non_blocking = non_blocking)
    mask = torch.from_numpy(batch['mask']).to(device, non_blocking = non_blocking)
    act = torch.from_numpy(batch['act']).long().to(device, non_blocking = non_blocking)
    return obs, glob, mask, act


def _train_batch(model, optimizer, batch, device, grad_clip):
    obs, glob, mask, act = _to_device(batch, device)
    input_dict = {'is_training': True, 'obs': {'observation': obs, 'global': glob, 'action_mask': mask}}
    logits = model(input_dict)
    loss = F.cross_entropy(logits, act)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item(), len(batch['act'])


def _resolve_data_dir(data_dir, base_dir):
    path = Path(data_dir)
    return path if path.is_absolute() else base_dir / path


def _load_manifest(data_dir):
    with open(data_dir / 'count.json', encoding = 'utf-8') as f:
        manifest = json.load(f)
    if isinstance(manifest, dict):
        chunks = manifest.get('chunks', [])
        return {
            'schema': manifest.get('schema', 'local-chunk-v1'),
            'total_samples': int(manifest.get('total_samples', sum(int(x['samples']) for x in chunks))),
            'chunks': chunks,
        }
    if isinstance(manifest, list):
        chunks = [{'file': '%d.npz' % i, 'samples': int(samples)} for i, samples in enumerate(manifest)]
        return {
            'schema': 'legacy-match-v1',
            'total_samples': int(sum(manifest)),
            'chunks': chunks,
        }
    raise ValueError('Unsupported count.json format in %s' % data_dir)


def _pad_obs_array(obs):
    channels = CNNModel.OBS_CHANNELS
    if obs.shape[1] == channels:
        return obs
    if obs.shape[1] > channels:
        return obs[:, :channels]
    padded = np.zeros((obs.shape[0], channels, obs.shape[2], obs.shape[3]), dtype = obs.dtype)
    padded[:, :obs.shape[1]] = obs
    return padded


def _load_chunk(data_dir, chunk):
    with np.load(data_dir / chunk['file']) as d:
        obs = _pad_obs_array(d['obs'])
        if 'glob' in d:
            glob = d['glob']
        else:
            glob = np.zeros((obs.shape[0], CNNModel.GLOBAL_SIZE), dtype = np.float32)
        return {
            'obs': obs,
            'glob': glob.astype(np.float32, copy = False),
            'mask': d['mask'],
            'act': d['act'],
        }


def _iter_loaded_chunks(data_dir, chunks, prefetch_chunks):
    if prefetch_chunks <= 0:
        for chunk in chunks:
            yield chunk, _load_chunk(data_dir, chunk)
        return

    with ThreadPoolExecutor(max_workers = prefetch_chunks) as executor:
        chunk_iter = iter(chunks)
        pending = []
        for _ in range(prefetch_chunks):
            try:
                chunk = next(chunk_iter)
            except StopIteration:
                break
            pending.append((chunk, executor.submit(_load_chunk, data_dir, chunk)))
        while pending:
            chunk, future = pending.pop(0)
            try:
                next_chunk = next(chunk_iter)
                pending.append((next_chunk, executor.submit(_load_chunk, data_dir, next_chunk)))
            except StopIteration:
                pass
            yield chunk, future.result()


def _count_batches(chunks, batch_size):
    return sum((int(chunk['samples']) + batch_size - 1) // batch_size for chunk in chunks)


def _take_indexed_batch(payload, indices):
    return {key: value[indices] for key, value in payload.items()}


def _take_sliced_batch(payload, start, end):
    return {key: value[start:end] for key, value in payload.items()}


def _iter_train_batches(payload, batch_size, rng):
    sample_count = len(payload['act'])
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    for start in range(0, sample_count, batch_size):
        yield _take_indexed_batch(payload, indices[start : start + batch_size])


def _iter_eval_batches(payload, batch_size):
    sample_count = len(payload['act'])
    for start in range(0, sample_count, batch_size):
        yield _take_sliced_batch(payload, start, start + batch_size)


def _run_local_training(args, device, base_dir):
    _maybe_set_npu(device)
    logdir = Path(args.logdir) if args.logdir else base_dir / 'model'
    (logdir / 'checkpoint').mkdir(parents = True, exist_ok = True)
    metrics_path = _resolve_metrics_path(args, base_dir)

    train_data_dir = _resolve_data_dir(args.train_data_dir, base_dir)
    valid_data_dir = _resolve_data_dir(args.valid_data_dir, base_dir)
    train_manifest = _load_manifest(train_data_dir)
    valid_manifest = _load_manifest(valid_data_dir)
    train_chunks = list(train_manifest['chunks'])
    valid_chunks = list(valid_manifest['chunks'])
    train_batches = _count_batches(train_chunks, args.batch_size)
    rng = np.random.default_rng(seed = args.seed)

    print('Using device:', device)
    print('Train samples:', train_manifest['total_samples'], 'chunks:', len(train_chunks), 'batches:', train_batches)
    print('Valid samples:', valid_manifest['total_samples'], 'chunks:', len(valid_chunks))
    if train_manifest['schema'] == 'legacy-match-v1':
        print('Warning: training data uses legacy per-match npz files. Re-run preprocess.py for faster chunked loading.')
    model = CNNModel().to(device)
    _load_resume_checkpoint(model, args.resume, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    best_acc = float('-inf')
    best_epoch = None
    epochs_without_improvement = 0

    for e in range(args.start_epoch, args.start_epoch + args.epochs):
        print('Epoch', e)
        _save_cpu_checkpoint(model, logdir / 'checkpoint' / ('%d.pkl' % e))
        train_loss_sum = 0.0
        train_count = 0
        train_order = train_chunks[:]
        rng.shuffle(train_order)
        iteration = 0
        for _, chunk_payload in _iter_loaded_chunks(train_data_dir, train_order, args.prefetch_chunks):
            for d in _iter_train_batches(chunk_payload, args.batch_size, rng):
                loss_value, batch_count = _train_batch(model, optimizer, d, device, args.grad_clip)
                if iteration % 128 == 0:
                    print('Iteration %d/%d' % (iteration, train_batches), 'policy_loss', loss_value)
                train_loss_sum += loss_value * batch_count
                train_count += batch_count
                iteration += 1
            del chunk_payload
        print('Run validation:')
        correct = 0
        val_count = 0
        with torch.no_grad():
            for _, chunk_payload in _iter_loaded_chunks(valid_data_dir, valid_chunks, args.prefetch_chunks):
                for d in _iter_eval_batches(chunk_payload, args.batch_size):
                    obs, glob, mask, act = _to_device(d, device)
                    input_dict = {'is_training': False, 'obs': {'observation': obs, 'global': glob, 'action_mask': mask}}
                    logits = model(input_dict)
                    pred = logits.argmax(dim = 1)
                    correct += torch.eq(pred, act).sum().item()
                    val_count += len(act)
                del chunk_payload
        acc = correct / val_count if val_count else 0.0
        avg_loss = train_loss_sum / train_count if train_count else 0.0
        print('Epoch', e + 1, 'Validate acc:', acc)
        _append_metrics_row(metrics_path, {
            'epoch': e + 1,
            'mode': 'local',
            'train_loss': avg_loss,
            'validate_acc': acc,
            'train_samples': train_count,
            'validate_samples': val_count,
        })

        if acc > best_acc:
            best_acc = acc
            best_epoch = e + 1
            epochs_without_improvement = 0
            _save_best_checkpoint(model, logdir)
            print('New best validation accuracy:', best_acc, 'at epoch', best_epoch)
        else:
            epochs_without_improvement += 1
            print('No improvement for', epochs_without_improvement, 'epoch(s). Best epoch:', best_epoch, 'best acc:', best_acc)

        if args.patience >= 0 and epochs_without_improvement >= args.patience:
            print('Early stopping triggered at epoch', e + 1, 'best epoch:', best_epoch, 'best acc:', best_acc)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Supervised training for MahjongGB')
    parser.add_argument('--train-data-dir', type = str, default = 'data/train', help = 'Local training directory containing count.json and chunk_*.npz files')
    parser.add_argument('--valid-data-dir', type = str, default = 'data/valid', help = 'Local validation directory containing count.json and chunk_*.npz files')
    parser.add_argument('--logdir', type = str, default = None, help = 'Directory for checkpoints and logs')
    parser.add_argument('--batch-size', type = int, default = 1024)
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--weight-decay', type = float, default = 1e-4)
    parser.add_argument('--grad-clip', type = float, default = 1.0)
    parser.add_argument('--device', type = str, default = 'auto', choices = ['auto', 'cpu', 'cuda', 'npu'])
    parser.add_argument('--prefetch-chunks', type = int, default = 1, help = 'Number of local npz chunks to load in the background')
    parser.add_argument('--metrics-file', type = str, default = None, help = 'CSV file for per-epoch metrics')
    parser.add_argument('--resume', type = str, default = None, help = 'Checkpoint path to continue training from')
    parser.add_argument('--start-epoch', type = int, default = 0, help = 'Epoch number used for resumed checkpoint naming and metrics')
    parser.add_argument('--append-metrics', action = 'store_true', help = 'Append to an existing metrics CSV instead of replacing it')
    parser.add_argument('--patience', type = int, default = 3, help = 'Stop after this many consecutive epochs without validation improvement; use -1 to disable')
    parser.add_argument('--seed', type = int, default = 20240618, help = 'Random seed for chunk order and in-chunk shuffling')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    device = _resolve_device(args.device)

    _run_local_training(args, device, base_dir)
