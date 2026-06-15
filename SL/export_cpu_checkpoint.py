from pathlib import Path
import argparse


def _load_torch_npu_if_available():
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
    return obj


def _cpu_state_dict(state_dict):
    cpu = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        cpu[clean_key] = value
    return cpu


def main():
    parser = argparse.ArgumentParser(description = 'Re-export a checkpoint as CPU-only state_dict.')
    parser.add_argument('input', help = 'Input checkpoint saved from the training environment.')
    parser.add_argument('output', help = 'Output checkpoint for Botzone CPU runtime.')
    args = parser.parse_args()

    _load_torch_npu_if_available()

    import torch

    checkpoint = torch.load(args.input, map_location = torch.device('cpu'))
    state_dict = _unwrap_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError('Expected a state_dict-like checkpoint, got %r.' % type(state_dict).__name__)

    output = Path(args.output)
    output.parent.mkdir(parents = True, exist_ok = True)
    torch.save(_cpu_state_dict(state_dict), output)
    print('Saved CPU checkpoint to %s' % output)


if __name__ == '__main__':
    main()
