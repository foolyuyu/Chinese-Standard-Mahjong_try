# Agent part
from feature import FeatureAgent

# Botzone interaction
from pathlib import Path
import numpy as np

_MODEL = None

def _resolve_checkpoint_path():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / 'model' / 'checkpoint' / '16.pkl',
        base_dir / 'model' / 'checkpoint' / '13.pkl',
        base_dir / 'model' / 'checkpoint_2' / '13.pkl',
        Path('/data/testrl.pt'),
        Path('/data/19.pkl'),
        Path('/data/13.pkl'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('No checkpoint file found for Botzone submission.')

def load_model():
    global _MODEL
    if _MODEL is None:
        import torch
        from model import CNNModel
        _MODEL = CNNModel()
        _MODEL.load_state_dict(torch.load(_resolve_checkpoint_path(), map_location = torch.device('cpu')))
        _MODEL.train(False)
    return _MODEL

def fallback_response(obs):
    mask = obs.get('action_mask')
    if mask is None:
        return 'Pass'
    for action in np.flatnonzero(mask):
        if FeatureAgent.OFFSET_ACT['Play'] <= action < FeatureAgent.OFFSET_ACT['Chi']:
            return agent.action2response(int(action))
    for action in np.flatnonzero(mask):
        if action != FeatureAgent.OFFSET_ACT['Hu']:
            return agent.action2response(int(action))
    return 'Pass'

def obs2response(obs):
    try:
        model = load_model()
        import torch
    except Exception:
        return fallback_response(obs)
    glob = obs.get('global')
    if glob is None:
        glob = np.zeros(10, dtype = np.float32)
    with torch.no_grad():
        logits, _ = model({
            'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
            'global': torch.from_numpy(np.expand_dims(glob, 0)),
            'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))
        })
        action = logits.argmax(dim = 1).item()
    response = agent.action2response(action)
    return response

import sys

if __name__ == '__main__':
    angang = None
    zimo = False
    try:
        input() # 1
        while True:
            request = input()
            while not request.strip():
                request = input()
            request = request.split()
            if request[0] == '0':
                seatWind = int(request[1])
                agent = FeatureAgent(seatWind)
                agent.request2obs('Wind %s' % request[2])
                print('PASS', flush = True)
            elif request[0] == '1':
                agent.request2obs(' '.join(['Deal', *request[5:]]))
                print('PASS', flush = True)
            elif request[0] == '2':
                obs = agent.request2obs('Draw %s' % request[1])
                response = obs2response(obs)
                response = response.split()
                if response[0] == 'Hu':
                    print('HU', flush = True)
                elif response[0] == 'Play':
                    print('PLAY %s' % response[1], flush = True)
                elif response[0] == 'Gang':
                    print('GANG %s' % response[1], flush = True)
                    angang = response[1]
                elif response[0] == 'AnGang':
                    print('ANGANG %s' % response[1], flush = True)
                    angang = response[1]
                elif response[0] == 'BuGang':
                    print('BUGANG %s' % response[1], flush = True)
            elif request[0] == '3':
                p = int(request[1])
                if request[2] == 'DRAW':
                    agent.request2obs('Player %d Draw' % p)
                    zimo = True
                    print('PASS', flush = True)
                elif request[2] == 'GANG':
                    if p == seatWind and angang:
                        agent.request2obs('Player %d AnGang %s' % (p, angang))
                    elif zimo:
                        agent.request2obs('Player %d AnGang' % p)
                    else:
                        agent.request2obs('Player %d Gang' % p)
                    print('PASS', flush = True)
                elif request[2] == 'BUGANG':
                    obs = agent.request2obs('Player %d BuGang %s' % (p, request[3]))
                    if p == seatWind:
                        print('PASS', flush = True)
                    else:
                        response = obs2response(obs)
                        if response == 'Hu':
                            print('HU', flush = True)
                        else:
                            print('PASS', flush = True)
                else:
                    zimo = False
                    if request[2] == 'CHI':
                        agent.request2obs('Player %d Chi %s' % (p, request[3]))
                    elif request[2] == 'PENG':
                        agent.request2obs('Player %d Peng' % p)
                    obs = agent.request2obs('Player %d Play %s' % (p, request[-1]))
                    if p == seatWind:
                        print('PASS', flush = True)
                    else:
                        response = obs2response(obs)
                        response = response.split()
                        if response[0] == 'Hu':
                            print('HU', flush = True)
                        elif response[0] == 'Pass':
                            print('PASS', flush = True)
                        elif response[0] == 'Gang':
                            print('GANG %s' % response[1], flush = True)
                            angang = None
                        elif response[0] in ('Peng', 'Chi'):
                            obs = agent.request2obs('Player %d '% seatWind + ' '.join(response))
                            response2 = obs2response(obs)
                            print(' '.join([response[0].upper(), *response[1:], response2.split()[-1]]), flush = True)
                            agent.request2obs('Player %d Un' % seatWind + ' '.join(response))
            print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<', flush = True)
    except EOFError:
        pass
