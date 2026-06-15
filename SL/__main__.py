# Botzone interaction
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

DATA_DIR = '/data/19.pkl'
DEBUG = False
_MODEL = None
FeatureAgent = None
np = None

def debug(message):
    if DEBUG:
        print('[debug] ' + message, file = sys.stderr, flush = True)

debug('DIAG_VERSION 2026-06-15-0108 start')

def ensure_runtime():
    global FeatureAgent
    global np
    if FeatureAgent is None:
        debug('import runtime begin')
        import numpy as _np
        from feature import FeatureAgent as _FeatureAgent
        np = _np
        FeatureAgent = _FeatureAgent
        debug('import runtime ok')

def ensure_agent():
    global agent
    if agent is None:
        ensure_runtime()
        agent = FeatureAgent(seatWind)
        agent.request2obs('Wind %s' % prevalentWind)
        debug('agent init ok')
    return agent

def load_model():
    global _MODEL
    if _MODEL is None:
        debug('load_model begin')
        import torch
        debug('import torch ok')
        torch.set_num_threads(1)
        try:
            torch.backends.mkldnn.enabled = False
            debug('disable mkldnn ok')
        except Exception as e:
            debug('disable mkldnn skipped: %r' % e)
        from model import CNNModel
        debug('import CNNModel ok')
        _MODEL = CNNModel()
        debug('CNNModel init ok')
        state_dict = torch.load(DATA_DIR, map_location = torch.device('cpu'))
        debug('torch.load %s ok' % DATA_DIR)
        _MODEL.load_state_dict(state_dict)
        debug('load_state_dict ok')
        _MODEL.train(False)
        debug('model ready')
    return _MODEL

def obs2response(obs):
    ensure_runtime()
    model = load_model()
    import torch
    glob = obs.get('global')
    if glob is None:
        glob = np.zeros(10, dtype = np.float32)
    with torch.no_grad():
        debug('forward begin')
        logits = model({'is_training': False, 'obs': {'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)), 'global': torch.from_numpy(np.expand_dims(glob, 0)), 'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))}})
        debug('forward ok')
        action = logits.detach().numpy().flatten().argmax()
    response = agent.action2response(action)
    return response

if __name__ == '__main__':
    angang = None
    zimo = False
    agent = None
    seatWind = None
    prevalentWind = None
    try:
        input() # 1
        while True:
            request = input()
            while not request.strip():
                request = input()
            t = request.split()
            if t[0] == '0':
                seatWind = int(t[1])
                prevalentWind = t[2]
                print('PASS', flush = True)
            elif t[0] == '1':
                agent = ensure_agent()
                agent.request2obs(' '.join(['Deal', *t[5:]]))
                print('PASS', flush = True)
            elif t[0] == '2':
                agent = ensure_agent()
                debug('request draw %s begin' % t[1])
                obs = agent.request2obs('Draw %s' % t[1])
                debug('request draw %s ok' % t[1])
                response = obs2response(obs)
                t = response.split()
                if t[0] == 'Hu':
                    print('HU', flush = True)
                elif t[0] == 'Play':
                    print('PLAY %s' % t[1], flush = True)
                elif t[0] == 'Gang':
                    print('GANG %s' % t[1], flush = True)
                    angang = t[1]
                elif t[0] == 'AnGang':
                    print('GANG %s' % t[1], flush = True)
                    angang = t[1]
                elif t[0] == 'BuGang':
                    print('BUGANG %s' % t[1], flush = True)
            elif t[0] == '3':
                agent = ensure_agent()
                p = int(t[1])
                if t[2] == 'DRAW':
                    agent.request2obs('Player %d Draw' % p)
                    zimo = True
                    print('PASS', flush = True)
                elif t[2] == 'GANG':
                    if p == seatWind and angang:
                        agent.request2obs('Player %d AnGang %s' % (p, angang))
                    elif zimo:
                        agent.request2obs('Player %d AnGang' % p)
                    else:
                        agent.request2obs('Player %d Gang' % p)
                    print('PASS', flush = True)
                elif t[2] == 'BUGANG':
                    obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
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
                    if t[2] == 'CHI':
                        agent.request2obs('Player %d Chi %s' % (p, t[3]))
                    elif t[2] == 'PENG':
                        agent.request2obs('Player %d Peng' % p)
                    obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))
                    if p == seatWind:
                        print('PASS', flush = True)
                    else:
                        response = obs2response(obs)
                        t = response.split()
                        if t[0] == 'Hu':
                            print('HU', flush = True)
                        elif t[0] == 'Pass':
                            print('PASS', flush = True)
                        elif t[0] == 'Gang':
                            print('GANG', flush = True)
                            angang = None
                        elif t[0] in ('Peng', 'Chi'):
                            obs = agent.request2obs('Player %d '% seatWind + response)
                            response2 = obs2response(obs)
                            print(' '.join([t[0].upper(), *t[1:], response2.split()[-1]]), flush = True)
                            agent.request2obs('Player %d Un' % seatWind + response)
            print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<', flush = True)
    except EOFError:
        pass
