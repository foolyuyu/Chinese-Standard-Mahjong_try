from dataclasses import dataclass
from itertools import permutations

import numpy as np


TILE_LIST = [
    *('W%d' % (i + 1) for i in range(9)),
    *('T%d' % (i + 1) for i in range(9)),
    *('B%d' % (i + 1) for i in range(9)),
    *('F%d' % (i + 1) for i in range(4)),
    *('J%d' % (i + 1) for i in range(3)),
]
TILE_TO_INDEX = {tile: idx for idx, tile in enumerate(TILE_LIST)}

SUIT_LABELS = 'WTB'
HONOR_LABELS = 'J'

ACTION_PASS = 0
ACTION_HU = 1
ACTION_PLAY = 2
ACTION_CHI = 36
ACTION_PENG = 99
ACTION_GANG = 133
ACTION_ANGANG = 167
ACTION_BUGANG = 201

OBS_SHAPE = (148, 4, 9)
SUIT_OBS_SHAPE = (148, 3, 9)
HONOR_OBS_SHAPE = (148, 7)
OBS_TILE_COUNT = 36
SUIT_TILE_COUNT = 27
HONOR_TILE_COUNT = 7
ACT_SIZE = 235


@dataclass(frozen = True)
class AugmentationConfig:
    apply_prob: float = 0.0
    honor_weight: float = 0.25


def _build_tile_transform(suit_perm, rank_flip, honor_perm):
    suit_map = {src: dst for src, dst in zip(SUIT_LABELS, suit_perm)}
    honor_map = {src: dst for src, dst in zip(('J1', 'J2', 'J3'), honor_perm)}

    def transform_tile(tile):
        suit = tile[0]
        if suit in SUIT_LABELS:
            rank = int(tile[1])
            if rank_flip:
                rank = 10 - rank
            return suit_map[suit] + str(rank)
        if suit == HONOR_LABELS:
            return honor_map[tile]
        return tile

    return transform_tile


def _encode_chi(current_tile, called_tile):
    suit = called_tile[0]
    if suit not in SUIT_LABELS or current_tile[0] != suit:
        raise ValueError('Invalid chi tiles: %s %s' % (current_tile, called_tile))
    suit_idx = SUIT_LABELS.index(suit)
    return ACTION_CHI + suit_idx * 21 + (int(called_tile[1]) - 2) * 3 + int(current_tile[1]) - int(called_tile[1]) + 1


def _decode_action(action):
    if action == ACTION_PASS:
        return ('Pass', None)
    if action == ACTION_HU:
        return ('Hu', None)
    if ACTION_PLAY <= action < ACTION_CHI:
        return ('Play', TILE_LIST[action - ACTION_PLAY])
    if ACTION_CHI <= action < ACTION_PENG:
        offset = action - ACTION_CHI
        suit = SUIT_LABELS[offset // 21]
        within = offset % 21
        middle = within // 3 + 2
        pos = within % 3
        current = suit + str(middle + pos - 1)
        called = suit + str(middle)
        return ('Chi', (current, called))
    if ACTION_PENG <= action < ACTION_GANG:
        return ('Peng', TILE_LIST[action - ACTION_PENG])
    if ACTION_GANG <= action < ACTION_ANGANG:
        return ('Gang', TILE_LIST[action - ACTION_GANG])
    if ACTION_ANGANG <= action < ACTION_BUGANG:
        return ('AnGang', TILE_LIST[action - ACTION_ANGANG])
    if ACTION_BUGANG <= action < ACT_SIZE:
        return ('BuGang', TILE_LIST[action - ACTION_BUGANG])
    raise ValueError('Invalid action index: %d' % action)


def _encode_action(kind, payload):
    if kind == 'Pass':
        return ACTION_PASS
    if kind == 'Hu':
        return ACTION_HU
    if kind == 'Play':
        return ACTION_PLAY + TILE_TO_INDEX[payload]
    if kind == 'Chi':
        current, called = payload
        return _encode_chi(current, called)
    if kind == 'Peng':
        return ACTION_PENG + TILE_TO_INDEX[payload]
    if kind == 'Gang':
        return ACTION_GANG + TILE_TO_INDEX[payload]
    if kind == 'AnGang':
        return ACTION_ANGANG + TILE_TO_INDEX[payload]
    if kind == 'BuGang':
        return ACTION_BUGANG + TILE_TO_INDEX[payload]
    raise ValueError('Invalid action kind: %s' % kind)


def _build_variant(suit_perm, rank_flip, honor_perm):
    transform_tile = _build_tile_transform(suit_perm, rank_flip, honor_perm)
    tile_old_to_new = np.arange(OBS_TILE_COUNT, dtype = np.int64)
    for old_idx, tile in enumerate(TILE_LIST):
        tile_old_to_new[old_idx] = TILE_TO_INDEX[transform_tile(tile)]

    tile_new_to_old = np.arange(OBS_TILE_COUNT, dtype = np.int64)
    for old_idx, new_idx in enumerate(tile_old_to_new):
        tile_new_to_old[new_idx] = old_idx

    suit_old_to_new = np.arange(SUIT_TILE_COUNT, dtype = np.int64)
    for old_idx in range(SUIT_TILE_COUNT):
        suit_old_to_new[old_idx] = tile_old_to_new[old_idx]
    suit_new_to_old = np.arange(SUIT_TILE_COUNT, dtype = np.int64)
    for old_idx, new_idx in enumerate(suit_old_to_new):
        suit_new_to_old[new_idx] = old_idx

    honor_old_to_new = np.arange(HONOR_TILE_COUNT, dtype = np.int64)
    for old_idx in range(HONOR_TILE_COUNT):
        honor_old_to_new[old_idx] = tile_old_to_new[27 + old_idx] - 27
    honor_new_to_old = np.arange(HONOR_TILE_COUNT, dtype = np.int64)
    for old_idx, new_idx in enumerate(honor_old_to_new):
        honor_new_to_old[new_idx] = old_idx

    action_old_to_new = np.empty(ACT_SIZE, dtype = np.int64)
    for old_action in range(ACT_SIZE):
        kind, payload = _decode_action(old_action)
        if kind in ('Pass', 'Hu'):
            action_old_to_new[old_action] = old_action
            continue
        if kind == 'Chi':
            current, called = payload
            new_current = transform_tile(current)
            new_called = transform_tile(called)
            action_old_to_new[old_action] = _encode_action('Chi', (new_current, new_called))
            continue
        transformed_payload = transform_tile(payload)
        action_old_to_new[old_action] = _encode_action(kind, transformed_payload)

    action_new_to_old = np.empty(ACT_SIZE, dtype = np.int64)
    for old_idx, new_idx in enumerate(action_old_to_new):
        action_new_to_old[new_idx] = old_idx

    honor_identity = honor_perm == ('J1', 'J2', 'J3')
    return {
        'suit_perm': suit_perm,
        'rank_flip': rank_flip,
        'honor_perm': honor_perm,
        'tile_old_to_new': tile_old_to_new,
        'tile_new_to_old': tile_new_to_old,
        'suit_new_to_old': suit_new_to_old,
        'honor_new_to_old': honor_new_to_old,
        'action_old_to_new': action_old_to_new,
        'action_new_to_old': action_new_to_old,
        'honor_identity': honor_identity,
    }


def build_variants():
    variants = []
    for suit_perm in permutations(SUIT_LABELS):
        for rank_flip in (False, True):
            for honor_perm in permutations(('J1', 'J2', 'J3')):
                variants.append(_build_variant(suit_perm, rank_flip, honor_perm))
    return variants


VARIANTS = build_variants()
VARIANT_WEIGHTS = np.array([
    1.0 if variant['honor_identity'] else 1.0
    for variant in VARIANTS
], dtype = np.float64)
VARIANT_WEIGHTS /= VARIANT_WEIGHTS.sum()


def sample_variant(rng, honor_weight = 0.25):
    weights = np.array([
        1.0 if variant['honor_identity'] else max(0.0, honor_weight)
        for variant in VARIANTS
    ], dtype = np.float64)
    weights /= weights.sum()
    idx = rng.choice(len(VARIANTS), p = weights)
    return VARIANTS[idx]


def transform_sample(sample, variant):
    suit_obs, honor_obs = _sample_split_obs(sample)
    glob = sample['glob']
    mask = sample['mask']
    act = sample['act']

    transformed_suit_obs = _transform_suit_sample(suit_obs, variant)
    transformed_honor_obs = _transform_honor_sample(honor_obs, variant)
    transformed_mask = mask[variant['action_new_to_old']]
    transformed_act = int(variant['action_old_to_new'][int(act)])

    return {
        'suit_obs': transformed_suit_obs,
        'honor_obs': transformed_honor_obs,
        'glob': glob,
        'mask': transformed_mask,
        'act': transformed_act,
    }


def transform_batch(batch, variant):
    suit_obs, honor_obs = _batch_split_obs(batch)
    transformed_suit_obs = _transform_suit_batch(suit_obs, variant)
    transformed_honor_obs = _transform_honor_batch(honor_obs, variant)
    transformed_mask = batch['mask'][:, variant['action_new_to_old']]
    transformed_act = variant['action_old_to_new'][batch['act'].astype(np.int64)].astype(batch['act'].dtype, copy = False)

    return {
        'suit_obs': transformed_suit_obs.astype(suit_obs.dtype, copy = False),
        'honor_obs': transformed_honor_obs.astype(honor_obs.dtype, copy = False),
        'glob': batch['glob'].copy(),
        'mask': transformed_mask.astype(batch['mask'].dtype, copy = False),
        'act': transformed_act,
    }


def _sample_split_obs(sample):
    if 'suit_obs' in sample and 'honor_obs' in sample:
        return sample['suit_obs'], sample['honor_obs']
    obs = sample['obs'].reshape(OBS_SHAPE[0], OBS_TILE_COUNT)
    return obs[:, :27].reshape(SUIT_OBS_SHAPE), obs[:, 27:34].reshape(HONOR_OBS_SHAPE)


def _batch_split_obs(batch):
    if 'suit_obs' in batch and 'honor_obs' in batch:
        return batch['suit_obs'], batch['honor_obs']
    obs = batch['obs'].reshape(batch['obs'].shape[0], OBS_SHAPE[0], OBS_TILE_COUNT)
    return obs[:, :, :27].reshape((obs.shape[0], *SUIT_OBS_SHAPE)), obs[:, :, 27:34].reshape((obs.shape[0], *HONOR_OBS_SHAPE))


def _transform_suit_sample(suit_obs, variant):
    flat = suit_obs.reshape(SUIT_OBS_SHAPE[0], SUIT_TILE_COUNT)
    return flat[:, variant['suit_new_to_old']].reshape(SUIT_OBS_SHAPE)


def _transform_honor_sample(honor_obs, variant):
    flat = honor_obs.reshape(HONOR_OBS_SHAPE)
    return flat[:, variant['honor_new_to_old']]


def _transform_suit_batch(suit_obs, variant):
    flat = suit_obs.reshape(suit_obs.shape[0], SUIT_OBS_SHAPE[0], SUIT_TILE_COUNT)
    return flat[:, :, variant['suit_new_to_old']].reshape(suit_obs.shape)


def _transform_honor_batch(honor_obs, variant):
    flat = honor_obs.reshape(honor_obs.shape[0], HONOR_OBS_SHAPE[0], HONOR_TILE_COUNT)
    return flat[:, :, variant['honor_new_to_old']].reshape(honor_obs.shape)


def maybe_augment_sample(sample, rng, config):
    if config is None or config.apply_prob <= 0:
        return sample
    if rng.random() >= config.apply_prob:
        return sample
    variant = sample_variant(rng, honor_weight = config.honor_weight)
    return transform_sample(sample, variant)


def maybe_augment_batch(batch, rng, config):
    if config is None or config.apply_prob <= 0:
        return batch

    suit_obs = batch['suit_obs'].copy()
    honor_obs = batch['honor_obs'].copy()
    glob = batch['glob'].copy()
    mask = batch['mask'].copy()
    act = batch['act'].copy()

    indices = np.nonzero(rng.random(len(act)) < config.apply_prob)[0]
    if len(indices) == 0:
        return {
            'suit_obs': suit_obs,
            'honor_obs': honor_obs,
            'glob': glob,
            'mask': mask,
            'act': act,
        }

    for idx in indices:
        variant = sample_variant(rng, honor_weight = config.honor_weight)
        sample = {
            'suit_obs': suit_obs[idx],
            'honor_obs': honor_obs[idx],
            'glob': glob[idx],
            'mask': mask[idx],
            'act': act[idx],
        }
        transformed = transform_sample(sample, variant)
        suit_obs[idx] = transformed['suit_obs']
        honor_obs[idx] = transformed['honor_obs']
        glob[idx] = transformed['glob']
        mask[idx] = transformed['mask']
        act[idx] = transformed['act']

    return {
        'suit_obs': suit_obs,
        'honor_obs': honor_obs,
        'glob': glob,
        'mask': mask,
        'act': act,
    }


def build_augmented_subset_batch(batch, rng, config):
    if config is None or config.apply_prob <= 0:
        return None

    selected = np.nonzero(rng.random(len(batch['act'])) < config.apply_prob)[0]
    if len(selected) == 0:
        return None

    suit_obs = []
    honor_obs = []
    glob = []
    mask = []
    act = []
    for idx in selected:
        variant = sample_variant(rng, honor_weight = config.honor_weight)
        transformed = transform_sample({
            'suit_obs': batch['suit_obs'][idx],
            'honor_obs': batch['honor_obs'][idx],
            'glob': batch['glob'][idx],
            'mask': batch['mask'][idx],
            'act': batch['act'][idx],
        }, variant)
        suit_obs.append(transformed['suit_obs'])
        honor_obs.append(transformed['honor_obs'])
        glob.append(transformed['glob'])
        mask.append(transformed['mask'])
        act.append(transformed['act'])

    return {
        'suit_obs': np.stack(suit_obs, axis = 0),
        'honor_obs': np.stack(honor_obs, axis = 0),
        'glob': np.stack(glob, axis = 0),
        'mask': np.stack(mask, axis = 0),
        'act': np.asarray(act),
    }
