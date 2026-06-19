from agent import MahjongGBAgent
from collections import defaultdict
from functools import lru_cache
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise


class FeatureAgent(MahjongGBAgent):

    '''
    suit_observation: 148*3*9 + honor_observation: 148*7 + global(23)
        hand4 + each player(meld7 + discard28)*4 + remaining4
        meld7 = chi1/2/3/4 + peng + exposed_gang + concealed_gang
    global: 23
        seat wind one-hot 4 + prevalent wind one-hot 4 + isAboutKong + min_remaining/21.0
        + route progress features:
          normal shanten/effective, seven pairs shanten/effective,
          knitted distance/effective, knitted straight distance/effective
        + low-risk fan-potential shape statistics:
          max suit ratio, terminal/honor ratio, pair count,
          triplet-like count, chow candidate count
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    '''

    OBS_SIZE = 148
    GLOBAL_SIZE = 23
    ACT_SIZE = 235
    PLAYER_COUNT = 4
    TILE_COUNT = 36
    HAND_SIZE = 4
    MELD_SIZE = 7
    DISCARD_SIZE = 28
    REMAINING_SIZE = 4

    OFFSET_OBS = {
        'HAND' : 0,
        'MELD' : 4,
        'DISCARD' : 32,
        'REMAINING' : 144
    }
    OFFSET_ACT = {
        'Pass' : 0,
        'Hu' : 1,
        'Play' : 2,
        'Chi' : 36,
        'Peng' : 99,
        'Gang' : 133,
        'AnGang' : 167,
        'BuGang' : 201
    }
    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),
        *('T%d' % (i + 1) for i in range(9)),
        *('B%d' % (i + 1) for i in range(9)),
        *('F%d' % (i + 1) for i in range(4)),
        *('J%d' % (i + 1) for i in range(3))
    ]
    OFFSET_TILE = {c : i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.prevalentWind = 0
        self.packs = [[] for i in range(4)]
        self.concealedGangTiles = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, self.TILE_COUNT), dtype = np.int8)
        self.global_obs = np.zeros(self.GLOBAL_SIZE, dtype = np.float32)
        self._global_embedding_update()

    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang

    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request):
        t = request.split()
        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self._global_embedding_update()
            return
        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            self._meld_embedding_update()
            self._discard_embedding_update()
            return
        if t[0] == 'Huang':
            self.valid = []
            return self._obs()
        if t[0] == 'Draw':
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn = True, isAboutKong = self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            self._meld_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()
        if t[2] == 'Hu':
            self.valid = []
            return self._obs()
        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self._discard_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3):
                            tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self._meld_embedding_update()
            self._discard_embedding_update()
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            self._meld_embedding_update()
            self._discard_embedding_update()
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self._meld_embedding_update()
            self._discard_embedding_update()
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            self._meld_embedding_update()
            self._discard_embedding_update()
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            self._meld_embedding_update()
            self._discard_embedding_update()
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            self.concealedGangTiles[p].append(t[3] if p == 0 else None)
            if p == 0:
                self.shownTiles[t[3]] += 4
            self._meld_embedding_update()
            self._discard_embedding_update()
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
                self._hand_embedding_update()
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            self._meld_embedding_update()
            self._discard_embedding_update()
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)

    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def action2response(self, action):
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['Gang']]
        if action < self.OFFSET_ACT['BuGang']:
            return 'AnGang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    '''
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    '''
    def response2action(self, response):
        t = response.split()
        if t[0] == 'Pass':
            return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu':
            return self.OFFSET_ACT['Hu']
        if t[0] == 'Play':
            return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi':
            return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng':
            return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang':
            return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang':
            return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang':
            return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    def _obs(self):
        self._global_embedding_update()
        self._remaining_embedding_update()
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        suit_obs, honor_obs = self._split_tile_embedding()
        return {
            'suit_observation': suit_obs.copy(),
            'honor_observation': honor_obs.copy(),
            'global': self.global_obs.copy(),
            'action_mask': mask
        }

    def _split_tile_embedding(self):
        suit_obs = self.obs[:, :27].reshape((self.OBS_SIZE, 3, 9))
        honor_obs = self.obs[:, 27:34]
        return suit_obs, honor_obs

    def _global_embedding_update(self):
        self.global_obs[:] = 0
        self.global_obs[self.seatWind] = 1
        self.global_obs[4 + self.prevalentWind] = 1
        self.global_obs[8] = 1.0 if self.isAboutKong else 0.0
        self.global_obs[9] = min(self.tileWall) / 21.0
        if hasattr(self, 'hand'):
            features = self._hand_progress_features()
            self.global_obs[10] = self._normalize_shanten(features['normal_shanten'])
            self.global_obs[11] = self._normalize_effective(features['normal_effective'])
            self.global_obs[12] = self._normalize_shanten(features['seven_pairs_shanten'])
            self.global_obs[13] = self._normalize_effective(features['seven_pairs_effective'])
            self.global_obs[14] = self._normalize_distance(features['knitted_distance'], 14)
            self.global_obs[15] = self._normalize_effective(features['knitted_effective'])
            self.global_obs[16] = self._normalize_distance(features['knitted_straight_distance'], 9)
            self.global_obs[17] = self._normalize_effective(features['knitted_straight_effective'])
            shape_stats = self._fan_potential_shape_stats()
            self.global_obs[18] = shape_stats['max_suit_ratio']
            self.global_obs[19] = shape_stats['terminal_honor_ratio']
            self.global_obs[20] = shape_stats['pair_count']
            self.global_obs[21] = shape_stats['triplet_like_count']
            self.global_obs[22] = shape_stats['chow_candidate_count']

    def _hand_progress_features(self):
        counts = self._tile_counts(self.hand)
        open_melds = len(self.packs[0])
        normal_distance = lambda x: self._normal_shanten(x, open_melds)
        seven_distance = lambda x: self._seven_pairs_route_distance(x, open_melds)
        knitted_distance = lambda x: self._knitted_route_distance(x, open_melds)
        knitted_straight_distance = self._knitted_straight_distance
        return {
            'normal_shanten': normal_distance(tuple(counts)),
            'normal_effective': self._route_effective_tile_count(counts, normal_distance),
            'seven_pairs_shanten': seven_distance(tuple(counts)),
            'seven_pairs_effective': self._route_effective_tile_count(counts, seven_distance),
            'knitted_distance': knitted_distance(tuple(counts)),
            'knitted_effective': self._route_effective_tile_count(counts, knitted_distance),
            'knitted_straight_distance': knitted_straight_distance(tuple(counts)),
            'knitted_straight_effective': self._route_effective_tile_count(counts, knitted_straight_distance),
        }

    @staticmethod
    def _normalize_shanten(shanten):
        return (min(max(shanten, -1), 6) + 1) / 7.0

    @staticmethod
    def _normalize_distance(distance, max_distance):
        return min(max(distance, 0), max_distance) / float(max_distance)

    @staticmethod
    def _normalize_effective(effective_tiles):
        return min(effective_tiles, 64) / 64.0

    def _tile_counts(self, tiles):
        counts = [0] * len(self.TILE_LIST)
        for tile in tiles:
            if tile in self.OFFSET_TILE:
                counts[self.OFFSET_TILE[tile]] += 1
        return counts

    def _fan_potential_shape_stats(self):
        hand_counts = self._tile_counts(self.hand)
        counts = self._tile_counts(self.hand + self._own_pack_tiles())
        total_tiles = max(1, sum(counts))
        suit_counts = [sum(counts[i * 9 : (i + 1) * 9]) for i in range(3)]
        terminal_honor_count = sum(counts[idx] for idx in self._terminal_honor_indices())
        pair_count = sum(1 for count in hand_counts if count >= 2)
        triplet_like_count = sum(1 for count in counts if count >= 3)
        chow_candidate_count = self._own_chow_count()
        for suit in range(3):
            base = suit * 9
            for start in range(7):
                if hand_counts[base + start] and hand_counts[base + start + 1] and hand_counts[base + start + 2]:
                    chow_candidate_count += 1
        return {
            'max_suit_ratio': max(suit_counts) / total_tiles,
            'terminal_honor_ratio': terminal_honor_count / total_tiles,
            'pair_count': min(pair_count, 7) / 7.0,
            'triplet_like_count': min(triplet_like_count, 4) / 4.0,
            'chow_candidate_count': min(chow_candidate_count, 21) / 21.0,
        }

    def _own_pack_tiles(self):
        tiles = []
        for packType, tile, offer in self.packs[0]:
            if tile not in self.OFFSET_TILE:
                continue
            if packType == 'CHI':
                if tile[0] not in 'WTB':
                    continue
                num = int(tile[1])
                for delta in [-1, 0, 1]:
                    seq_tile = tile[0] + str(num + delta)
                    if seq_tile in self.OFFSET_TILE:
                        tiles.append(seq_tile)
            elif packType == 'PENG':
                tiles.extend([tile] * 3)
            elif packType == 'GANG':
                tiles.extend([tile] * 4)
        return tiles

    def _own_chow_count(self):
        return sum(1 for packType, tile, offer in self.packs[0] if packType == 'CHI')

    @staticmethod
    @lru_cache(maxsize = 1)
    def _terminal_honor_indices():
        return tuple([0, 8, 9, 17, 18, 26, *range(27, 34)])

    def _route_effective_tile_count(self, counts, distance_fn):
        hand_size = sum(counts)
        if hand_size % 3 == 2:
            best_effective = 0
            for idx, count in enumerate(counts):
                if count == 0:
                    continue
                counts[idx] -= 1
                best_effective = max(best_effective, self._route_effective_tile_count_after_discard(counts, distance_fn))
                counts[idx] += 1
            return best_effective
        return self._route_effective_tile_count_after_discard(counts, distance_fn)

    def _route_effective_tile_count_after_discard(self, counts, distance_fn):
        base_distance = distance_fn(tuple(counts))
        effective_tiles = 0
        for idx, tile in enumerate(self.TILE_LIST):
            remaining = 4 - self.shownTiles[tile] - counts[idx]
            if remaining <= 0:
                continue
            counts[idx] += 1
            next_distance = distance_fn(tuple(counts))
            counts[idx] -= 1
            if next_distance < base_distance:
                effective_tiles += min(4, remaining)
        return effective_tiles

    @staticmethod
    def _seven_pairs_route_distance(counts_tuple, open_melds):
        if open_melds > 0:
            return 7
        return FeatureAgent._seven_pairs_shanten(counts_tuple)

    @staticmethod
    def _knitted_route_distance(counts_tuple, open_melds):
        if open_melds > 0:
            return 14
        present = {idx for idx, count in enumerate(counts_tuple) if count > 0}
        return min(max(0, 14 - len(present & candidate)) for candidate in FeatureAgent._knitted_candidates())

    @staticmethod
    def _knitted_straight_distance(counts_tuple):
        present = {idx for idx, count in enumerate(counts_tuple) if count > 0}
        return min(9 - len(present & candidate) for candidate in FeatureAgent._knitted_straight_candidates())

    @staticmethod
    @lru_cache(maxsize = 1)
    def _knitted_candidates():
        honor_indices = set(range(27, 34))
        return tuple(honor_indices | candidate for candidate in FeatureAgent._knitted_straight_candidates())

    @staticmethod
    @lru_cache(maxsize = 1)
    def _knitted_straight_candidates():
        from itertools import permutations
        rank_groups = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
        candidates = []
        for suit_perm in permutations(range(3)):
            indices = set()
            for group, suit in zip(rank_groups, suit_perm):
                base = suit * 9
                indices.update(base + rank for rank in group)
            candidates.append(frozenset(indices))
        return tuple(candidates)

    @staticmethod
    @lru_cache(maxsize = 200000)
    def _seven_pairs_shanten(counts_tuple):
        pairs = sum(1 for count in counts_tuple if count >= 2)
        unique = sum(1 for count in counts_tuple if count > 0)
        return 6 - pairs + max(0, 7 - unique)

    @staticmethod
    @lru_cache(maxsize = 200000)
    def _normal_shanten(counts_tuple, open_melds):
        from itertools import product
        block_states = [
            FeatureAgent._block_states(counts_tuple[0:9], True),
            FeatureAgent._block_states(counts_tuple[9:18], True),
            FeatureAgent._block_states(counts_tuple[18:27], True),
            FeatureAgent._block_states(counts_tuple[27:34], False),
        ]
        best = 8
        for states in product(*block_states):
            melds = sum(state[0] for state in states)
            taatsu = sum(state[1] for state in states)
            pair = min(1, sum(state[2] for state in states))
            max_taatsu = min(taatsu, max(0, 4 - open_melds - melds))
            best = min(best, 8 - 2 * (open_melds + melds) - max_taatsu - pair)
        return best

    @staticmethod
    @lru_cache(maxsize = 200000)
    def _block_states(counts_tuple, allow_sequence):
        counts = list(counts_tuple)
        states = set()

        def dfs(melds, taatsu, pair):
            idx = 0
            while idx < len(counts) and counts[idx] == 0:
                idx += 1
            if idx == len(counts):
                states.add((melds, taatsu, pair))
                return

            if counts[idx] >= 3:
                counts[idx] -= 3
                dfs(melds + 1, taatsu, pair)
                counts[idx] += 3

            if allow_sequence and idx <= 6 and counts[idx + 1] > 0 and counts[idx + 2] > 0:
                counts[idx] -= 1
                counts[idx + 1] -= 1
                counts[idx + 2] -= 1
                dfs(melds + 1, taatsu, pair)
                counts[idx] += 1
                counts[idx + 1] += 1
                counts[idx + 2] += 1

            if counts[idx] >= 2:
                counts[idx] -= 2
                dfs(melds, taatsu, 1)
                dfs(melds, taatsu + 1, pair)
                counts[idx] += 2

            if allow_sequence and idx <= 7 and counts[idx + 1] > 0:
                counts[idx] -= 1
                counts[idx + 1] -= 1
                dfs(melds, taatsu + 1, pair)
                counts[idx] += 1
                counts[idx + 1] += 1

            if allow_sequence and idx <= 6 and counts[idx + 2] > 0:
                counts[idx] -= 1
                counts[idx + 2] -= 1
                dfs(melds, taatsu + 1, pair)
                counts[idx] += 1
                counts[idx + 2] += 1

            counts[idx] -= 1
            dfs(melds, taatsu, pair)
            counts[idx] += 1

        dfs(0, 0, 0)
        return FeatureAgent._prune_states(states)

    @staticmethod
    def _prune_states(states):
        pruned = []
        for state in states:
            dominated = False
            for other in states:
                if other == state:
                    continue
                if other[0] >= state[0] and other[1] >= state[1] and other[2] >= state[2]:
                    dominated = True
                    break
            if not dominated:
                pruned.append(state)
        return tuple(pruned)

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + self.HAND_SIZE] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            if tile in self.OFFSET_TILE:
                cnt = min(d[tile], self.HAND_SIZE)
                self.obs[self.OFFSET_OBS['HAND'] : self.OFFSET_OBS['HAND'] + cnt, self.OFFSET_TILE[tile]] = 1

    def _meld_embedding_update(self):
        begin = self.OFFSET_OBS['MELD']
        end = self.OFFSET_OBS['DISCARD']
        self.obs[begin : end] = 0
        for p in range(self.PLAYER_COUNT):
            base = begin + p * self.MELD_SIZE
            for tile in self.concealedGangTiles[p]:
                if tile is None:
                    self.obs[base + 6] = 1
                elif tile in self.OFFSET_TILE:
                    self.obs[base + 6, self.OFFSET_TILE[tile]] = 1
            chi_idx = 0
            for packType, tile, offer in self.packs[p]:
                if packType == 'CHI':
                    if tile not in self.OFFSET_TILE:
                        continue
                    suit = tile[0]
                    num = int(tile[1])
                    if suit not in 'WTB':
                        continue
                    seq = [suit + str(num + i) for i in range(3)]
                    if any(seq_tile not in self.OFFSET_TILE for seq_tile in seq):
                        continue
                    if chi_idx >= 4:
                        continue
                    row = base + chi_idx
                    for seq_tile in seq:
                        self.obs[row, self.OFFSET_TILE[seq_tile]] = 1
                    chi_idx += 1
                elif packType == 'PENG':
                    if tile in self.OFFSET_TILE:
                        self.obs[base + 4, self.OFFSET_TILE[tile]] = 1
                elif packType == 'GANG':
                    if tile in self.OFFSET_TILE:
                        self.obs[base + 5, self.OFFSET_TILE[tile]] = 1

    def _discard_embedding_update(self):
        begin = self.OFFSET_OBS['DISCARD']
        end = self.OFFSET_OBS['REMAINING']
        self.obs[begin : end] = 0
        for p in range(self.PLAYER_COUNT):
            base = begin + p * self.DISCARD_SIZE
            for idx, tile in enumerate(self.history[p][-self.DISCARD_SIZE:]):
                if tile in self.OFFSET_TILE:
                    self.obs[base + idx, self.OFFSET_TILE[tile]] = 1

    def _remaining_embedding_update(self):
        begin = self.OFFSET_OBS['REMAINING']
        self.obs[begin : ] = 0
        for tile, idx in self.OFFSET_TILE.items():
            remaining = 4 - self.shownTiles[tile] - self.hand.count(tile)
            remaining = max(0, min(3, remaining))
            self.obs[begin + remaining, idx] = 1

    def _check_mahjong(self, winTile, isSelfDrawn = False, isAboutKong = False):
        try:
            fans = MahjongFanCalculator(
                pack = tuple(self.packs[0]),
                hand = tuple(self.hand),
                winTile = winTile,
                flowerCount = 0,
                isSelfDrawn = isSelfDrawn,
                is4thTile = (self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong = isAboutKong,
                isWallLast = self.wallLast,
                seatWind = self.seatWind,
                prevalentWind = self.prevalentWind,
                verbose = True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8:
                raise Exception('Not Enough Fans')
        except:
            return False
        return True
