import math
import numpy as np
import torch

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import env.observation_encoding as observation_encoding
from utils.check_winner import check_winner_board

# ============================================================
#   MCTS structures
# ============================================================

class MCTSNode:
    __slots__ = (
        "board", "current_player", "prior",
        "visit_counts", "value_sums", "children",
        "is_terminal", "winner"
    )

    def __init__(self, board, current_player, board_size, win_length):
        self.board = board  # 1D np array
        self.current_player = current_player  # +1 or -1

        self.prior = None  # set when expanded
        num_cells = board_size * board_size
        self.visit_counts = np.zeros(num_cells, dtype=np.int32)
        self.value_sums = np.zeros(num_cells, dtype=np.float32)
        self.children = {}

        self.winner = check_winner_board(board, board_size, win_length)
        self.is_terminal = self.winner is not None

    def Q_values(self):
        N = np.maximum(1, self.visit_counts)
        return self.value_sums / N


class MCTS:
    def __init__(self, net, cfg, device="cpu"):
        self.net = net
        self.cfg = cfg
        self.device = device
        self.board_size = cfg.BOARD_SIZE
        self.win_length = cfg.WIN_LENGTH
        self.num_cells = self.board_size * self.board_size

        # defaults if not in cfg
        self.num_simulations = getattr(cfg, "MCTS_SIMULATIONS", 50)
        self.c_puct = getattr(cfg, "MCTS_C_PUCT", 1.5)
        self.dirichlet_alpha = getattr(cfg, "MCTS_DIRICHLET_ALPHA", 0.3)

    def run(self, board, current_player, add_dirichlet=True):
        """
        Run MCTS from (board, current_player) and return π_t (visit distribution).
        """
        self.root_player = current_player
        root = MCTSNode(board.copy(), current_player,
                        self.board_size, self.win_length)

        if not root.is_terminal:
            self._expand(root)
            if add_dirichlet:
                self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = []

            # Selection
            while True:
                if node.is_terminal:
                    value = self._terminal_value(node.winner)
                    break

                if node.prior is None:
                    # Leaf: expand
                    value = self._expand(node)
                    break

                action, next_node = self._select_child(node)
                search_path.append((node, action))
                node = next_node

            # Backpropagation
            for parent, action in search_path:
                parent.visit_counts[action] += 1
                parent.value_sums[action] += value

        visits = root.visit_counts.astype(np.float32)
        if visits.sum() <= 0:
            # fallback uniform over legal
            pi = np.zeros(self.num_cells, dtype=np.float32)
            legal = np.where(board == 0)[0]
            pi[legal] = 1.0 / len(legal)
            return pi

        pi = visits / visits.sum()
        return pi

    # ----- helpers -----

    def _terminal_value(self, winner):
        if winner == 0 or winner is None:
            return 0.0
        return 1.0 if winner == self.root_player else -1.0

    def _expand(self, node):
        """
        NN evaluation + expansion.
        Return scalar value from root_player POV.
        """
        obs = observation_encoding.encode_obs(
            node.board,
            node.current_player,
            self.cfg
        )
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(obs_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            v = float(v.item())

        valid_mask = (node.board == 0)
        probs = probs * valid_mask
        s = probs.sum()
        if s <= 0:
            legal = np.where(valid_mask)[0]
            probs = np.zeros_like(probs)
            if len(legal) > 0:
                probs[legal] = 1.0 / len(legal)
        else:
            probs /= s

        node.prior = probs

        # convert v (node.current_player POV) → root_player POV
        value = v if node.current_player == self.root_player else -v
        return value

    def _select_child(self, node):
        """
        PUCT: select action with max(Q + U)
        
        CRITICAL FIX (Approach A - Root POV):
        Q values are stored from root_player's perspective throughout the tree.
        When current_player != root_player, we must flip Q so the current player
        maximizes their own advantage, not the opponent's.
        """
        prior = node.prior
        N = node.visit_counts
        Q = node.Q_values()  # Values from root_player POV
        
        # FIX: Flip Q when current player is not root player
        # This ensures current player maximizes their own value
        if node.current_player != self.root_player:
            Q = -Q
        
        total_N = N.sum()

        legal = np.where(node.board == 0)[0]
        if len(legal) == 0:
            return None, node

        U = np.zeros_like(prior, dtype=np.float32)
        U[legal] = (
            self.c_puct * prior[legal] *
            math.sqrt(total_N + 1) /
            (1.0 + N[legal])
        )

        scores = Q + U
        best_action = int(legal[np.argmax(scores[legal])])

        if best_action not in node.children:
            new_board = node.board.copy()
            new_board[best_action] = node.current_player
            child = MCTSNode(new_board, -node.current_player,
                             self.board_size, self.win_length)
            node.children[best_action] = child

        return best_action, node.children[best_action]

    def _add_dirichlet_noise(self, root, epsilon=0.25):
        """
        Add Dirichlet noise at root for exploration (training only).
        """
        legal = np.where(root.board == 0)[0]
        if len(legal) == 0:
            return

        alpha = self.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(legal))

        root.prior[legal] = (
            (1 - epsilon) * root.prior[legal] +
            epsilon * noise
        )


# ============================================================
#   Self-play and training data creation
# ============================================================

def self_play_game(env, mcts, cfg):
    """
    Generate one self-play game with MCTS (with exploration).
    Returns:
        states:  list of board arrays
        pis:     list of π_t distributions
        players: list of current_player (+1/-1)
        winner:  final winner from env (1, -1, 0)
    """
    states, pis, players = [], [], []

    env.reset()
    done = False
    winner = None

    while not done:
        board = env.board.copy()
        current_player = env.current_player

        pi = mcts.run(board, current_player, add_dirichlet=True)

        states.append(board)
        pis.append(pi)
        players.append(current_player)

        # Sample move from π for exploration
        action = int(np.random.choice(len(pi), p=pi))
        _, done, winner = env.step(action)

    return states, pis, players, winner


def make_training_data(states, pis, players, winner, cfg):
    """
    Convert one game into (obs, π, z) triples.
    z is from POV of the player who moved in that state.
    """
    data = []
    for board, pi, player in zip(states, pis, players):
        if winner == 0:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0

        obs = observation_encoding.encode_obs(board, player, cfg)
        data.append((obs.astype(np.float32),
                     pi.astype(np.float32),
                     float(z)))
    return data