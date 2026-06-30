import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#   Policy + Value Network (CNN, AlphaZero-style)
# ============================================================

class PolicyValueNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.board_size = cfg.BOARD_SIZE
        self.num_cells = self.board_size * self.board_size

        # Use cfg.HIDDEN_SIZE if available; otherwise default
        channels = getattr(cfg, "HIDDEN_SIZE", 64)

        # Shared convolutional trunk
        # Input: (B, 2, N, N)  # my_pieces, opp_pieces
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

        # Policy head: conv → flatten → linear → N*N logits
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * self.board_size * self.board_size,
                                   self.num_cells)

        # Value head: conv → flatten → FC → scalar tanh
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(self.board_size * self.board_size, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        """
        x: (B, 2 * N^2) from encode_obs
        Returns:
            logits: (B, N^2)  # unnormalized policy over cells
            value:  (B,)      # in [-1, 1]
        """
        B = x.shape[0]
        N = self.board_size

        # reshape to image
        x = x.view(B, 2, N, N)

        # shared trunk
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # policy head
        p = F.relu(self.policy_conv(x))      # (B, 2, N, N)
        p = p.view(B, -1)                    # (B, 2*N*N)
        logits = self.policy_fc(p)           # (B, N*N)

        # value head
        v = F.relu(self.value_conv(x))       # (B, 1, N, N)
        v = v.view(B, -1)                    # (B, N*N)
        v = F.relu(self.value_fc1(v))        # (B, H)
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,)

        return logits, v