"""
Sketch a tiny PyTorch model here for main.py to trace, e.g.:

    import torch.nn as nn

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(16, 4)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            return x

Keep it to Linear + ReLU layers only, matching what trace.py's docstring
scopes the tracer to support. Two Linear layers is enough to show fusion
mattering (each Linear -> ReLU pair is a fusion opportunity, and two of
them back to back also tests that fusion resets correctly at the matmul
boundary in between).
"""
