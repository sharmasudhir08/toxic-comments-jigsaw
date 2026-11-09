"""
Proposed GRU (experiment_plan.md, Sec. 9): ablation stages P1, P2, P3 (+ P4 from P3), 5 seeds.

    python -m src.rnn_proposed                       # once: cleaning v2 + FastText cache
    python -m src.train_proposed_gru --smoke      # tiny end-to-end check -> results_proposed_smoke/
    python -m src.train_proposed_gru              # full data -> results_proposed/
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.rnn_proposed import main

if __name__ == "__main__":
    main("gru")
