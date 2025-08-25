from math import log, sqrt

from policy.policy import Policy

# ----------------------------------------------------------------------
# This file is derived from ChemTSv2
#   https://github.com/molecule-generator-collection/ChemTSv2
#
# MIT License
# Ref. ChemTSv2: Functional molecular design using de novo molecule generator
# WIREs Computational Molecular Science 13, e1680, (2023). https://doi.org/10.1002/wcms.1680
# ----------------------------------------------------------------------

class Ucb1(Policy):
    def evaluate(child_state, conf):
        ucb1 = (child_state.total_reward / child_state.visits) + conf["c_val"] * sqrt(
            2 * log(child_state.parent_node.state.visits) / child_state.visits
        )
        return ucb1
