"""Goal-conditioned point-to-point tasks: move commanded body parts to target points.

    foot/        one-step foothold: lift the commanded foot, land it on the target, keep balance, stand still
    hand/        (planned) move a hand to a target point with whole-body balance
    whole_body/  (planned) several body-part goals at once
"""
from . import foot  # noqa: F401
