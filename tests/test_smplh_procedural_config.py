"""Config wiring for AMASS-free procedural SMPL-H motion."""

import pytest

from safety_bigym.config import HumanConfig


def test_smplh_motion_default_is_amass():
    cfg = HumanConfig()
    assert cfg.smplh_motion == "amass"


def test_smplh_motion_procedural_valid():
    cfg = HumanConfig(smplh_motion="procedural")
    assert cfg.smplh_motion == "procedural"


def test_smplh_motion_rejects_unknown():
    with pytest.raises(ValueError, match="smplh_motion"):
        HumanConfig(smplh_motion="walk")
