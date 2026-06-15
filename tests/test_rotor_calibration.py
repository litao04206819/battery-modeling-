import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.config import load_yaml
from bmr.rotor.bemt import BEMTRotor
from bmr.rotor.calibration import calibrate_to_coefficients


def _rotor():
    return BEMTRotor(load_yaml("rotor.yaml"))


def test_coefficients_positive():
    rotor = _rotor()
    ct, cq = rotor.coefficients(350.0)
    assert ct > 0 and cq > 0


def test_calibration_matches_targets():
    rotor = _rotor()
    target = load_yaml("rotor.yaml")["coefficients"]
    result = calibrate_to_coefficients(
        rotor, target_CT=target["CT"], target_CQ=target["CQ"], ref_speed_rad_s=350.0
    )
    # Calibrated coefficients should match the targets to within 5 %.
    assert result.ct_error / result.target_CT < 0.05
    assert result.cq_error / result.target_CQ < 0.05
    # Parameters stay inside the physical bounds.
    assert -12.0 <= result.pitch_offset_deg <= 12.0
    assert 0.004 <= result.cd0 <= 0.06


def test_calibration_mutates_rotor():
    rotor = _rotor()
    before = (rotor.pitch_offset, rotor.cd0)
    calibrate_to_coefficients(rotor, target_CT=0.105, target_CQ=0.0052, ref_speed_rad_s=350.0)
    after = (rotor.pitch_offset, rotor.cd0)
    assert after != before
