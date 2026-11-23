import pytest

from dg_wave.convergence import convergence_study


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("kind", ["p", "s"])
def test_three_resolution_wave_convergence(degree, kind):
    report = convergence_study(degree, kind)
    errors = [record["relative_velocity_l2_error"] for record in report["records"]]
    assert errors[2] < errors[1] < errors[0]
    # Smooth solutions on this shape-regular translational tetrahedral family
    # should attain p+1 in L2. Allow 0.2 order for finite-resolution asymptotics,
    # not a larger FP64 algebra tolerance or a replacement for refinement.
    assert report["observed_orders"][-1] >= degree + 1 - 0.2, report
    finest = report["records"][-1]
    assert finest["relative_speed_error"] < 0.01, report
    # Halving RK4 dt must change the solution by <1% of its spatial error.
    assert finest["temporal_difference_over_spatial_error"] < 0.01, report
