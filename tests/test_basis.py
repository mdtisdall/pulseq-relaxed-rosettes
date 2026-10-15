import numpy as np
import pytest

from relaxed_rosettes.recon.basis import decay_basis

TAU = np.linspace(0.0005, 0.045, 2000)


def test_basis_is_orthonormal_and_meets_the_tolerance() -> None:
    basis = decay_basis(TAU, 5.0, 100.0, tolerance=1e-3)
    np.testing.assert_allclose(basis.functions @ basis.functions.T, np.eye(basis.rank), atol=1e-12)
    assert basis.max_relative_error <= 1e-3
    assert 2 <= basis.rank <= 8
    # Curves between the ones the basis was built from are represented as well.
    for r2star in (7.3, 33.3, 91.0):
        curve = np.exp(-r2star * TAU)
        error = np.linalg.norm(curve - basis.expand(basis.project(curve))) / np.linalg.norm(curve)
        assert error <= 2e-3


def test_rank_grows_with_accuracy_and_range() -> None:
    loose = decay_basis(TAU, 5.0, 100.0, tolerance=1e-2)
    tight = decay_basis(TAU, 5.0, 100.0, tolerance=1e-5)
    wide = decay_basis(TAU, 5.0, 400.0, tolerance=1e-2)
    assert loose.rank < tight.rank
    assert loose.rank < wide.rank
    assert tight.max_relative_error <= 1e-5


def test_fixed_rank_and_its_error() -> None:
    basis = decay_basis(TAU, 5.0, 100.0, rank=2)
    assert basis.rank == 2
    worse = decay_basis(TAU, 5.0, 100.0, rank=1)
    assert basis.max_relative_error < worse.max_relative_error < 1
    # The coefficients of a batch of complex signals, and back.
    signals = np.exp(-np.outer([10.0, 50.0], TAU)) * np.exp(1j * np.array([[0.3], [1.2]]))
    assert basis.project(signals).shape == (2, 2)
    assert basis.expand(basis.project(signals)).shape == signals.shape


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((TAU, 50.0, 10.0), {}),
        ((TAU[:1], 5.0, 100.0), {}),
        ((TAU, 5.0, 100.0), {"curves": 1}),
        ((TAU, 5.0, 100.0), {"rank": 300}),
        ((TAU, 5.0, 100.0), {"tolerance": 0.0}),
    ],
)
def test_invalid_arguments(args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        decay_basis(*args, **kwargs)  # type: ignore[arg-type]
