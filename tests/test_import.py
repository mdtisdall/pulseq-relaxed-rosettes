import numpy as np
import pypulseq as pp
import pytest

import relaxed_rosettes
from relaxed_rosettes.config import Config
from relaxed_rosettes.system import make_opts


def test_package_imports() -> None:
    assert relaxed_rosettes.__version__


def test_make_opts_matches_config() -> None:
    opts = make_opts(Config())
    assert opts.grad_raster_time == Config().system.grad_raster_time_s
    assert opts.max_grad == pytest.approx(28e-3 * opts.gamma)


def test_pypulseq_has_rotation_extension() -> None:
    # The pinned mcencini fork provides the rotation extension; PyPI pypulseq does not.
    rotation = pp.make_rotation(np.eye(3))
    assert rotation.type == "rot3D"
