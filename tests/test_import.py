import pypulseq as pp

import relaxed_rosettes


def test_package_imports() -> None:
    assert relaxed_rosettes.__version__


def test_pypulseq_available() -> None:
    assert pp.Opts() is not None
