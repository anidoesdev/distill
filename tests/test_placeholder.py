"""Placeholder — real tests added progressively from session 12 onward."""


def test_package_importable() -> None:
    import extractor  # noqa: F401

    assert extractor.__version__ == "0.1.0"
