def test_package_exposes_version():
    import avr

    assert avr.__version__ == "0.1.0"
