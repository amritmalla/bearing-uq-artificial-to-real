from bearing_uq.data.download import archive_url, relocate


def test_archive_url():
    assert archive_url("KA01").endswith("/BearingDataCenter/KA01.rar")


def test_relocate_keeps_only_condition_files(tmp_path):
    work = tmp_path / "x" / "KA01"
    work.mkdir(parents=True)
    for name in ["N15_M07_F10_KA01_1.mat", "N15_M07_F10_KA01_2.mat", "N09_M07_F10_KA01_1.mat"]:
        (work / name).write_bytes(b"x")

    moved = relocate(tmp_path / "x", tmp_path / "raw", "KA01", "N15_M07_F10")

    assert moved == 2
    assert sorted(p.name for p in (tmp_path / "raw" / "KA01").iterdir()) == [
        "N15_M07_F10_KA01_1.mat", "N15_M07_F10_KA01_2.mat"]
