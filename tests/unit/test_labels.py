"""CompCars label txt -> viewpoint and bbox (PROTOCOL.md §7, issue M1-7)."""

from pathlib import Path

import pytest

from avr.data.labels import parse_label, parse_label_text


def test_lf_label_gives_viewpoint_and_bbox_xyxy():
    # label/78/27/2009/6e8c142af63a1e.txt as distributed (LF endings)
    label = parse_label_text("1\n1\n181 211 724 572\n")

    assert label.viewpoint == 1
    assert label.bbox_xyxy == (181, 211, 724, 572)
    assert label.bbox_degenerate is False


def test_crlf_on_the_viewpoint_line_is_tolerated():
    # label/119/1819/2014/8c99f37820c58b.txt: "5\r\n1\n85 124 707 421\n"
    label = parse_label_text("5\r\n1\n85 124 707 421\n")

    assert label.viewpoint == 5
    assert label.bbox_xyxy == (85, 124, 707, 421)


def test_uncertain_viewpoint_minus_one_is_accepted():
    assert parse_label_text("-1\n1\n10 10 20 20\n").viewpoint == -1


@pytest.mark.parametrize("viewpoint", ["0", "6", "x"])
def test_viewpoint_outside_minus_one_to_five_is_rejected(viewpoint: str):
    with pytest.raises(ValueError, match="viewpoint"):
        parse_label_text(f"{viewpoint}\n1\n10 10 20 20\n")


@pytest.mark.parametrize(
    "bbox",
    [
        "1 1 2 -15",  # label/28/1882/2014/19938294ed55f0.txt: negative y2
        "48 262 48 262",  # label/4/512/2015/4a8aea315e31d3.txt: zero area
        "100 10 50 20",  # x2 < x1
    ],
)
def test_degenerate_bbox_is_flagged_not_raised(bbox: str):
    label = parse_label_text(f"4\n1\n{bbox}\n")

    assert label.bbox_degenerate is True
    assert label.bbox_xyxy == tuple(int(v) for v in bbox.split())


def test_bbox_count_other_than_one_is_rejected():
    with pytest.raises(ValueError, match="bbox"):
        parse_label_text("4\n2\n1 1 2 2\n3 3 4 4\n")


def test_parse_label_reads_a_file(tmp_path: Path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"3\r\n1\n62 174 741 403\n")

    assert parse_label(path).viewpoint == 3
