from contours import make_example_image, detect_contours

def test_synthetic_contours(tmp_path):
    image = make_example_image(tmp_path / "example.png")
    result = detect_contours(image)
    assert len(result["contours"]) >= 2
    assert result["overlay"].shape == image.shape
