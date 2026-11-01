"""End-to-end UI smoke tests using Streamlit's headless AppTest."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _all_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_search_renders_results():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input(key="query").set_value("garbanzo bean dishes")
    at.button(key="search_btn").click().run()
    assert not at.exception
    text = _all_text(at)
    assert "Featured recipes" not in text
    for name in ("Chana Masala", "Classic Hummus", "Falafel Wrap", "Quinoa Chickpea Salad"):
        assert name in text


def test_example_button_and_pantry():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="ex0").click().run()
    assert at.text_input(key="query").value == "quick vegetarian breakfast"
    at.multiselect(key="pantry").set_value(["egg", "bread", "avocado"]).run()
    assert not at.exception
    text = _all_text(at)
    assert "Diet: Vegetarian" in text          # understood-intent chips rendered
    assert "Avocado Toast" in text
