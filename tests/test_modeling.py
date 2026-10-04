import pytest
from pretrained_llm_adaptation.modeling import require_transformers

def test_transformers_dependency_is_lazy() -> None:
    try:
        require_transformers()
    except RuntimeError as exc:
        assert "Transformers is required" in str(exc)
    except Exception as exc:
        pytest.fail(f"unexpected dependency failure: {exc}")
