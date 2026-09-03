import importlib.util
from pathlib import Path


def test_feature_module_defines_push_and_batch_views() -> None:
    path = Path("feast/feature_repo/features.py")
    spec = importlib.util.spec_from_file_location("feast_features", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = {
        fv.name
        for fv in [
            module.user_features,
            module.product_features,
            module.seller_features,
            module.session_features,
        ]
    }
    assert names == {
        "user_features",
        "product_features",
        "seller_features",
        "session_features",
    }
    assert module.session_push_source.name == "session_features_push"
    assert module.session_features.online
    assert module.user_features.source.name == "user_features_source"
