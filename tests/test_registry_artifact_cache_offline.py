import asyncio
import importlib.util
import sys
from pathlib import Path

import torch


def _import_server_module(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ONYX_SKIP_LOAD", "1")
    monkeypatch.setenv("ONYX_ALLOW_LOCAL_PATHS", "0")
    monkeypatch.setenv("ONYX_REGISTRY_URL", "http://registry.invalid")
    monkeypatch.setenv("ONYX_ARTIFACT_CACHE_URL", "http://artifact-cache.invalid")
    monkeypatch.setenv("ONYX_ALLOWED_PATH_PREFIXES", str(tmp_path))

    server_path = Path(__file__).resolve().parents[1] / "server.py"
    spec = importlib.util.spec_from_file_location("onyx_api_server_registry_test", server_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["onyx_api_server_registry_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_model_id_s3_resolves_via_artifact_cache_without_local_paths(monkeypatch, tmp_path):
    onyx_api = _import_server_module(monkeypatch, tmp_path)

    class Tok:
        eos_token_id = 2
        unk_token_id = 0

    ckpt_local = tmp_path / "ckpt.pt"
    ckpt_local.write_bytes(b"not a real checkpoint")

    async def fake_fetch(_model_id: int):
        return {
            "id": 1,
            "status": "staging",
            "artifact_uri": "s3://bucket/ckpt.pt",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_size_bytes": 123,
        }

    async def fake_resolve(*, artifact_uri: str, sha256: str, size_bytes: int):
        _ = (artifact_uri, sha256, size_bytes)
        return str(ckpt_local)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def get_num_params(self):
            return 1

    loaded_paths = []

    def fake_load_model(*, checkpoint_path, tokenizer, device, dtype, model_config_path=None):
        _ = (tokenizer, device, dtype, model_config_path)
        loaded_paths.append(checkpoint_path)
        return M().eval(), onyx_api.OnyxConfig()

    mgr = onyx_api.ModelManager(tokenizer=Tok(), device=torch.device("cpu"), dtype=torch.float32)
    monkeypatch.setattr(mgr, "_fetch_registry_model", fake_fetch)
    monkeypatch.setattr(mgr, "_resolve_via_artifact_cache", fake_resolve)
    monkeypatch.setattr(onyx_api, "load_onyx_model", fake_load_model)

    async def run():
        bundle, _key = await mgr.get_bundle(onyx_api.ModelSelector(model_id=1))
        assert bundle.key.checkpoint_path == str(ckpt_local)

    asyncio.run(run())
    assert loaded_paths == [str(ckpt_local)]

