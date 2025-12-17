import importlib.util
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient


def test_generate_accepts_eval_service_fields_and_returns_text(monkeypatch, tmp_path):
    # Ensure startup doesn't try to load real tokenizer/model.
    monkeypatch.setenv("ONYX_SKIP_LOAD", "1")
    monkeypatch.setenv("ONYX_ALLOW_LOCAL_PATHS", "1")
    monkeypatch.setenv("ONYX_ALLOWED_PATH_PREFIXES", str(tmp_path))

    # Import server module by path (pytest plugin import mode can hide cwd).
    server_path = Path(__file__).resolve().parents[1] / "server.py"
    spec = importlib.util.spec_from_file_location("onyx_api_server", server_path)
    assert spec is not None and spec.loader is not None
    onyx_api = importlib.util.module_from_spec(spec)
    sys.modules["onyx_api_server"] = onyx_api
    spec.loader.exec_module(onyx_api)

    class Tok:
        eos_token_id = 2
        unk_token_id = 0

        def encode(self, text, return_tensors="pt"):
            _ = text
            return torch.tensor([[1, 1, 1]], dtype=torch.long)

        def decode(self, token_ids, skip_special_tokens=True):
            _ = skip_special_tokens
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join("x" for _ in token_ids)

        def convert_tokens_to_ids(self, token):
            _ = token
            return self.unk_token_id

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def get_num_params(self):
            return 1

        def init_memory_states(self, batch_size, device, dtype):
            return [{"count": torch.zeros((batch_size,), device=device, dtype=dtype)}]

        def forward(self, input_ids, memory_states=None, update_memories=True, inference_mode=True, kv_cache=None, position_offset=0):
            _ = (memory_states, update_memories, inference_mode, kv_cache, position_offset)
            B, S = input_ids.shape
            # Always emit eos so generation stops immediately.
            logits = torch.full((B, S, 8), -10.0, dtype=torch.float32)
            logits[:, :, 2] = 10.0
            return {"logits": logits, "memory_states": self.init_memory_states(B, input_ids.device, torch.float32), "kv_cache": kv_cache}

    # Patch tokenizer/model manager for offline testing.
    monkeypatch.setattr(onyx_api, "AutoTokenizer", type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: Tok())}))

    async def fake_get_bundle(_selector):
        key = onyx_api.ModelKey(checkpoint_path=str(tmp_path / "a.pt"), model_config_path=None, device="cpu", dtype="float32")
        bundle = onyx_api.ModelBundle(key=key, model=M().eval(), tokenizer=Tok(), device=torch.device("cpu"), dtype=torch.float32, config=onyx_api.OnyxConfig())
        return bundle, "cpu:float32:" + str(tmp_path / "a.pt") + "|"

    # Create client (startup will create manager, but won't preload).
    with TestClient(onyx_api.app) as client:
        onyx_api.state.manager = type("Mgr", (), {"get_bundle": staticmethod(fake_get_bundle)})()

        resp = client.post(
            "/generate",
            json={
                "prompt": "hi",
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": False,
                "model_id": 1,
                "artifact_uri": str(tmp_path / "a.pt"),
                "eval": {"suite": "smoke-v1", "sample_id": "x"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("text"), str)
