# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for restart-safe artifact publication markers."""

import json
from types import SimpleNamespace

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata import artifact_lifecycle as al
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot


def _transfer(tmp_path):
    return SimpleNamespace(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(
            ArtifactCacheRoot(
                name="primary",
                source_root=tmp_path / "cache",
                target_root=tmp_path / "cache",
            ),
        ),
    )


def _identity():
    return p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name="test-model",
    )


def test_publish_marker_skips_a_live_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    ctx = SimpleNamespace(global_rank=0)

    marker_path = al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity())

    assert marker_path is not None
    assert al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity()) is None


def test_publish_marker_skips_a_live_owner_without_starttime(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: None)
    ctx = SimpleNamespace(global_rank=0)

    marker_path = al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity())

    assert marker_path is not None
    assert json.loads(marker_path.read_text())["starttime"] is None
    assert al.mark_publish_scheduled(ctx, _transfer(tmp_path), _identity()) is None


def test_publish_marker_skips_an_owner_without_pid_permission(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)

    def denied_pid(pid, signal):
        raise PermissionError

    monkeypatch.setattr(al.os, "kill", denied_pid)
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps({"version": 1, "pid": 1234, "starttime": "100", "worker_rank": 0})
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) is None
    assert json.loads(marker_path.read_text())["pid"] == 1234


def test_publish_marker_reclaims_a_dead_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)

    def dead_pid(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(al.os, "kill", dead_pid)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "200")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps({"version": 1, "pid": 1234, "starttime": "100", "worker_rank": 0})
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text()) == {
        "pid": 4242,
        "starttime": "200",
        "version": 1,
        "worker_rank": 1,
    }


def test_publish_marker_reclaims_a_reused_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "new-starttime")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps(
            {"version": 1, "pid": 1234, "starttime": "old-starttime", "worker_rank": 0}
        )
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text())["starttime"] == "new-starttime"


def test_publish_marker_reclaims_legacy_rank_only_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("0")

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text())["pid"] == 4242
    assert json.loads(marker_path.read_text())["starttime"] == "100"


def test_publish_marker_reclaims_a_boolean_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(transfer, identity, "publish-scheduled")
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps({"version": 1, "pid": True, "starttime": "100", "worker_rank": 0})
    )

    assert al.mark_publish_scheduled(SimpleNamespace(global_rank=1), transfer, identity) == marker_path
    assert json.loads(marker_path.read_text())["pid"] == 4242


def test_mooncake_marker_hit_is_not_a_p2p_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    transfer = _transfer(tmp_path)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "artifact.bin").write_bytes(b"installed")
    identity = _identity()
    marker_path = al.artifact_marker_path(
        transfer, identity, "mooncake-install-attempted"
    )
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("artifact-id\n")

    calls = []
    monkeypatch.setattr(
        al,
        "install_from_mooncake",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    ctx = SimpleNamespace(
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    result = al.install_mooncake_artifact_once(
        ctx, transfer, identity, engine_label="test"
    )

    assert result.status is al.MooncakeInstallStatus.ALREADY_INSTALLED
    assert calls == []


def test_mooncake_miss_is_shared_by_live_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    transfer = _transfer(tmp_path)
    identity = _identity()
    ctx = SimpleNamespace(
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )

    calls = []

    def miss(*args, **kwargs):
        calls.append(1)
        raise al.MooncakeArtifactCacheMiss("missing")

    monkeypatch.setattr(al, "install_from_mooncake", miss)
    try:
        al.install_mooncake_artifact_once(
            ctx, transfer, identity, engine_label="test"
        )
    except al.MooncakeArtifactCacheMiss:
        pass
    else:
        raise AssertionError("expected Mooncake miss")

    try:
        al.install_mooncake_artifact_once(
            ctx, transfer, identity, engine_label="test"
        )
    except al.MooncakeArtifactCacheMiss:
        pass
    else:
        raise AssertionError("expected shared Mooncake miss")
    assert calls == [1]


def test_mooncake_miss_is_shared_without_starttime(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: None)
    transfer = _transfer(tmp_path)
    identity = _identity()
    ctx = SimpleNamespace(
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    calls = []

    def miss(*args, **kwargs):
        calls.append(1)
        raise al.MooncakeArtifactCacheMiss("missing")

    monkeypatch.setattr(al, "install_from_mooncake", miss)
    for _ in range(2):
        try:
            al.install_mooncake_artifact_once(
                ctx, transfer, identity, engine_label="test"
            )
        except al.MooncakeArtifactCacheMiss:
            pass
        else:
            raise AssertionError("expected shared Mooncake miss")

    assert calls == [1]


def test_mooncake_marker_keeps_an_owner_without_pid_permission(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(
        transfer, identity, "mooncake-install-attempted"
    )
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text(
        json.dumps({"status": "miss", "pid": 1234, "starttime": "100"})
    )

    def denied_pid(pid, signal):
        raise PermissionError

    monkeypatch.setattr(al.os, "kill", denied_pid)
    try:
        al.install_mooncake_artifact_once(
            SimpleNamespace(
                node_rank=0,
                accelerator_backend=SimpleNamespace(name="cuda"),
            ),
            transfer,
            identity,
            engine_label="test",
        )
    except al.MooncakeArtifactCacheMiss:
        pass
    else:
        raise AssertionError("expected shared Mooncake miss")


def test_mooncake_marker_reclaims_invalid_owner_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(al.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(al.os, "getpid", lambda: 4242)
    monkeypatch.setattr(al, "_process_starttime", lambda pid: "100")
    transfer = _transfer(tmp_path)
    identity = _identity()
    marker_path = al.artifact_marker_path(
        transfer, identity, "mooncake-install-attempted"
    )
    marker_path.parent.mkdir(parents=True)

    calls = []

    def miss(*args, **kwargs):
        calls.append(1)
        raise al.MooncakeArtifactCacheMiss("missing")

    monkeypatch.setattr(al, "install_from_mooncake", miss)
    ctx = SimpleNamespace(
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )
    for invalid_marker in (
        {"status": "miss", "pid": True, "starttime": "100"},
        {"status": "miss", "pid": 1234},
    ):
        marker_path.write_text(json.dumps(invalid_marker))
        try:
            al.install_mooncake_artifact_once(
                ctx, transfer, identity, engine_label="test"
            )
        except al.MooncakeArtifactCacheMiss:
            pass
        else:
            raise AssertionError("expected Mooncake miss")
        assert json.loads(marker_path.read_text())["pid"] == 4242

    assert calls == [1, 1]


def _mooncake_publish_context():
    return SimpleNamespace(
        global_rank=0,
        node_rank=0,
        accelerator_backend=SimpleNamespace(name="cuda"),
    )


def test_mooncake_publish_without_p2p_returns_mooncake_source_id(monkeypatch, tmp_path):
    transfer = _transfer(tmp_path)
    source = tmp_path / "cache"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"cache")
    identity = _identity()
    ctx = _mooncake_publish_context()
    bundle = object()
    al._mooncake_publish_needed.clear()
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(transfer, "prepare_source", lambda: bundle, raising=False)
    published = []
    monkeypatch.setattr(
        al,
        "publish_to_mooncake",
        lambda *args, **kwargs: published.append(args[2]),
    )
    al._mark_mooncake_publish_needed(ctx, transfer, identity)

    result = al._publish_mooncake_then_p2p_artifact(
        ctx,
        transfer,
        identity,
        engine_label="test",
        p2p_publish_fn=lambda *_: pytest.fail("P2P must be disabled"),
        p2p_publish_available=False,
        log=al.logger,
    )

    assert result == "mooncake-artifact-cache"
    assert published == [bundle]


def test_mooncake_publish_failure_uses_p2p_when_available(monkeypatch, tmp_path):
    transfer = _transfer(tmp_path)
    source = tmp_path / "cache"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"cache")
    identity = _identity()
    ctx = _mooncake_publish_context()
    al._mooncake_publish_needed.clear()
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(transfer, "prepare_source", lambda: object(), raising=False)
    monkeypatch.setattr(al, "publish_to_mooncake", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    al._mark_mooncake_publish_needed(ctx, transfer, identity)

    result = al._publish_mooncake_then_p2p_artifact(
        ctx,
        transfer,
        identity,
        engine_label="test",
        p2p_publish_fn=lambda *_: SimpleNamespace(endpoint=SimpleNamespace(mx_source_id="p2p-id")),
        p2p_publish_available=True,
        log=al.logger,
    )

    assert result == "p2p-id"


def test_mooncake_publish_failure_without_p2p_is_explicit(monkeypatch, tmp_path):
    transfer = _transfer(tmp_path)
    source = tmp_path / "cache"
    source.mkdir()
    (source / "artifact.bin").write_bytes(b"cache")
    identity = _identity()
    ctx = _mooncake_publish_context()
    al._mooncake_publish_needed.clear()
    monkeypatch.setenv("MX_ARTIFACT_BACKEND", "mooncake")
    monkeypatch.setattr(transfer, "prepare_source", lambda: object(), raising=False)
    monkeypatch.setattr(al, "publish_to_mooncake", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    al._mark_mooncake_publish_needed(ctx, transfer, identity)

    with pytest.raises(RuntimeError, match="Mooncake artifact publish failed"):
        al._publish_mooncake_then_p2p_artifact(
            ctx,
            transfer,
            identity,
            engine_label="test",
            p2p_publish_fn=lambda *_: pytest.fail("P2P must be disabled"),
            p2p_publish_available=False,
            log=al.logger,
        )
