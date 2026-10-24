"""Call-local scan work and byte-bound publication checks; no live writes."""
import contextlib
from collections import Counter
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from obsidian_vault_mcp import authoring_transaction as tx
from obsidian_vault_mcp import bo_contract, bo_guard, config
from obsidian_vault_mcp.tools import build_orchestrator as bo
from tests.test_build_orchestrator_tools import _build
from tests.test_vault_authoring_speed import configure, feature, fake_registry, S, SEED

_REAL_PARSE = bo_contract.parse_schedule_document


@pytest.fixture
def scan_vault(tmp_path, monkeypatch):
    root = tmp_path / 'vault'
    target = root / S
    target.parent.mkdir(parents=True)
    target.write_text(SEED)
    configure(monkeypatch, root)
    projection = bo_contract.authoring_field_projection()
    monkeypatch.setattr(bo_contract, 'authoring_field_projection', lambda: {
        **projection, 'accepted_input_fields': projection['accepted_input_fields'] | {'status'},
    })
    monkeypatch.setattr(tx, '_guard_inputs', lambda *a: 'stable-synthetic-guard-inputs')
    monkeypatch.setenv('BO_PATH_GUARD_MODE', 'enforce')
    monkeypatch.setattr(bo_contract, 'freely_mutable_statuses', lambda: {'ready'})
    counts = Counter()
    def parse(content, source_name='<string>'):
        counts[source_name, content] += 1
        body = content.split('---', 2)[2] if content.startswith('---') else content
        return yaml.safe_load(body[body.index('builds:'):])
    monkeypatch.setattr(bo_contract, 'parse_schedule_document', _REAL_PARSE)
    monkeypatch.setattr(bo_contract, '_schedule_parser', lambda: SimpleNamespace(parse_schedule_document=parse))
    return root, counts


def schedule(root, name='context.yaml', bid='context'):
    path = root / bo_guard.SCHEDULES_PREFIX / name
    path.write_text('builds:\n  - id: '+bid+'\n')
    return path


def scan_call(bid):
    # On B there is no call-local scan API: exercise the same ordinary scan.
    reads = getattr(bo_contract, 'authoring_reads', contextlib.nullcontext)
    scan = getattr(bo_guard, 'schedule_scan', contextlib.nullcontext)
    with reads(), scan():
        return bo_guard._disk_schedules_referencing(bid)


@pytest.mark.parametrize('entrypoint', ['chain', 'build', 'feature', 'existing', 'validate'])
def test_seven_rows_parse_each_disk_document_once(scan_vault, monkeypatch, entrypoint):
    root, counts = scan_vault
    for i in range(12):
        schedule(root, f'context-{i}.yaml', f'context-{i}')
    if entrypoint == 'chain':
        raw = bo.bo_create_chain([_build(f'scan-{i}') for i in range(7)], S, dry_run=True)
    elif entrypoint == 'build':
        raw = bo.bo_create_build(_build('scan-single'), S, dry_run=True)
    elif entrypoint == 'validate':
        raw = bo.bo_validate_build_graph([_build('scan-validate')], S)
    elif entrypoint == 'existing':
        path = root / bo_guard.SPECS_PREFIX / 'scan-existing.md'
        path.parent.mkdir(parents=True)
        path.write_text('---\nbuild_id: scan-existing\ntier: simple\nproject: edge-trading-system\ndepends_on: []\n---\n# Existing\n\nDo the thing.\n')
        raw = bo.bo_activate_existing_spec('scan-existing', S)
    else:
        from obsidian_vault_mcp import authoring_templates
        monkeypatch.setattr(authoring_templates, 'registry_snapshot', lambda **kw: fake_registry())
        raw = bo.bo_author_feature(**feature(), dry_run=True)
    result = json.loads(raw)
    assert result['ok'], result
    repeated = {source: n for (source, content), n in counts.items() if n > 1}
    assert not repeated, repeated
    if entrypoint != 'validate':
        assert {bo_guard.SCHEDULES_PREFIX+f'context-{i}.yaml' for i in range(12)} <= {s for s, c in counts}


@pytest.mark.parametrize('change', ['edit', 'add', 'delete', 'rename', 'same-stat'])
def test_schedule_changes_between_calls(scan_vault, change):
    root, counts = scan_vault
    path = schedule(root, bid='before')
    assert scan_call('before') == [str(path.relative_to(root))]
    if change in ('edit', 'same-stat'):
        st = path.stat()
        path.write_text(path.read_text().replace('before', 'after!'))
        if change == 'same-stat':
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
            assert path.stat().st_size == st.st_size
            assert path.stat().st_mtime_ns == st.st_mtime_ns
        assert scan_call('before') == []
        assert scan_call('after!') == [str(path.relative_to(root))]
    elif change == 'add':
        added = schedule(root, 'added.yaml', 'before')
        assert scan_call('before') == sorted([str(path.relative_to(root)), str(added.relative_to(root))])
    elif change == 'delete':
        path.unlink()
        assert scan_call('before') == []
    else:
        renamed = path.with_name('renamed.yaml')
        path.rename(renamed)
        assert scan_call('before') == [str(renamed.relative_to(root))]


@pytest.mark.parametrize('change', ['edit', 'add', 'delete', 'rename', 'same-stat'])
@pytest.mark.parametrize('when', ['preparation', 'lock', 'staging'])
def test_midcall_changes_discard_approval(scan_vault, monkeypatch, change, when):
    root, counts = scan_vault
    path = schedule(root, bid='before')
    prepares = 0
    changed = False
    original_prepare = bo._prepare_graph
    def mutate():
        nonlocal changed
        if changed:
            return
        changed = True
        if change in ('edit', 'same-stat'):
            st = path.stat()
            path.write_text(path.read_text().replace('before', 'after!'))
            if change == 'same-stat':
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
                assert path.stat().st_size == st.st_size
                assert path.stat().st_mtime_ns == st.st_mtime_ns
        elif change == 'add':
            schedule(root, 'added.yaml', 'after!')
        elif change == 'delete':
            path.unlink()
        else:
            path.rename(path.with_name('renamed.yaml'))
    def prepare(*args, **kwargs):
        nonlocal prepares
        prepares += 1
        result = original_prepare(*args, **kwargs)
        if prepares == 1 and when == 'preparation':
            mutate()
        if prepares > 1:
            # A fresh preparation must actually observe the new disk view.
            actual = bo_guard._disk_schedules_referencing('after!' if change in ('edit','same-stat','add') else 'before')
            expected = [] if change == 'delete' else [bo_guard.SCHEDULES_PREFIX+('renamed.yaml' if change == 'rename' else 'added.yaml' if change == 'add' else 'context.yaml')]
            assert actual == expected
            result.update(ok=False, errors=[{'code': 'fresh-view-refused'}])
        return result
    monkeypatch.setattr(bo, '_prepare_graph', prepare)
    if when == 'lock':
        original = tx.vault_lock.registry_lock
        @contextlib.contextmanager
        def locked(*args, **kwargs):
            with original(*args, **kwargs):
                mutate()
                yield
        monkeypatch.setattr(tx.vault_lock, 'registry_lock', locked)
    if when == 'staging':
        original = tx._stage_file
        def stage(*args, **kwargs):
            result = original(*args, **kwargs)
            if 'specs' in Path(args[0]).parts:
                mutate()
            return result
        monkeypatch.setattr(tx, '_stage_file', stage)
    result = json.loads(bo.bo_create_build(_build('midcall'), S))
    assert changed
    assert not result['ok'], result
    assert prepares == 2, result
    assert not (root / bo_guard.SPECS_PREFIX / 'midcall.md').exists()
    assert (root / S).read_text() == SEED


def test_every_schedule_participates_in_reference_index(scan_vault):
    root, counts = scan_vault
    paths = [schedule(root, f'{i:02}.yaml', 'shared') for i in range(20)]
    assert scan_call('shared') == [str(p.relative_to(root)) for p in paths]


def test_index_and_approval_fingerprint_use_the_same_read(scan_vault, monkeypatch):
    root, counts = scan_vault
    path = schedule(root, bid='before')
    original_read = Path.read_bytes
    changed = False
    prepares = 0
    original_prepare = bo._prepare_graph
    def read(p):
        nonlocal changed
        raw = original_read(p)
        if p == path and not changed:
            changed = True
            st = path.stat()
            path.write_bytes(raw.replace(b'before', b'after!'))
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        return raw
    def prepare(*args, **kwargs):
        nonlocal prepares
        prepares += 1
        result = original_prepare(*args, **kwargs)
        if prepares > 1:
            assert bo_guard._disk_schedules_referencing('after!') == [str(path.relative_to(root))]
            result.update(ok=False, errors=[{'code': 'fresh-view-refused'}])
        return result
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(bo, '_prepare_graph', prepare)
    result = json.loads(bo.bo_create_build(_build('read-race'), S))
    assert changed
    assert not result['ok'], result
    assert prepares == 2
    assert not (root / bo_guard.SPECS_PREFIX / 'read-race.md').exists()


def test_resolution_once_per_call_and_fresh_next_call(tmp_path, monkeypatch):
    reads = getattr(bo_contract, 'authoring_reads', contextlib.nullcontext)
    path = tmp_path / 'release'
    path.write_text('/release/a')
    calls = []
    def resolve():
        calls.append(1)
        return path.read_text()
    monkeypatch.setattr(config, 'BO_AUTHORING_CONTRACT_PATH', '')
    monkeypatch.setattr(bo_contract, '_resolve_live_orchestrator_directory', resolve)
    with reads():
        assert bo_contract.resolve_authoring_contract_path() == '/release/a/authoring_contract.py'
        assert bo_contract.resolve_authoring_contract_path() == '/release/a/authoring_contract.py'
    st = path.stat()
    path.write_text('/release/b')
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    with reads():
        assert bo_contract.resolve_authoring_contract_path() == '/release/b/authoring_contract.py'
    assert len(calls) == 2


def test_cached_documents_do_not_share_mutable_results(scan_vault):
    root, counts = scan_vault
    reads = getattr(bo_contract, 'authoring_reads', contextlib.nullcontext)
    with reads():
        first = bo_contract.parse_schedule_document('builds: [{id: original}]', 'one')
        first['builds'][0]['id'] = 'poisoned'
        assert bo_contract.parse_schedule_document('builds: [{id: original}]', 'one')['builds'][0]['id'] == 'original'
    assert sum(counts.values()) == 1
