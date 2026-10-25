"""Redeploy and exception-cache witnesses, with optional disk BO corpus."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from obsidian_vault_mcp import authoring_transaction as tx, bo_contract, bo_guard, config
from obsidian_vault_mcp.tools import build_orchestrator as bo
from tests.test_build_orchestrator_tools import _build
from tests.test_guard_schedule_scan_speed import scan_vault
from tests.test_vault_authoring_speed import S, SEED

_REAL_SCHEDULE_PARSER = bo_contract._schedule_parser


@pytest.fixture
def releases(scan_vault, tmp_path, monkeypatch):
    root, counts = scan_vault
    module = _REAL_SCHEDULE_PARSER()
    def parse(content, source_name='<string>'):
        counts[source_name, content] += 1
        return module.parse_schedule_document(content, source_name=source_name)
    monkeypatch.setattr(bo_contract, '_schedule_parser', lambda: SimpleNamespace(parse_schedule_document=parse))
    corpus = os.environ.get('VAF_CORPUS')
    if corpus:
        shutil.copytree(Path(corpus) / tx.NAMESPACE, root / tx.NAMESPACE, dirs_exist_ok=True)
        (root / S).write_text(SEED)
    dirs = [tmp_path / f'release-{i}' for i in range(4)]
    for directory in dirs:
        directory.mkdir()
        (directory / 'authoring_contract.py').write_text('# synthetic release\n')
    serving = tmp_path / 'serving.conf'
    serving.write_text(str(dirs[0]))
    monkeypatch.setattr(config, 'BO_AUTHORING_CONTRACT_PATH', '')
    monkeypatch.setattr(bo_contract, '_resolve_live_orchestrator_directory', lambda: serving.read_text())
    return root, counts, dirs, serving


@pytest.mark.parametrize('phase', ['preparation', 'lock', 'staging'])
@pytest.mark.parametrize('switches', [0, 1, 2])
def test_seven_row_release_switch(releases, monkeypatch, phase, switches):
    root, counts, dirs, serving = releases
    prepared = []
    changes = 0
    def switch():
        nonlocal changes
        if changes < switches:
            changes += 1
            serving.write_text(str(dirs[changes]))
    original = bo._prepare_graph
    def prepare(*args, **kwargs):
        prepared.append(bo_contract.resolve_authoring_contract_path())
        # Catches an unlimited-retry mutation without hanging the test.
        assert len(prepared) <= 2, 'release retry budget exceeded'
        result = original(*args, **kwargs)
        if phase == 'preparation':
            switch()
        return result
    monkeypatch.setattr(bo, '_prepare_graph', prepare)
    if phase == 'lock':
        locked = tx.vault_lock.registry_lock
        @contextlib.contextmanager
        def lock(*args, **kwargs):
            with locked(*args, **kwargs):
                switch()
                yield
        monkeypatch.setattr(tx.vault_lock, 'registry_lock', lock)
    if phase == 'staging':
        stage = tx._stage_file
        def staging(path, *args):
            result = stage(path, *args)
            if path.name == 'vaf-6.md':
                switch()
            return result
        monkeypatch.setattr(tx, '_stage_file', staging)
    result = json.loads(bo.bo_create_chain([_build(f'vaf-{i}') for i in range(7)], S,
                                         request_id='release-switch'))
    print('RELEASE_WITNESS', phase, switches, result.get('ok'), prepared)
    assert changes == switches
    assert prepared == [str(d / 'authoring_contract.py') for d in dirs[:min(switches + 1, 2)]]
    specs = list((root / bo_guard.SPECS_PREFIX).glob('vaf-*.md'))
    receipts = list((root / tx.NAMESPACE / '.authoring-receipts').glob(hashlib.sha256(b'release-switch').hexdigest() + '.json'))
    if switches == 2:
        assert not result['ok'], result
        assert result['code'] == 'bo_release_changed', result
        assert not specs and not receipts
        assert (root / S).read_text() == SEED
    else:
        assert result['ok'], result
        assert len(specs) == 7 and len(receipts) == 1
        assert 'vaf-6' in (root / S).read_text()


@pytest.mark.parametrize('cached', [False, True])
def test_parse_failure_preserves_exception_and_traceback(monkeypatch, cached):
    class ScheduleError(ValueError):
        pass
    original = ScheduleError('malformed schedule')
    cause = RuntimeError('syntax context')
    calls = []
    def parser(*args, **kwargs):
        calls.append(1)
        raise original from cause
    monkeypatch.setattr(bo_contract, '_schedule_parser', lambda: SimpleNamespace(parse_schedule_document=parser))
    with bo_contract.authoring_reads():
        with pytest.raises(ScheduleError) as first:
            bo_contract.parse_schedule_document('builds: [', 'broken.yaml')
        traceback = first.value.__traceback__
        if cached:
            with pytest.raises(ScheduleError) as second:
                bo_contract.parse_schedule_document('builds: [', 'broken.yaml')
            assert second.value is original
            chain = second.value.__traceback__
            while chain is not None and chain is not traceback:
                chain = chain.tb_next
            assert chain is traceback
            assert second.value.__cause__ is cause
        assert first.value is original and first.value.__cause__ is cause
        assert calls == [1]


def test_parse_cache_is_call_local(monkeypatch):
    calls = []
    def parser(*args, **kwargs):
        calls.append(1)
        raise ValueError('broken')
    monkeypatch.setattr(bo_contract, '_schedule_parser', lambda: SimpleNamespace(parse_schedule_document=parser))
    for _ in range(2):
        with bo_contract.authoring_reads(), pytest.raises(ValueError):
            bo_contract.parse_schedule_document('broken', 'broken.yaml')
    assert len(calls) == 2


@pytest.fixture
def live_corpus(tmp_path, monkeypatch, authoring_db):
    corpus = os.environ.get('VAF_CORPUS')
    if not corpus:
        pytest.skip('set VAF_CORPUS to a disk copy of Personal/Build Orchestrator')
    root = tmp_path / 'vault'
    shutil.copytree(Path(corpus) / tx.NAMESPACE, root / tx.NAMESPACE)
    (root / S).write_text(SEED)
    monkeypatch.setattr(config, 'VAULT_PATH', root)
    monkeypatch.setenv('BO_VAULT_ROOT', str(root))
    monkeypatch.setenv('BO_PATH_GUARD_MODE', 'enforce')
    monkeypatch.setenv('VAULT_WRITE_CONTRACT_MODE', 'off')
    monkeypatch.setenv('VAULT_MUTATION_LOCK_DIR', str(tmp_path / 'locks'))
    monkeypatch.setenv('VAULT_MUTATION_LEDGER_MODE', 'off')
    monkeypatch.setattr(bo_guard, '_freely_mutable_statuses_cache', None)
    return root


def test_live_seven_row_timing(live_corpus):
    import time
    timings = []
    for i in range(3):
        before = time.perf_counter()
        result = json.loads(bo.bo_create_chain([
            _build(f'vaf-timing-{j}', deployment_intent='deferred') for j in range(7)
        ], S, dry_run=True))
        elapsed = time.perf_counter() - before
        assert result['ok'], result
        timings.append(round(elapsed, 3))
        assert elapsed < 15, timings
    print('SEVEN_ROW_TIMINGS', timings)
    assert (live_corpus / S).read_text() == SEED
    assert not list((live_corpus / bo_guard.SPECS_PREFIX).glob('vaf-timing-*.md'))


def test_live_corpus_guard_decisions(live_corpus):
    """Real adapter decisions on 12 evenly spaced corpus schedules + attacks."""
    import hashlib
    from obsidian_vault_mcp.write_contract import WriteContext
    paths = sorted((live_corpus / bo_guard.SCHEDULES_PREFIX).rglob('*.yaml'))
    selected = paths[::max(1, len(paths) // 12)][:12]
    decisions = []
    for path in selected:
        old = path.read_text()
        relative = str(path.relative_to(live_corpus))
        for attack, new in [('no-op', old), ('malformed', 'builds: [')]:
            result = bo_guard.evaluate_content(WriteContext(path=relative, old_content=old,
                                                            new_content=new, tool='vaf-parity'))
            decisions.append((relative, attack, result.blocked,
                              [(i.rule_id, i.severity) for i in result.issues]))
    print('GUARD_PARITY', len(decisions), hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest())
    assert len(decisions) == 24
    assert all(not item[2] for item in decisions if item[1] == 'no-op')
    assert all(item[2] for item in decisions if item[1] == 'malformed')


def test_release_identity_changes_behind_same_path(releases, monkeypatch):
    root, counts, dirs, serving = releases
    alias = serving.parent / 'release-alias'
    alias.symlink_to(dirs[0], target_is_directory=True)
    serving.write_text(str(alias))
    prepared = []
    original = bo._prepare_graph
    def prepare(*args, **kwargs):
        prepared.append(str(Path(bo_contract.resolve_authoring_contract_path()).resolve()))
        result = original(*args, **kwargs)
        if len(prepared) == 1:
            alias.unlink()
            alias.symlink_to(dirs[1], target_is_directory=True)
        return result
    monkeypatch.setattr(bo, '_prepare_graph', prepare)
    result = json.loads(bo.bo_create_chain([_build(f'vaf-alias-{i}') for i in range(7)], S))
    assert result['ok'], result
    assert prepared == [str(d / 'authoring_contract.py') for d in dirs[:2]]


def test_release_retry_after_ordinary_input_retries(releases, monkeypatch):
    root, counts, dirs, serving = releases
    prepared = []
    original = bo._prepare_graph
    def prepare(*args, **kwargs):
        prepared.append(bo_contract.resolve_authoring_contract_path())
        result = original(*args, **kwargs)
        if len(prepared) <= 2:
            raise tx.PreparationChanged('synthetic ordinary input change')
        if len(prepared) == 3:
            serving.write_text(str(dirs[1]))
        return result
    monkeypatch.setattr(bo, '_prepare_graph', prepare)
    result = json.loads(bo.bo_create_chain([_build(f'vaf-budget-{i}') for i in range(7)], S))
    assert result['ok'], result
    assert prepared == [str(dirs[0] / 'authoring_contract.py')] * 3 + [str(dirs[1] / 'authoring_contract.py')]


def test_ordinary_retry_cannot_reset_release_budget(releases, monkeypatch):
    root, counts, dirs, serving = releases
    prepared = []
    original = bo._prepare_graph
    def prepare(*args, **kwargs):
        prepared.append(bo_contract.resolve_authoring_contract_path())
        result = original(*args, **kwargs)
        if len(prepared) == 1:
            serving.write_text(str(dirs[1]))
        elif len(prepared) == 2:
            raise tx.PreparationChanged('synthetic ordinary input change')
        elif len(prepared) == 3:
            serving.write_text(str(dirs[2]))
        else:
            pytest.fail('release retry budget was reset')
        return result
    monkeypatch.setattr(bo, '_prepare_graph', prepare)
    result = json.loads(bo.bo_create_chain([_build(f'vaf-cumulative-{i}') for i in range(7)], S))
    assert not result['ok'], result
    assert result['code'] == 'bo_release_changed'
    assert len(prepared) == 3
    assert not list((root / bo_guard.SPECS_PREFIX).glob('vaf-cumulative-*.md'))
    assert (root / S).read_text() == SEED
