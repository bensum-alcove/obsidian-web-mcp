"""Witnesses, crash/serialization attacks and strict formula/replay controls."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import frontmatter
import pytest
import yaml

from obsidian_vault_mcp import bo_contract, bo_guard, config, vault_lock
from obsidian_vault_mcp.tools import build_orchestrator as bo
from tests.test_build_orchestrator_tools import _build, _fake_render_graph, _ok_validate_graph

S = "Personal/Build Orchestrator/schedules/speed.yaml"
SEED = "---\nproject: edge-trading-system\n---\n\nbuilds: []\n"


def module(name):
    return importlib.import_module("obsidian_vault_mcp." + name)


def fake_registry():
    return {"version": 9, "models": [
        {"id":"test-review-model", "provider":"codex", "roles":["reviewer"], "status":"active"},
        {"id":"test-build-model", "provider":"codex", "roles":["executor"], "status":"active"},
    ], "pools":{"frontier_review":["test-review-model"], "standard_review":["test-review-model"]}}


def configure(monkeypatch, root):
    monkeypatch.setattr(config,"VAULT_PATH",root)
    monkeypatch.setattr(config,"BO_PATH_GUARD_MODE","off")
    monkeypatch.setenv("BO_PATH_GUARD_MODE","off")
    monkeypatch.setenv("VAULT_WRITE_CONTRACT_MODE","off")
    monkeypatch.setenv("VAULT_MUTATION_LOCK_DIR",str(root.parent/'locks'))
    monkeypatch.setenv("VAULT_MUTATION_LEDGER_MODE","off")
    monkeypatch.setattr(bo_contract,"check_version",lambda timeout=None:{"schema_version":6,"contract_version":"1.0.0"})
    def render(items,timeout=None):
        result=_fake_render_graph(items,timeout)
        for item in items:
            dumped=yaml.safe_dump({'builds':[item['schedule_entry']]},sort_keys=False,allow_unicode=True)
            result['rendered'][item['build_id']]['schedule_entry']='\n'.join('  '+line for line in dumped.splitlines()[1:])+'\n'
        return result
    monkeypatch.setattr(bo_contract,"render_graph",render)
    monkeypatch.setattr(bo_contract,"validate_graph",_ok_validate_graph)
    monkeypatch.setattr(bo_contract,"preflight_ids",lambda ids,timeout=None:{"results":dict.fromkeys(ids)})
    monkeypatch.setattr(bo_contract,"preflight_source_schedule",lambda ids,timeout=None:{"results":dict.fromkeys(ids)})
    # Independent syntax parser in this orchestration fixture; real adapter
    # integration and the exact unchanged P2 controls run separately.
    def parse(content,source_name=None):
        text=content.split('---',2)[2] if content.startswith('---') else content
        return yaml.safe_load(text[text.index('builds:'):])
    monkeypatch.setattr(bo_contract,"parse_schedule_document",parse)
    monkeypatch.setattr(bo_contract,"authoring_field_projection",lambda:{
        "accepted_input_fields":frozenset(_build()) | frozenset({"depends_on","deployment_intent","work_role","completion_contract","resources","review_gate","engine","codex_model"}),
        "schedule_entry_keys":frozenset({"id","spec_path","project","tier","depends_on","work_role","engine","codex_model","resources"}),
        "spec_persist_keys":frozenset({"depends_on","deployment_intent","work_role","completion_contract","review_gate","resources","engine","codex_model"}),
    })
    # Transaction validation batch has its own subprocess helper. Keep this
    # unit fixture hermetic while exercising its cache logic independently.
    monkeypatch.setattr(bo_contract,"_invoke",lambda payload,timeout=None:{"results":dict.fromkeys(payload.get("build_ids",[]))})


@pytest.fixture
def scratch(tmp_path,monkeypatch):
    root=tmp_path/'vault';p=root/S;p.parent.mkdir(parents=True);p.write_text(SEED)
    configure(monkeypatch,root)
    return root


def specs(root):
    return sorted((root/'Personal/Build Orchestrator/specs').glob('*.md'))


def data_files(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def fail_staging(monkeypatch, third=False):
    """Same witness on B and candidate: fail after three specs or at schedule."""
    if hasattr(bo,'authoring_transaction'):
        tx=module('authoring_transaction'); original=tx._stage_file; count=0
        def fail(path,data,mode):
            nonlocal count
            if 'specs' in path.parts:
                count+=1
                if third and count==4: raise OSError('W1 client interruption')
            if not third and path.suffix=='.yaml': raise OSError('W2 schedule failure')
            return original(path,data,mode)
        monkeypatch.setattr(tx,'_stage_file',fail)
    else:
        original=bo.write_file_atomic;count=0
        def fail(path,content,**kwargs):
            nonlocal count
            if 'specs/' in path:
                count+=1
                if third and count==4: raise OSError('W1 client interruption')
            if not third and path.endswith('.yaml'):raise OSError('W2 schedule failure')
            return original(path,content,**kwargs)
        monkeypatch.setattr(bo,'write_file_atomic',fail)


def test_w1_seven_row_failure_leaves_no_specs(scratch,monkeypatch):
    fail_staging(monkeypatch,third=True)
    r=json.loads(bo.bo_create_chain([_build(f'w1-{i}') for i in range(7)],S))
    assert not r['ok'];assert not specs(scratch)
    assert (scratch/S).read_text()==SEED


def test_w2_schedule_failure_leaves_no_spec(scratch,monkeypatch):
    fail_staging(monkeypatch)
    r=json.loads(bo.bo_create_build(_build('w2'),S))
    assert not r['ok'];assert not specs(scratch)
    assert (scratch/S).read_text()==SEED


def test_w4_every_formula_executor_keeps_implementation_policy(scratch,monkeypatch):
    templates=module('authoring_templates');monkeypatch.setattr(templates,'registry_snapshot',lambda **kw:fake_registry())
    r=json.loads(bo.bo_author_feature(**feature(),dry_run=True))
    assert r['ok'],r
    assert 'Source and test edits within this build' in r['rendered']['feature']['spec']
    assert r['rendered']['feature']['spec'].count('## Checkout write policy')==1


def test_all_seven_visible_at_the_single_commit_point(scratch,monkeypatch):
    tx=module('authoring_transaction'); original=tx._exchange
    def inspect(before,stage):
        assert not specs(scratch)
        assert (scratch/S).read_text()==SEED
        assert len(list((stage/'specs').glob('*.md')))==7
        original(before,stage)
        assert len(specs(scratch))==7
        assert 'w1-6' in (scratch/S).read_text()
    monkeypatch.setattr(tx,'_exchange',inspect)
    assert json.loads(bo.bo_create_chain([_build(f'w1-{i}') for i in range(7)],S))['ok']


def test_durability_failure_rolls_back_the_whole_snapshot(scratch,monkeypatch):
    tx=module('authoring_transaction');original=tx._sync_dir;exchanged=False
    exchange=tx._exchange
    def swap(a,b):
        nonlocal exchanged
        exchange(a,b);exchanged=not exchanged
    def fail(path):
        if exchanged: raise OSError('directory sync failure')
        return original(path)
    monkeypatch.setattr(tx,'_exchange',swap);monkeypatch.setattr(tx,'_sync_dir',fail)
    r=json.loads(bo.bo_create_chain([_build('rollback-a'),_build('rollback-b')],S))
    assert not r['ok'];assert not specs(scratch);assert (scratch/S).read_text()==SEED


def test_crash_between_spec_staging_and_schedule_is_invisible(scratch):
    # A real process exit, not an exception that could trigger Python cleanup.
    script='''
import os
from pathlib import Path
from pytest import MonkeyPatch
from tests.test_vault_authoring_speed import configure,S,_build
from obsidian_vault_mcp.tools import build_orchestrator as bo
from obsidian_vault_mcp import authoring_transaction as tx
m=MonkeyPatch(); configure(m,Path(os.environ['VAS_VAULT']))
original=tx._stage_file
def crash(path,data,mode):
 if path.suffix=='.yaml':os._exit(73)
 original(path,data,mode)
tx._stage_file=crash
bo.bo_create_chain([_build('crash-a'),_build('crash-b')],S)
'''
    p=subprocess.run([sys.executable,'-c',script],env=dict(os.environ,VAS_VAULT=str(scratch)),timeout=20)
    assert p.returncode==73;assert not specs(scratch);assert (scratch/S).read_text()==SEED


@pytest.mark.parametrize('kind',['build','chain'])
def test_dry_run_is_full_validation_without_any_write(scratch,kind):
    before=data_files(scratch)
    fn=bo.bo_create_build if kind=='build' else bo.bo_create_chain
    r=json.loads(fn(_build() if kind=='build' else [_build()],S,dry_run=True,request_id='preview'))
    assert r['ok'] and r['dry_run'];assert r['rendered'];assert data_files(scratch)==before


def test_request_replay_and_payload_mismatch(scratch):
    first=json.loads(bo.bo_create_build(_build(),S,request_id='stable'))
    assert first['ok'],first
    assert json.loads(bo.bo_create_build(_build(),S,request_id='stable'))==first
    bad=json.loads(bo.bo_create_build(_build(title='different'),S,request_id='stable'))
    assert bad['code']=='request_id_payload_mismatch';assert len(specs(scratch))==1


def test_replay_after_first_failure_returns_first_failure(scratch,monkeypatch):
    fail_staging(monkeypatch)
    first=json.loads(bo.bo_create_build(_build(),S,request_id='failed'))
    assert not first['ok'];monkeypatch.undo()
    configure(monkeypatch,scratch)
    assert json.loads(bo.bo_create_build(_build(),S,request_id='failed'))==first
    assert not specs(scratch)


def test_concurrent_calls_preserve_both_rows(scratch):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda bid:json.loads(bo.bo_create_build(_build(bid),S)),['parallel-a','parallel-b']))
    assert all(r['ok'] for r in results),results
    assert len(specs(scratch))==2
    assert {e['id'] for e in bo_guard.schedule_builds_from_content((scratch/S).read_text())}=={'parallel-a','parallel-b'}


def test_pending_log_writer_uses_the_same_registry(scratch,monkeypatch):
    tx=module('authoring_transaction');entered=threading.Event();release=threading.Event();done=threading.Event()
    original=tx._exchange
    def swap(a,b):
        entered.set();assert release.wait(10);original(a,b)
    monkeypatch.setattr(tx,'_exchange',swap)
    target=scratch/'Personal/Build Orchestrator/operator-note.md'
    with ThreadPoolExecutor(max_workers=2) as pool:
        author=pool.submit(bo.bo_create_build,_build(),S)
        assert entered.wait(10)
        def sync():
            vault_lock.atomic_write(target,b'pending log');done.set()
        writer=pool.submit(sync)
        assert not done.wait(.05)
        release.set();assert json.loads(author.result())['ok'];writer.result()
    assert target.read_bytes()==b'pending log'


def test_async_returns_immediately_and_restart_is_unknown(scratch,monkeypatch):
    jobs=module('authoring_jobs');entered=threading.Event();release=threading.Event()
    original=bo._activate
    def slow(*args,**kw):
        entered.set();assert release.wait(10);return original(*args,**kw)
    monkeypatch.setattr(bo,'_activate',slow)
    start=time.monotonic();r=json.loads(bo.bo_create_build(_build(),S,wait=False,request_id='async'))
    assert time.monotonic()-start<1 and r['job_id'];assert entered.wait(5)
    try:
        assert json.loads(bo.bo_authoring_job(r['job_id']))['status']=='running'
        monkeypatch.setattr(jobs,'_INSTANCE','restarted')
        assert json.loads(bo.bo_authoring_job(r['job_id']))['status']=='unknown'
        assert json.loads(bo.bo_create_build(_build(),S,request_id='async'))['status']=='unknown'
    finally:release.set()
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        status=json.loads(bo.bo_authoring_job(r['job_id']))
        if status['status'] in ('succeeded','failed'):break
        time.sleep(.01)
    assert status['status']=='succeeded',status


def test_synchronous_commit_receipt_recovers_lost_result(scratch,monkeypatch):
    jobs=module('authoring_jobs');r=json.loads(bo.bo_create_build(_build(),S,request_id='receipt'))
    pointer=next((scratch/'.bo-authoring-state/requests').glob('*.json'));bid=json.loads(pointer.read_text())['job_id']
    path=scratch/'.bo-authoring-state/jobs'/f'{bid}.json';record=json.loads(path.read_text())
    record.update(status='running',instance='old-server');record.pop('result');path.write_text(json.dumps(record))
    assert json.loads(bo.bo_create_build(_build(),S,request_id='receipt'))==r
    assert len(specs(scratch))==1


def feature(**overrides):
    return dict(feature_id='feature',title='Faster authoring',project='obsidian-web-mcp',what_this_means='A useful feature.',
                probes='Reproduce the witness.',scope='Implement the bounded change.',proof='Exercise the failure path.',
                out_of_scope='No unrelated work.',risk_domain='control-plane',blast_radius='subsystem',reversible=True,
                shadowable=True,commit_prefix='VAS-')|overrides


@pytest.mark.parametrize('project',['obsidian-web-mcp','build-orchestrator'])
def test_formula_provenance_and_machine_review_gate(scratch,monkeypatch,project):
    templates=module('authoring_templates');monkeypatch.setattr(templates,'registry_snapshot',lambda **kw:fake_registry())
    r=json.loads(bo.bo_author_feature(**feature(project=project)))
    assert r['ok'],r;assert len(specs(scratch))==3
    stored={p.stem:frontmatter.loads(p.read_text()).metadata for p in specs(scratch)}
    assert stored['feature-review']['depends_on']==['feature']
    assert stored['feature-deploy']['depends_on']==['feature-review']
    assert stored['feature-deploy']['review_gate']['required_verdict']=='APPROVED'
    for fm in stored.values():
        assert fm['template_version']=='1' and len(fm['template_hash'])==64
    collision=json.loads(bo.bo_author_feature(**feature(project=project)))
    assert not collision['ok'];assert len(specs(scratch))==3


def test_feature_dry_run_does_not_cache_or_write(scratch,monkeypatch):
    templates=module('authoring_templates');monkeypatch.setattr(templates,'registry_snapshot',lambda **kw:fake_registry())
    before=data_files(scratch)
    r=json.loads(bo.bo_author_feature(**feature(),dry_run=True))
    assert r['ok'],r;assert data_files(scratch)==before


@pytest.mark.parametrize('bad',['{{ unknown }}','{{ title | shell }}','{% include "/etc/passwd" %}','{{ title'])
def test_strict_template_variables_and_filters(scratch,bad):
    templates=module('authoring_templates');raw=(templates.DEFAULTS/'build.md').read_text()+'\n'+bad
    with pytest.raises(ValueError):templates._validate(raw,'build',[])


def test_model_literal_with_valid_hash_is_rejected_and_lkg_is_verified(scratch):
    t=module('authoring_templates');directory=scratch/t.VAULT_TEMPLATES;directory.mkdir(parents=True)
    path=directory/'build.md';good=(t.DEFAULTS/'build.md').read_text().replace("version: '1'","version: '2'")
    path.write_text(good);valid,_=t.load_template('build','build',[])
    path.write_text(good+'\nUse gpt-6.1-sol.\n')
    cached=scratch/'.bo-authoring-state/templates/build.json'
    # Hash-valid but model-bearing content is still invalid, even as an LKG.
    cached.write_text(json.dumps({'raw':path.read_text(),'hash':t._digest(path.read_text())}))
    fallback,notes=t.load_template('build','build',['gpt-6.1-sol'])
    assert fallback['version']=='1';assert notes[0]['code']=='template_fallback'
    _,second=t.load_template('build','build',['gpt-6.1-sol']);assert second==[]
    cached.write_text(json.dumps({'raw':good,'hash':'tampered'}))
    fallback,_=t.load_template('build','build',[]);assert fallback['version']=='1'
    cached.write_text(json.dumps({'raw':good,'hash':t._digest(good)}))
    fallback,_=t.load_template('build','build',[]);assert fallback['version']=='2'


def test_malformed_template_falls_back(scratch):
    t=module('authoring_templates');path=scratch/t.VAULT_TEMPLATES/'build.md';path.parent.mkdir(parents=True)
    path.write_text('---\nversion: [\n---\n{{ unknown }}')
    result,warnings=t.load_template('build','build',[])
    assert result['version']=='1' and warnings


def test_generated_context_is_bounded_and_read_only(scratch,monkeypatch):
    t=module('authoring_templates');monkeypatch.setattr(t,'registry_snapshot',lambda **kw:fake_registry())
    checklist=scratch/t.CHECKLIST;checklist.parent.mkdir(parents=True);checklist.write_text('First rule.\nSecond rule.\n')
    before=data_files(scratch);start=time.monotonic();raw=bo.bo_authoring_context('obsidian-web-mcp');r=json.loads(raw)
    assert r['ok'] and len(raw.encode())<8192 and time.monotonic()-start<5
    assert r['authoring_checklist_lines']==['First rule.','Second rule.']
    assert r['reviewer_classes']['frontier_review']==['test-review-model']
    assert 'Scope' in r['required_spec_sections'];assert data_files(scratch)==before


@pytest.mark.parametrize('change',['reorder','whitespace','title','depends_on'])
def test_w3_frozen_append_cannot_edit_or_reorder_existing(scratch,monkeypatch,change):
    from obsidian_vault_mcp.write_contract import WriteContext
    old='builds:\n  - id: a\n    title: A\n  - id: b\n    title: B\n'
    new=old+'  - id: c\n    title: C\n'
    if change=='reorder':new='builds:\n  - id: b\n    title: B\n  - id: a\n    title: A\n  - id: c\n    title: C\n'
    if change=='whitespace':new=new.replace('title: A','title:  A')
    if change=='title':new=new.replace('title: A','title: changed')
    if change=='depends_on':new=new.replace('title: A','title: A\n    depends_on: [b]')
    monkeypatch.setattr(bo_contract,'preflight_ids',lambda ids,timeout=None:{'results':dict.fromkeys(ids,'running')})
    monkeypatch.setattr(bo_guard,'_freely_mutable_statuses',lambda:{'pending','ready'})
    monkeypatch.setattr(bo_contract,'preflight_schedule_rewrite',lambda *a,**kw:{'errors':[{'code':'frozen_contract_mutation','build_id':'a','message':'frozen'}]})
    assert bo_guard._schedule_rewrite_issues(WriteContext(path=S,old_content=old,new_content=new,tool='test'))


def test_w3_only_literal_add_only_append_relaxes_frozen_context(scratch,monkeypatch):
    from obsidian_vault_mcp.write_contract import WriteContext
    old='builds:\n  - id: a\n    title: A\n'
    new=old+'  - id: b\n    title: B\n'
    monkeypatch.setattr(bo_contract,'preflight_schedule_rewrite',lambda *a,**kw:{'errors':[{'code':'frozen_contract_mutation','build_id':'a','message':'missing baseline'}]})
    assert not bo_guard._schedule_rewrite_issues(WriteContext(path=S,old_content=old,new_content=new,tool='test'))


def test_unsupported_atomic_exchange_and_symlink_fail_closed(scratch,monkeypatch):
    tx=module('authoring_transaction')
    monkeypatch.setattr(tx,'_exchange',lambda *a: (_ for _ in ()).throw(OSError('unsupported exchange')))
    assert not json.loads(bo.bo_create_build(_build(),S))['ok'];assert not specs(scratch)
    (scratch/'Personal/Build Orchestrator/link').symlink_to(scratch/S)
    r=json.loads(bo.bo_create_build(_build(),S));assert not r['ok'] and 'symlink' in r['error'];assert not specs(scratch)


def test_search_and_frontmatter_index_see_commit_and_future_writes(scratch,monkeypatch):
    # Use the real server callback: atomic directory exchange must reinstall
    # watches, not leave the index observing the removed old directory inode.
    import obsidian_vault_mcp.server as server
    from obsidian_vault_mcp.tools.search import vault_search
    monkeypatch.setattr(config,'FRONTMATTER_INDEX_DEBOUNCE',.01)
    monkeypatch.setattr(server,'_semantic_started',False)
    server.frontmatter_index.start()
    try:
        result=json.loads(bo.bo_create_build(_build('indexed',body_markdown='uniquewitnesscontent'),S))
        assert result['ok'],result
        paths={r['path'] for r in server.frontmatter_index.search_by_field('build_id','indexed','exact')}
        assert 'Personal/Build Orchestrator/specs/indexed.md' in paths
        found=json.loads(vault_search('uniquewitnesscontent'))
        assert found['results'],found
        # A normal create event proves the replacement directory is watched.
        # Atomic-file move events are a pre-existing watcher limitation,
        # independent of namespace publication (read/search changes excluded).
        (scratch/'Personal/Build Orchestrator/future.md').write_text(
            '---\nbuild_id: future\n---\nFuture content.')
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if server.frontmatter_index.search_by_field('build_id','future','exact'):break
            time.sleep(.02)
        assert server.frontmatter_index.search_by_field('build_id','future','exact')
    finally:server.frontmatter_index.stop()


def test_validation_cache_is_bulk_read_only_and_does_not_outlive_pass(scratch,monkeypatch):
    # Restore the actual cache wrapper replaced by this file's unit fixture.
    import importlib.util
    path=Path(bo_contract.__file__)
    # Use a fresh module namespace without modifying the shared imported one.
    spec=importlib.util.spec_from_file_location('obsidian_vault_mcp._cache_probe',path)
    adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
    calls=[]
    def invoke(payload,timeout=None):
        calls.append(payload)
        return {'results':{bid:'ready' for bid in payload.get('build_ids',[])}}
    monkeypatch.setattr(adapter,'_invoke_uncached',invoke)
    monkeypatch.setattr(adapter,'resolve_authoring_contract_path',lambda:'/synthetic/authoring_contract.py')
    with adapter.validation_batch(['a','b']):
        assert adapter.preflight_ids(['a'])['results']=={'a':'ready'}
        adapter.preflight_ids(['a']);adapter.preflight_source_schedule(['b'])
    assert len(calls)==2
    adapter.preflight_ids(['a']);assert len(calls)==3


def test_simultaneous_same_request_has_only_one_commit(scratch,monkeypatch):
    tx=module('authoring_transaction');original=tx._exchange;entered=threading.Event();release=threading.Event();count=0
    def swap(a,b):
        nonlocal count
        count+=1;entered.set();assert release.wait(5);original(a,b)
    monkeypatch.setattr(tx,'_exchange',swap)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(bo.bo_create_build,_build(),S,request_id='same')
        assert entered.wait(5)
        replay=json.loads(bo.bo_create_build(_build(),S,request_id='same'))
        assert replay['status'] in ('queued','running')
        release.set();assert json.loads(first.result())['ok']
    assert count==1 and len(specs(scratch))==1


def test_staged_bytes_are_outside_all_vault_search_paths(scratch,monkeypatch):
    tx=module('authoring_transaction');original=tx._exchange
    def inspect(root,stage):
        assert not stage.is_relative_to(scratch)
        assert not any('secret-staged-marker' in p.read_text() for p in scratch.rglob('*.md'))
        original(root,stage)
    monkeypatch.setattr(tx,'_exchange',inspect)
    assert json.loads(bo.bo_create_build(_build(body_markdown='secret-staged-marker'),S))['ok']


def test_real_async_process_restart_reports_unknown(scratch):
    script='''
import os,time,json
from pathlib import Path
from pytest import MonkeyPatch
from tests.test_vault_authoring_speed import configure,S,_build
from obsidian_vault_mcp.tools import build_orchestrator as bo
m=MonkeyPatch();configure(m,Path(os.environ['VAS_VAULT']))
bo._activate=lambda *a,**k:time.sleep(100)
result=bo.bo_create_build(_build(),S,request_id='restart',wait=False)
print(result,flush=True)
os._exit(0)
'''
    p=subprocess.run([sys.executable,'-c',script],env=dict(os.environ,VAS_VAULT=str(scratch)),capture_output=True,text=True,timeout=10)
    assert p.returncode==0,p.stderr
    job=json.loads(p.stdout)
    assert json.loads(bo.bo_authoring_job(job['job_id']))['status']=='unknown'
    assert json.loads(bo.bo_create_build(_build(),S,request_id='restart'))['status']=='unknown'
    assert not specs(scratch)


def test_global_transaction_lock_blocks_a_separate_path_writer(scratch):
    target=scratch/'Personal/Build Orchestrator/process-note.md'
    script='''
import sys
from pathlib import Path
from obsidian_vault_mcp import vault_lock
print('ready',flush=True)
vault_lock.atomic_write(Path(sys.argv[1]),b'process write')
'''
    with vault_lock.registry_lock():
        p=subprocess.Popen([sys.executable,'-c',script,str(target)],stdout=subprocess.PIPE,text=True)
        assert p.stdout.readline().strip()=='ready'
        assert p.poll() is None and not target.exists()
    try:
        assert p.wait(timeout=5)==0
        assert target.read_bytes()==b'process write'
    finally:
        if p.poll() is None:p.kill();p.wait()

# Repair-round controls: these exercise behavior, and run on the original API.
@pytest.mark.parametrize('phase', ['validation', 'index'])
@pytest.mark.parametrize('process', [False, True])
def test_slow_preparation_and_index_allow_unrelated_writer(scratch, monkeypatch, phase, process):
    tx = module('authoring_transaction')
    entered, release = threading.Event(), threading.Event()
    def pause():
        entered.set()
        assert release.wait(15)
    if phase == 'validation':
        original = bo._prepare_graph
        def prepare(*args, **kwargs):
            pause()
            return original(*args, **kwargs)
        monkeypatch.setattr(bo, '_prepare_graph', prepare)
    else:
        monkeypatch.setattr(tx, '_commit_hooks', [lambda paths: pause()])
    target = scratch / 'Personal/Build Orchestrator/logs/concurrent.md'
    child = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        author = pool.submit(bo.bo_create_build, _build('slow-author'), S)
        assert entered.wait(10)
        try:
            start = time.monotonic()
            if process:
                child = subprocess.Popen([sys.executable, '-c',
                    'from pathlib import Path;import sys;from obsidian_vault_mcp.vault_lock import atomic_write;atomic_write(Path(sys.argv[1]),b"other writer")', str(target)])
                child.wait(timeout=5)
                assert child.returncode == 0
            else:
                writer = pool.submit(vault_lock.atomic_write, target, b'other writer')
                writer.result(timeout=5)
            assert time.monotonic() - start < 5
            assert target.read_bytes() == b'other writer'
        finally:
            release.set()
            if child:
                child.wait(timeout=10)
        assert json.loads(author.result(timeout=15))['ok']
    assert target.read_bytes() == b'other writer'


def test_namespace_replacement_during_staging_refuses_publication(scratch, monkeypatch):
    tx = module('authoring_transaction')
    target = scratch / 'Personal/Build Orchestrator/other.md'
    target.write_bytes(b'old')
    original = tx._stage_file
    def replace(path, data, mode):
        original(path, data, mode)
        if path.suffix == '.yaml':
            temp = target.with_suffix('.replacement')
            temp.write_bytes(b'other writer bytes')
            os.replace(temp, target)  # deliberately uncooperating writer
    monkeypatch.setattr(tx, '_stage_file', replace)
    r = json.loads(bo.bo_create_build(_build('stale'), S, request_id='stale'))
    assert not r['ok'], r
    assert 'namespace changed' in r['error']
    assert target.read_bytes() == b'other writer bytes'
    assert not specs(scratch) and (scratch / S).read_text() == SEED
    assert not list((scratch / 'Personal/Build Orchestrator/.authoring-receipts').glob('*'))


def test_registry_lock_excludes_independently_started_path_writer(scratch):
    # Separate interpreters, pipes as readiness/release events. No inherited
    # thread gate can accidentally satisfy the cross-process assertion.
    owner = subprocess.Popen([sys.executable, '-u', '-c',
        'import sys;from obsidian_vault_mcp.vault_lock import registry_lock\nwith registry_lock():\n print("locked",flush=True);sys.stdin.readline()'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    child = None
    try:
        assert owner.stdout.readline().strip() == 'locked'
        target = scratch / 'process-writer.md'
        child = subprocess.Popen([sys.executable, '-u', '-c',
            'import sys;from pathlib import Path;from obsidian_vault_mcp.vault_lock import atomic_write\nprint("ready",flush=True);sys.stdin.readline();atomic_write(Path(sys.argv[1]),b"done")', str(target)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline().strip() == 'ready'
        child.stdin.write('go\n'); child.stdin.flush()
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=.3)
        assert not target.exists()
        owner.stdin.write('release\n'); owner.stdin.flush()
        assert owner.wait(timeout=5) == 0 and child.wait(timeout=5) == 0
        assert target.read_bytes() == b'done'
    finally:
        if owner.poll() is None: owner.kill()
        owner.wait(timeout=5)
        if child and child.poll() is None: child.kill()
        if child: child.wait(timeout=5)


def _frozen_context(monkeypatch):
    from obsidian_vault_mcp.write_contract import WriteContext
    old = 'builds:\n  - id: a\n    title: A\n  - id: b\n    title: B\n'
    monkeypatch.setattr(bo_contract, 'parse_schedule_document', bo_contract._schedule_parser().parse_schedule_document)
    monkeypatch.setattr(bo_contract, 'preflight_schedule_rewrite', lambda *a, **kw: {
        'errors': [{'code': 'frozen_contract_mutation', 'build_id': 'a', 'message': 'base frozen refusal'}]})
    monkeypatch.setattr(bo_contract, 'validate_graph', lambda *a, **kw: {'errors': []})
    return old, lambda new: bo_guard._schedule_rewrite_issues(WriteContext(S, old, new))


def test_nonappend_rewrite_keeps_base_frozen_refusal(scratch, monkeypatch):
    old, issues = _frozen_context(monkeypatch)
    assert issues(old.replace('title: B', 'title: changed'))
    assert not issues(old + '  - id: c\n    title: C\n')


@pytest.mark.parametrize('suffix', [
    'project: build-orchestrator\n', 'builds: []\n', 'other: override\n',
    '---\nproject: build-orchestrator\n', '...\n',
    '    title: duplicate\n',
])
def test_append_suffix_cannot_override_schedule_keys(scratch, monkeypatch, suffix):
    old, issues = _frozen_context(monkeypatch)
    assert issues(old + '  - id: c\n    title: C\n' + suffix)


@pytest.mark.parametrize('field', ['title','what_this_means','probes','scope','proof','out_of_scope','commit_prefix'])
@pytest.mark.parametrize('level', [1,2])
def test_formula_caller_policy_heading_never_deletes_sections(scratch, monkeypatch, field, level):
    t = module('authoring_templates')
    monkeypatch.setattr(t, 'registry_snapshot', lambda **kw: fake_registry())
    before = data_files(scratch)
    r = json.loads(bo.bo_author_feature(**feature(**{field:'Caller {{ literal }}\n' + '#'*level + ' Checkout write policy\nUnsafe policy'}), dry_run=True))
    if r['ok']:
        text = r['rendered']['feature']['spec']
        for heading in ('What this means','Start gate','Probes','Scope','Proof','Out of scope','Output'):
            assert '## '+heading+'\n' in text
        assert text.count('## Checkout write policy') == 1
        assert 'Source and test edits within this build' in text
        assert 'Caller {{ literal }}' in text
    else:
        assert 'policy heading' in r['error']
    assert data_files(scratch) == before


@pytest.mark.parametrize('change', ['schedule','spec','guard','db','registry'])
def test_preparation_input_change_cannot_authorize_stale_commit(scratch, monkeypatch, change):
    tx = module('authoring_transaction')
    original = bo._prepare_graph
    spec = scratch / 'Personal/Build Orchestrator/specs/context.md'
    spec.parent.mkdir(parents=True); spec.write_text('original context')
    schema = scratch / '_schema_rules.json';schema.write_text('{}')
    db_path = scratch / 'db/orchestrator.db'; db_path.parent.mkdir(); db_path.write_bytes(b'synthetic metadata fixture')
    monkeypatch.setenv('BO_TEST_DB_DIR', str(db_path.parent))
    registry = scratch / 'registry.yaml'; registry.write_text('synthetic metadata fixture')
    monkeypatch.setenv('BO_MODEL_REGISTRY_PATH', str(registry))
    calls = 0
    def prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            target = {'schedule': scratch/S, 'spec': spec, 'guard': schema, 'db': db_path, 'registry': registry}[change]
            target.write_text(target.read_text()+'\n')
        else:
            result['ok'] = False
            result['errors'] = [{'code':'changed_input_refused'}]
        return result
    monkeypatch.setattr(bo, '_prepare_graph', prepare)
    r = json.loads(bo.bo_create_build(_build('stale-approval'), S))
    assert not r['ok'], r
    assert not (spec.parent/'stale-approval.md').exists()
    assert calls >= 2 or 'changed' in r.get('error', '')

@pytest.mark.parametrize('baseline', ['entry','file'])
@pytest.mark.parametrize('change', ['pending-reorder','protected-comment','semantic-edit','append-edit','append-reorder'])
def test_ordinary_rewrite_matches_real_adapter(scratch, monkeypatch, baseline, change):
    import hashlib
    monkeypatch.syspath_prepend(str(Path(bo_contract.resolve_authoring_contract_path()).parent))
    import authoring_contract as ac
    from obsidian_vault_mcp.write_contract import WriteContext
    # Use complete real-adapter entry shapes, with one running sibling.
    entries = [dict(id=bid, title=bid, description='Synthetic entry', run_when='no deps', tier='simple', project='edge-trading-system',
                    spec_path=f'Personal/Build Orchestrator/specs/{bid}.md', depends_on=[])
               for bid in ['a','b','c']]
    old = yaml.safe_dump({'builds': entries}, sort_keys=False)
    monkeypatch.setattr(bo_contract, 'parse_schedule_document', bo_contract._schedule_parser().parse_schedule_document)
    if change == 'pending-reorder':
        new = yaml.safe_dump({'builds': [entries[0], entries[2], entries[1]]}, sort_keys=False)
    elif change == 'protected-comment':
        new = old.replace('title: a', 'title: a # harmless comment')
    elif change in ('semantic-edit','append-edit'):
        new = old.replace('title: a', 'title: changed')
    else:
        new = yaml.safe_dump({'builds': [entries[1],entries[0],entries[2]]}, sort_keys=False)
    if change.startswith('append-'):
        new += yaml.safe_dump({'builds': [entries[0] | {'id':'d', 'title':'d', 'spec_path':'Personal/Build Orchestrator/specs/d.md'}]}, sort_keys=False).split('builds:\n', 1)[1]
    state = dict(id='a', status='running', completion_contract_hash='frozen', source_schedule=S,
                 schedule_hash=ac._hash_entry(entries[0]) if baseline == 'entry' else hashlib.sha256(old.encode()).hexdigest())
    monkeypatch.setattr(ac, '_baseline_schedule_text', lambda path: old)
    def authoritative(path, builds, **kwargs):
        r = ac.validate_schedule_rewrite(path, builds, mode='compat_existing',
            existing_state_lookup=lambda bid: state if bid == 'a' else None,
            schedule_bindings_lookup=lambda path: [state])
        return {'errors':r.errors}
    monkeypatch.setattr(bo_contract, 'preflight_schedule_rewrite', authoritative)
    monkeypatch.setattr(bo_contract, 'preflight_ids', lambda ids, **kw: {'results':{bid:'running' if bid=='a' else 'pending' for bid in ids}})
    monkeypatch.setattr(bo_guard, '_freely_mutable_statuses', lambda: {'pending'})
    monkeypatch.setattr(bo_contract, 'validate_graph', lambda *a, **kw: {'errors':[]})
    issues = bo_guard._schedule_rewrite_issues(WriteContext(S, old, new))
    base_errors = authoritative(S, bo_guard.schedule_builds_from_content(new))['errors']
    assert bool(issues) == bool(base_errors), (issues, base_errors)
    if change in ('semantic-edit','append-edit'):
        assert any('frozen_contract_mutation' in i.message for i in issues)
    if change.startswith('append-'):
        # Even an independently allowed ordinary rewrite cannot use the
        # add-only exception to suppress a distinct base frozen refusal.
        monkeypatch.setattr(bo_contract, 'preflight_schedule_rewrite', lambda *a, **kw: {
            'errors':[{'code':'frozen_contract_mutation','build_id':'a','message':'frozen'}]})
        assert bo_guard._schedule_rewrite_issues(WriteContext(S, old, new))


def test_same_schedule_preparations_in_independent_processes_retry(scratch):
    script = '''
import sys,json
from pathlib import Path
from pytest import MonkeyPatch
from tests.test_vault_authoring_speed import configure,S,_build
from obsidian_vault_mcp.tools import build_orchestrator as bo
m=MonkeyPatch();configure(m,Path(sys.argv[1]));original=bo._prepare_graph;calls=0
def prepare(*a,**kw):
 global calls
 calls+=1;result=original(*a,**kw)
 if calls==1: print('prepared',flush=True);sys.stdin.readline()
 return result
bo._prepare_graph=prepare
result=json.loads(bo.bo_create_build(_build(sys.argv[2]),S,request_id=sys.argv[2]))
print(json.dumps({'result':result,'calls':calls}),flush=True)
'''
    children = [subprocess.Popen([sys.executable, '-u', '-c', script, str(scratch), bid],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                for bid in ['process-a','process-b']]
    try:
        for child in children: assert child.stdout.readline().strip() == 'prepared'
        results = []
        for child in children:
            child.stdin.write('commit\n');child.stdin.flush()
            output, _ = child.communicate(timeout=30)
            assert child.returncode == 0
            results.append(json.loads(output.strip()))
        assert all(r['result']['ok'] for r in results), results
        assert results[1]['calls'] >= 2
        entries = bo_guard.schedule_builds_from_content((scratch / S).read_text())
        assert {e['id'] for e in entries} == {'process-a','process-b'}
        assert len(specs(scratch)) == 2
        assert len(list((scratch/'Personal/Build Orchestrator/.authoring-receipts').glob('*.json'))) == 2
    finally:
        for child in children:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)


def test_postcommit_index_failure_is_a_warning(scratch, monkeypatch):
    tx = module('authoring_transaction')
    def fail(paths): raise OSError('synthetic index failure')
    monkeypatch.setattr(tx, '_commit_hooks', [fail])
    r = json.loads(bo.bo_create_build(_build('index-failure'), S, request_id='index-warning'))
    assert r['ok'] and r['created']
    assert 'authoring committed; index refresh needs retry' in r['warnings']
    assert len(specs(scratch)) == 1
    assert json.loads(bo.bo_create_build(_build('index-failure'), S, request_id='index-warning')) == r


def test_formula_checks_required_sections_after_policy_replacement(scratch, monkeypatch):
    t = module('authoring_templates')
    monkeypatch.setattr(t, 'registry_snapshot', lambda **kw: fake_registry())
    template = scratch / t.VAULT_TEMPLATES / 'build.md'
    template.parent.mkdir(parents=True)
    raw = (t.DEFAULTS / 'build.md').read_text()
    # A structurally valid override still must not let the policy splice
    # remove every safety-bearing section after rendering.
    raw = raw.replace('## What this means', '# Checkout write policy\nUnexpected policy\n\n## What this means', 1)
    template.write_text(raw)
    before = data_files(scratch)
    r = json.loads(bo.bo_author_feature(**feature(), dry_run=True))
    assert not r['ok'], r
    assert 'required sections' in r['error']
    assert data_files(scratch) == before
