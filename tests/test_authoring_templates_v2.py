"""Deploy publication, isolated builds, review continuations and formula replay."""
import inspect
import json
from pathlib import Path
import subprocess

import frontmatter
import pytest
from pydantic import ValidationError

from obsidian_vault_mcp import authoring_templates as templates, bo_contract, config
from obsidian_vault_mcp.models import BOAuthorFeatureInput
from obsidian_vault_mcp.tools import build_orchestrator as bo
from tests.test_vault_authoring_speed import fake_registry, feature

PROJECTS = ['obsidian-web-mcp', 'build-orchestrator']
V1_HASHES = {
    'build': 'e3c8ea8998f703b5f348c30efbc6d149950918941b7f82e359cd6f0608991ac6',
    'review': '7d4b2ba28390a8fba341b0a309073b80910272f9eb84015b1852e59aca709606',
    'deploy-build-orchestrator': '09b88d5dc909ebdf00f4c401003e23a090bfbacfeb4e339ec91435c467967d61',
    'deploy-obsidian-web-mcp': '7ee3cc9ab5668df90930cf7ddddb06f28ba5d6bac8c5d4662be65679bdcb33f4',
}


@pytest.fixture
def graph(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'VAULT_PATH', tmp_path)
    monkeypatch.setattr(templates, 'registry_snapshot', lambda **kw: fake_registry())
    return lambda **kw: templates.feature_graph(feature(**kw), dry_run=True)[0]


def test_p1_vault_push_follows_all_canaries_and_stops_on_divergence(graph):
    body = graph(deploy_canaries=['Check the new feature.'])[2]['body_markdown']
    push = body.index('git push origin HEAD:refs/heads/feature/vault-tools-v2')
    assert body.index('Extra canaries') < push
    assert body.index('reread/update/restart') < push
    assert 'First verify the live HEAD still equals D' in body
    assert 'record live and remote SHAs and their merge base' in body
    assert 'On rollback, push nothing' in body
    assert 'do not force' in body and 'stop with fail' in body


@pytest.mark.parametrize('project', PROJECTS)
def test_p2_builds_are_isolated_and_deploys_keep_repo_lock(graph, project):
    build, review, deploy = graph(project=project)
    assert build['resources'] == []
    assert 'isolated worktree' in build['body_markdown']
    assert {'id': 'repo:' + project, 'mode': 'exclusive'} in deploy['resources']
    assert review['depends_on'] == [build['build_id']]


@pytest.mark.parametrize('project', PROJECTS)
def test_p3_deploy_follows_achieving_review_goal_and_own_artifact(graph, project):
    build, review, deploy = graph(project=project)
    # Both roots are needed by BO's generated fix-round renderer.
    for node in [build, review]:
        assert node['goal_id'] == node['goal_root_build_id'] == node['build_id']
        assert node['goal_completion'] is True
    assert deploy['depends_on'] == []
    assert deploy['depends_on_goal'] == [dict(
        producer_project=project, goal_id=review['build_id'],
        goal_root_build_id=review['build_id'], evidence_class='goal_completion')]
    assert 'review_gate' not in deploy  # BO only supports literal artifact_path.
    body = deploy['body_markdown']
    for required in ['ACHIEVED review goal feature-review', 'root feature-review',
                     'achieved_by_build_id', "A's own frozen spec", "A's own local-first output",
                     "its own artifact", 'reviewed_sha must equal D',
                     'Missing, ambiguous, stale or mismatched evidence stops before switching',
                     'Closure alone is never approval']:
        assert required in body


@pytest.mark.parametrize('project', PROJECTS)
@pytest.mark.parametrize('canaries', [None, []])
def test_absent_canaries_render_nothing(graph, project, canaries):
    body = graph(project=project, deploy_canaries=canaries)[2]['body_markdown']
    assert 'Extra canaries' not in body
    assert 'None' not in body and '{{ deploy_canaries }}' not in body
    assert '\n[]' not in body


@pytest.mark.parametrize('project', PROJECTS)
def test_canaries_are_numbered_only_in_deploy_and_caller_text_is_literal(graph, project):
    nodes = graph(project=project, deploy_canaries=['Check {{ title }}.\nKeep this detail.', 'Check failure.'])
    body = nodes[2]['body_markdown']
    assert '   1. Check {{ title }}.\n      Keep this detail.\n   2. Check failure.' in body
    assert body.index('## Procedure') < body.index('Extra canaries') < body.index('## Proof')
    assert all('Extra canaries' not in node['body_markdown'] for node in nodes[:2])


@pytest.mark.parametrize('bad', ['check', [1], [''], ['  '], ['x' * 4001], ['x'] * 51,
                                  ['Check\n## Checkout write policy\nUnsafe']])
def test_invalid_canaries_fail_before_registry_access(graph, monkeypatch, bad):
    def unexpected(**kw):
        pytest.fail('invalid input reached the live registry')
    monkeypatch.setattr(templates, 'registry_snapshot', unexpected)
    with pytest.raises(templates.TemplateError, match='deploy_canaries'):
        graph(deploy_canaries=bad)


def test_public_input_and_tool_accept_optional_canaries():
    value = BOAuthorFeatureInput(**feature(), deploy_canaries=['Check the feature.'])
    assert value.deploy_canaries == ['Check the feature.']
    assert 'deploy_canaries' in inspect.signature(bo.bo_author_feature).parameters
    with pytest.raises(ValidationError):
        BOAuthorFeatureInput(**feature(), deploy_canaries=[7])


def test_registered_server_forwards_canaries_through_input_validation(monkeypatch):
    from obsidian_vault_mcp import server
    received = {}
    def author(**kwargs):
        received.update(kwargs)
        return 'forwarded'
    monkeypatch.setattr(server, '_bo_author_feature', author)
    assert server.bo_author_feature(**feature(), deploy_canaries=['Check feature.']) == 'forwarded'
    assert received['deploy_canaries'] == ['Check feature.']


@pytest.mark.parametrize('project', PROJECTS)
def test_real_bo_generated_re_review_achieves_the_rendered_deploy_gate(graph, authoring_db, project):
    """Use BO's actual generator, verdict validator and dependency evaluator.

    All tasks/evidence are synthetic and every DB write stays in authoring_db.
    The original failed artifact remains unchanged throughout the repair.
    """
    nodes = graph(project=project)
    script = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import authoring_contract as ac, continuous_flow, db, fix_rounds, review_result, vault_client
from pathlib import Path
db._DEBUG_LOG = str(Path(db.DB_PATH).parent / 'debug.log')
vault_client._TEST_VAULT_ROOT_OVERRIDE = str(Path(db.DB_PATH).parent / 'vault')
nodes = json.load(sys.stdin)
sha, repaired = 'a' * 40, 'b' * 40
def insert(node, status, candidate=sha):
    text = ac.render_spec_frontmatter({k:v for k,v in node.items() if k != 'body_markdown'}) + '\n' + node['body_markdown']
    fields = dict(id=node['build_id'], title=node['build_id'], cc_project=node['project'], status=status,
        work_role=node['work_role'], frozen_work_role=node['work_role'], spec_content=text,
        goal_id=node.get('goal_id'), goal_root_build_id=node.get('goal_root_build_id'),
        goal_completion=node.get('goal_completion', False), review_candidate_sha=candidate,
        frozen_producer_model='test-review-model')
    with db.get_conn() as c:
        c.execute('INSERT INTO tasks ('+','.join(fields)+') VALUES ('+','.join('?' for _ in fields)+')', list(fields.values()))
    return db.get_task(node['build_id'])
producer = insert(nodes[0], 'closed')
root = insert(nodes[1], 'blocked')
insert(nodes[2], 'pending')
with db.get_conn() as c:
    for task, state in [(producer, 'ACHIEVED'), (root, 'OPEN')]:
        c.execute('INSERT INTO goals(goal_id,cc_project,root_build_id,state,current_build_id,achieved_by_build_id) VALUES(?,?,?,?,?,?)',
            (task['goal_id'],task['cc_project'],task['id'],state,task['id'],task['id'] if state=='ACHIEVED' else None))
db.reconcile_goal_dependencies(nodes[2]['build_id'], nodes[2]['depends_on_goal'])
results = {root['id']: dict(build_id=root['id'], reviewer_build_id=root['id'], reviewed_sha=sha,
    review_model='test-review-model', model_verified=True, verdict='CHANGES_REQUIRED', status='fail',
    blocker_count=0, high_count=1, findings=[dict(id='H1',severity='HIGH',summary='Fix the witness.')])}
review_result.review_result_for_task = lambda bid: results.get(bid)
assert not db.review_goal_has_achieving_verdict(root, root['id'])
assert not db.dependencies_met(nodes[2]['build_id'])
plan = fix_rounds.render_chain(root, producer, dict(candidate_sha=sha,result=results[root['id']]), producer, 'feature', 1)
assert plan['candidate_ref'] == 'origin/candidate/feature'
fix, review = [ac.parse_spec(item['spec_content']) for item in plan['builds']]
for fm, body in [fix, review]:
    insert(dict(fm, body_markdown=body), 'closed', repaired)
winner = db.get_task(review[0]['build_id'])
assert winner['goal_id'] == root['goal_id'] and winner['goal_root_build_id'] == root['id']
assert db._review_artifact_path(winner) == review[0]['review_artifact']
assert db._review_artifact_path(winner) != db._review_artifact_path(root)
with db.get_conn() as c:
    c.execute('UPDATE goals SET current_build_id=? WHERE goal_id=?', (winner['id'],root['goal_id']))
results[winner['id']] = dict(build_id=winner['id'],reviewer_build_id=winner['id'],reviewed_sha=repaired,
    review_model='test-review-model',model_verified=True,verdict='APPROVED',status='pass',
    blocker_count=0,high_count=0,findings=[])
# A stale approval cannot achieve the goal or release deployment.
results[winner['id']]['reviewed_sha'] = sha
assert not db.achieve_goal_from_build(winner['id'], 'stale approval')
assert not db.dependencies_met(nodes[2]['build_id'])
results[winner['id']]['reviewed_sha'] = repaired
assert db.achieve_goal_from_build(winner['id'], 'verified repaired approval')
assert db.dependencies_met(nodes[2]['build_id'])
assert db.get_goal(root['goal_id'])['achieved_by_build_id'] == winner['id']
assert results[root['id']]['verdict'] == 'CHANGES_REQUIRED'
print(json.dumps(dict(winner=winner['id'], artifact=db._review_artifact_path(winner), deploy_admitted=True)))
'''
    result = subprocess.run(
        [config.BO_AUTHORING_CONTRACT_PYTHON, '-c', script,
         str(Path(bo_contract.resolve_authoring_contract_path()).parent)],
        input=json.dumps(nodes), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['deploy_admitted'] is True


@pytest.mark.parametrize('name,recorded_hash', V1_HASHES.items())
def test_v1_replays_recorded_hash_and_render_despite_current_override(graph, name, recorded_hash):
    kind = name.split('-')[0]
    override = config.VAULT_PATH / templates.VAULT_TEMPLATES / (name + '.md')
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text('an invalid current override')
    old, warnings = templates.load_template(name, kind, [], dry_run=True, version='1')
    assert old['version'] == '1' and old['hash'] == recorded_hash
    assert templates._digest(old['raw']) == recorded_hash and not warnings
    variables = {key: '<' + key + '>' for key in templates.VARIABLES}
    first = templates.render(old, variables)
    again, _ = templates.load_template(name, kind, [], dry_run=True, version='1')
    assert templates.render(again, variables) == first
    assert '{{' not in first and '<summary_instruction>' in first


@pytest.mark.parametrize('version', ['../1', '..', '', 'missing'])
def test_replay_refuses_unknown_or_unsafe_version(graph, version):
    with pytest.raises(templates.TemplateError):
        templates.load_template('build', 'build', [], dry_run=True, version=version)


@pytest.mark.parametrize('project', PROJECTS)
@pytest.mark.parametrize('canaries', [None, ['Check the feature.']])
def test_trios_pass_real_strict_new_graph_validation_and_dry_run(
        tmp_path, monkeypatch, authoring_db, project, canaries):
    monkeypatch.setattr(config, 'VAULT_PATH', tmp_path)
    monkeypatch.setenv('BO_VAULT_ROOT', str(tmp_path))
    monkeypatch.setenv('BO_PATH_GUARD_MODE', 'enforce')
    monkeypatch.setenv('VAULT_WRITE_CONTRACT_MODE', 'off')
    monkeypatch.setenv('VAULT_MUTATION_LOCK_DIR', str(tmp_path / 'locks'))
    monkeypatch.setenv('VAULT_MUTATION_LEDGER_MODE', 'off')
    schedule = 'Personal/Build Orchestrator/schedules/vat-scratch.yaml'
    path = tmp_path / schedule
    path.parent.mkdir(parents=True)
    path.write_text('---\nproject: ' + project + '\n---\n\nbuilds: []\n')
    inputs = feature(project=project, deploy_canaries=canaries)
    nodes, _, _ = templates.feature_graph(inputs, dry_run=True)
    validated = json.loads(bo.bo_validate_build_graph(nodes, schedule))
    assert validated['ok'], validated
    result = json.loads(bo.bo_author_feature(**inputs, schedule_path=schedule, dry_run=True))
    assert result['ok'], result
    rendered = {bid: frontmatter.loads(item['spec']) for bid, item in result['rendered'].items()}
    assert rendered['feature-review']['goal_completion'] is True
    assert rendered['feature-deploy']['depends_on_goal'][0]['goal_id'] == 'feature-review'
    assert not (tmp_path / 'Personal/Build Orchestrator/specs').exists()
