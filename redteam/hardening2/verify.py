"""Check final recorded evidence; no optimization or score substitution."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cfg = json.loads((ROOT / 'scoring/reference.json').read_text())['experiments']
report = json.loads((ROOT / 'redteam/hardening2/summary.json').read_text())
for split in ('train', 'test'):
    data = json.loads((ROOT / f'scoring/scores_{split}.json').read_text())
    assert set(data['experiments']) == set(cfg)
    for exp, e in cfg.items():
        p = data['experiments'][exp]
        assert set(p['results']) == set(e['results'])
        for method, vals in p['results'].items():
            assert not p['_status'][method]['failed'], (split, exp, method, p['_status'][method])
            assert set(vals) == set(e['metrics'])
            assert all(math.isfinite(v) and v >= 0 for v in vals.values())
        if split == 'test':
            assert '_per_scene' not in p and '_per_scene_summary' not in p
c = report['candidates']
assert report['repeat_max_abs_metric_difference'] < 1e-8
for method in ('orthline', 'orthline_constr', 'riemanline', 'no_optimization', 'gaussian_loss'):
    assert c[method]['rank_delta_vs_reference'] < 0, method
better = c['joint_verified_strict']
assert not better['failed']
assert better['rank_delta_ci95'][0] > 0, better['rank_delta_ci95']
assert better['paired_sequence_block_ci95']['open_structure_joint_ba_clustered'][0] > 0, better['paired_sequence_block_ci95']
for name in ('hidden_abspath', 'constant_lines', 'bad_rotation', 'missing_frames', 'nan_geometry', 'overflow_geometry', 'omit_proposed'):
    assert c[name]['failed'] and c[name]['rank_delta_vs_reference'] < 0, name
    for exp, ecfg in cfg.items():
        worst = json.loads((ROOT / 'scoring/scores_test.json').read_text())['_metadata']
        expected = worst['joint_ba_worst_values'] if exp.startswith('open_structure') else worst['worst_values']
        # Historical audit vectors contain the six ranked targets; the added
        # direct-dimensionality display has coefficient zero.
        expected = {m: v for m, v in expected.items() if abs(ecfg['metrics'][m]['coefficient']) > 0}
        assert c[name]['results'][exp] == expected, (name, exp)
for name in ('zero_params', 'negative_params', 'flags', 'observation_replace', 'scale_gauge', 'public_gt_lookup'):
    assert abs(c[name]['rank_delta_vs_reference']) < 1e-7, (name, c[name]['rank_delta_vs_reference'])
print('Final evidence checks passed: paired dataset headroom, weaker controls, full failure vectors, no paying replayed shortcut.')
