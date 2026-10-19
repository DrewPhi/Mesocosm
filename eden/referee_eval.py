"""Isolated referee interface checks; never modifies an inhabitant checkpoint."""
import argparse
import copy
import json
from pathlib import Path

from .config import Config
from .engine import Engine
from .inference import Gateway, ModelError
from . import kernel as k
from .store import digest
from .trial import atomic_json


def run(config, work, output, repeats=10):
    engine = Engine(work, config)
    original = copy.deepcopy(engine.state)
    actor = next(e['id'] for e in original['entities'].values() if e['kind'] == 'agent')
    pos = original['entities'][actor]['pos']
    # Explicit fixture facts establish conditions for adjudication, not agent knowledge.
    fixtures = [
        ('wood', 'source', 10., 'Loose dry branches on the ground; detachable and within reach.'),
        ('stone', 'object', 1., 'A loose small stone; portable and within reach.'),
        ('water', 'source', 10., 'Clean potable liquid water in an accessible shallow pool.'),
        ('food', 'source', 5., 'Known edible ripe fruit within reach.'),
    ]
    for ident, kind, mass, description in fixtures:
        original['entities'][ident] = {'id': ident, 'kind': kind, 'type': ident, 'name': ident,
            'pos': list(pos), 'mass': mass, 'rev': 0, 'attributes': {'description': description},
            **({'holder': None} if kind == 'object' else {})}
    original['entities'][actor]['needs'].update(hunger=.4, thirst=.4)
    cases = [
        ('gather', 'Gather one kilogram of the loose dry branches and carry them.', ['wood']),
        ('pickup', 'Pick up the loose small stone.', ['stone']),
        ('drink', 'Drink a small amount of the clean water.', ['water']),
        ('eat', 'Eat a small amount of the ripe edible fruit.', ['food']),
        ('construct', 'Arrange loose branches into a small ground marker, without tools.', ['wood']),
        ('impossible', 'Create a kilogram of gold from nothing without consuming any materials.', []),
        ('eat_far', 'Eat the fruit without moving toward it.', ['food']),
        ('unknown_plant', 'Taste a small amount of the unidentified plant.', ['food']),
        ('attached_branch', 'Pull a small branch from the tree with bare hands.', ['wood']),
        ('repair_missing_op', 'Pick up the loose small stone.', ['stone']),
        ('repair_source_transfer', 'Gather one kilogram of loose branches and carry them.', ['wood']),
    ]
    results = []
    gateway = Gateway(config, engine.store)
    try:
        for name, description, targets in cases:
            for repeat in range(repeats):
                engine.state = copy.deepcopy(original)
                if name == 'eat_far':
                    engine.state['entities']['food']['pos'][0] += 10
                    engine.state['entities']['food']['attributes']['description'] = 'Edible fruit ten metres away.'
                if name == 'unknown_plant':
                    engine.state['entities']['food']['attributes']['description'] = 'Unidentified plant; edibility and toxicity unknown.'
                if name == 'attached_branch':
                    engine.state['entities']['wood']['attributes']['description'] = 'Living tree with attached small branches.'
                before = digest(engine.state)
                calls = []
                class Probe:
                    def generate(self, role, context, *args, **kwargs):
                        calls.append(copy.deepcopy(context))
                        if context.get('stage') == 'encoding' and sum(c.get('stage') == 'encoding' for c in calls) == 1 and name.startswith('repair_'):
                            delta = ([{'id': 'stone', 'holder': actor}] if name == 'repair_missing_op'
                                     else [{'op': 'transfer_item', 'id': 'wood', 'holder': actor}])
                            return {'delta': delta}, None
                        return gateway.generate(role, context, *args, **kwargs)
                engine.gateway = Probe()
                result = {'case': name, 'repeat': repeat, 'valid': False}
                try:
                    ruling, call_id = engine.judge(actor, {'kind': 'world_action', 'description': description, 'targets': targets})
                    candidate = k.apply_delta(engine.state, actor, ruling['delta'], config)
                    assert digest(engine.state) == before
                    # A valid representation is necessary; success is not demanded for uncertain tasks.
                    check = True
                    if name in {'gather', 'repair_source_transfer'}:
                        check = candidate['entities']['wood']['mass'] < 10 and any(e.get('holder') == actor and e['id'] not in original['entities'] for e in candidate['entities'].values())
                    if name == 'eat_far':
                        check = candidate['entities'][actor]['needs']['hunger'] == .4 and candidate['entities']['food']['mass'] == 5.
                    if name == 'impossible':
                        check = ruling['outcome'] in {'failed', 'rejected'} and not ruling['delta']
                    if name in {'pickup', 'repair_missing_op'}:
                        check = candidate['entities']['stone'].get('holder') == actor
                    if name in {'eat', 'drink'}:
                        need = 'hunger' if name == 'eat' else 'thirst'
                        check = candidate['entities'][actor]['needs'][need] < .4
                    result.update(valid=bool(check), ruling=ruling, call_id=call_id,
                                  adjudication_attempts=len(calls), repairs=[c['repair_error'] for c in calls if 'repair_error' in c])
                except (ModelError, ValueError, AssertionError) as exc:
                    result['error'] = str(exc)
                results.append(result)
                atomic_json(output, {'complete': False, 'passed': False, 'results': results})
                print(json.dumps({k: v for k, v in result.items() if k not in {'ruling', 'repairs'}}), flush=True)
        report = {'complete': True, 'passed': all(r['valid'] for r in results),
                  'cases': len(results), 'results': results,
                  'note': 'Isolated fixture judgments; repair cases inject a known bad first ruling. Human plausibility review remains necessary.'}
        atomic_json(output, report)
        engine.store.backup(Path(output).with_suffix('.sqlite'))
        return report
    finally:
        engine.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    result = run(Config.load(args.config).validate(), args.work, args.output, args.repeats)
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
