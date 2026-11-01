from tempfile import TemporaryDirectory
import unittest

from complexity_ensemble.regional import ExperimentResult
from complexity_ensemble.result_registry import load_run, run_display_name, save_run
from complexity_ensemble.services import result_summary


class RunNameTests(unittest.TestCase):
    def test_names_follow_model_and_training_approach(self):
        for model, approach in [('mlp', 'no_routing'), ('mlp', 'distillation'),
                                ('lstm', 'no_routing'), ('pinnmamba', 'no_routing')]:
            config = {'model': {'complex_expert': model}, 'strategy': {'approach': approach}}
            self.assertEqual(run_display_name(config), f'{model}_{approach}')

    def test_saved_names_and_api_summaries_keep_storage_ids_intact(self):
        config = {'model': {'complex_expert': 'mlp', 'simple_expert': 'rbf'},
                  'strategy': {'approach': 'no_routing', 'epochs': 20}}
        result = ExperimentResult([], {}, {}, {}, {}, [], [], config)
        with TemporaryDirectory() as root:
            run = save_run(result, root)
            restored = load_run(run.path / 'run.json')
            self.assertEqual(restored.run_id, run.run_id)
            self.assertEqual(restored.path.name, run.run_id)
            self.assertEqual(restored.display_name, 'mlp_no_routing')
            self.assertTrue(restored.label.startswith('mlp_no_routing'))
            summary = result_summary(restored)
            self.assertEqual(summary['display_name'], 'mlp_no_routing')
            self.assertEqual(summary['run_id'], run.run_id)


if __name__ == '__main__':
    unittest.main()
