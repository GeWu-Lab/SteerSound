"""Frozen thresholds and pitch denominator checks, without model downloads."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,math,unittest
# Load the root metric copy so test order cannot select an experiment-local package.
ROOT=Path(__file__).resolve().parents[1]/'metrics/selective_dataset_metrics'
def load(name):
 spec=importlib.util.spec_from_file_location('test_'+name,ROOT/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
rules=load('_rules');aggregation=load('aggregation')

class SelectiveMetrics(unittest.TestCase):
 def test_strict_clap_thresholds(self):
  for task,gate in [('generate',.08),('add',.04)]:
   self.assertFalse(rules.classify_row(SimpleNamespace(task=task,clap_target_global=gate))['metric_prediction'])
   self.assertTrue(rules.classify_row(SimpleNamespace(task=task,clap_target_global=gate+.0001))['metric_prediction'])
 def test_remove_boundary(self):
  row=SimpleNamespace(task='remove',remove_drop_db=6.,reference_preserve_db=-12.,output_preserve_db=-12.,preserve_drift_db=0.)
  self.assertTrue(rules.classify_row(row)['metric_prediction'])
  row.preserve_drift_db=3.;self.assertFalse(rules.classify_row(row)['metric_prediction'])
 def test_pitch_unmeasurable_excluded(self):
  row=SimpleNamespace(task='pitch',output_edit_pitch_valid=False,output_preserve_pitch_valid=True)
  self.assertFalse(rules.classify_row(row)['metric_in_denominator'])
  result=aggregation.summarize_completion([1,0,None,float('nan')])
  self.assertEqual((result['success_n'],result['valid_n'],result['excluded_n']),(1,2,2))
  self.assertEqual(result['success_rate'],.5)
if __name__=='__main__':unittest.main()
