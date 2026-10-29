"""CPU tests for loss gradients, strict patience and metric denominator rules."""
from pathlib import Path
import sys,unittest,math
import numpy as np
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'recovery/complex'),str(Path(__file__).resolve().parents[1]/'inference')]
from utils.stage_patience import RecoveryPatience
from utils.kstep_objective import detached_rollout_to_k,corrective_velocity_target,SIGMAS
from utils.objective import loss,pack,unpack
from metrics.complex_dataset_metrics.aggregation import tbg_all,summarize_metric

class ScientificContracts(unittest.TestCase):
 def test_stage_patience_uses_global_best(self):
  p=RecoveryPatience(best_score=1.,patience=2)
  for k,score in enumerate([1.1,1.2,1.3,1.3,1.3],1):
   p.update(k,score);self.assertEqual(p.should_stop,k==5)
  self.assertEqual(p.best_k,3)
 def test_patience_resets_on_improvement(self):
  p=RecoveryPatience(1.,2)
  for k,s in enumerate([1.,1.1,1.1,1.2,1.2],1):p.update(k,s)
  self.assertFalse(p.should_stop);self.assertEqual(p.consecutive_no_improvement,1)
 def test_detached_rollout_and_target(self):
  x=torch.ones(2,1,3,2,requires_grad=True);v=torch.nn.Parameter(torch.tensor(.5))
  z,_=detached_rollout_to_k(initial_noise=x,k=3,velocity_fn=lambda x,t:x*0+v)
  self.assertFalse(z.requires_grad)
  self.assertTrue(torch.allclose(z,torch.ones_like(z)-(1-SIGMAS[3])*.5))
  self.assertFalse(corrective_velocity_target(z,x,SIGMAS[3]).requires_grad)
 def test_hybrid_loss_all_tasks_backpropagates(self):
  class Toy(torch.nn.Module):
   def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.4))
   def forward(self,noi_img,**kwargs):return noi_img*self.weight
  p=dict(target=torch.randn(6,2,3,4),noise=torch.randn(6,2,3,4),tasks=torch.arange(6),cond={})
  p.update(noisy=p['noise'],sigma=torch.ones(6));model=Toy()
  value=loss(model,p,3);value.backward()
  self.assertTrue(torch.isfinite(model.weight.grad));self.assertNotEqual(model.weight.grad.item(),0.)
  self.assertTrue(torch.equal(unpack(pack(p['target']),p['target']),p['target']))
 def test_tbg_is_geometric_with_inverted_mae(self):
  tasks=dict(add=2,remove=2,generate=2,replace=2,pitch=.5,volume=.5)
  self.assertAlmostEqual(tbg_all(tasks,{k:1 for k in tasks}),100.)
  self.assertEqual(summarize_metric([1,None,float('nan')])['valid_n'],1)

if __name__=='__main__':unittest.main()
