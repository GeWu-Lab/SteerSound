"""Stop Recovery after consecutive prefix depths without strict global Val improvement."""
from dataclasses import dataclass
import math
@dataclass
class RecoveryPatience:
    best_score:float
    patience:int=2
    min_delta:float=0.0
    consecutive_no_improvement:int=0
    best_k:int=0
    last_k:int=0
    def __post_init__(self):
        if not math.isfinite(self.best_score):raise ValueError('Initial Val score must be finite')
        if self.patience<1 or self.min_delta<0:raise ValueError('Invalid patience/min_delta')
    def update(self,k,score):
        if k!=self.last_k+1:raise ValueError('Recovery depths must be processed consecutively')
        if not math.isfinite(score):raise ValueError('Val score must be finite')
        improved=score>self.best_score+self.min_delta
        if improved:
            self.best_score=float(score);self.best_k=int(k);self.consecutive_no_improvement=0
        else:self.consecutive_no_improvement+=1
        self.last_k=int(k)
        return improved
    @property
    def should_stop(self):return self.consecutive_no_improvement>=self.patience
    def state_dict(self):
        return dict(best_score=self.best_score,best_k=self.best_k,last_k=self.last_k,patience=self.patience,
                    min_delta=self.min_delta,consecutive_no_improvement=self.consecutive_no_improvement,
                    should_stop=self.should_stop)
