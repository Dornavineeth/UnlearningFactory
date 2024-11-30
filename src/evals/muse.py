from evals.base import Evaluator


# should we have separate files for each when each has barely any content?
class MUSEEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("MUSE", eval_cfg, **kwargs)
