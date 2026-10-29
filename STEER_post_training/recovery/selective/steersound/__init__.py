"""SteerSound audio editing with plain-language instructions."""
from .instructions import Instruction, parse_instruction
__all__=['SteerSoundPipeline','Instruction','parse_instruction']
def __getattr__(name):
    if name=='SteerSoundPipeline':
        from .pipeline import SteerSoundPipeline
        return SteerSoundPipeline
    raise AttributeError(name)
