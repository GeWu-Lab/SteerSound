"""Parse supported editing instructions; keep model control tokens internal."""
from __future__ import annotations
from dataclasses import dataclass
import math
import re

MODEL_PREFIX='<|im_start|>Reference audio: <|audio_bos|><|AUDIO|><|audio_eos|>'
TASK_IDS={'add':0,'remove':1,'generate':2,'replace':3,'volume':4,'pitch':5}
EXAMPLES={
 'generate':'Generate the sound of rain.',
 'add':'Add the sound of a dog barking.',
 'remove':'Remove the sound of a dog barking.',
 'replace':'Replace the sound of a dog barking by the sound of a cat meowing.',
 'volume':'Increase the volume of the sound of a dog barking by 3.0 dB.',
 'pitch':'Reduce the pitch of the sound of a dog barking by 2.0 semitones.',
}

def strip_model_prefix(text):
    if not isinstance(text,str):raise TypeError('instruction must be a string')
    value=text.strip()
    while value.startswith(MODEL_PREFIX):value=value[len(MODEL_PREFIX):].lstrip()
    if any(token in value for token in ('<|im_start|>','<|im_end|>','<|audio_bos|>','<|audio_eos|>','<|AUDIO|>')):
        raise ValueError('Unexpected embedded model-control token in instruction')
    return value

def model_prompt(text):
    """Preserve empty CFG text; add the checkpoint's prefix exactly once."""
    value=strip_model_prefix(text)
    return MODEL_PREFIX+' '+value if value else ''

@dataclass(frozen=True)
class Instruction:
    text:str
    task:str
    task_id:int
    change_vol:float=0.0
    change_pitch:float=0.0
    start_time:float|None=None
    end_time:float|None=None

    @property
    def conditioning_text(self):
        """Use the unit-free, one-decimal template seen during training."""
        if self.task not in ('volume','pitch'):return self.text
        amount=abs(self.change_vol if self.task=='volume' else self.change_pitch)
        return re.sub(r'\bby\s*[+-]?\d+(?:\.\d+)?\s*(?:dB|decibels?|semitones?|octaves?)?',
                      f'by {amount:.1f}',self.text,flags=re.I)


def parse_instruction(text):
    value=strip_model_prefix(text)
    if not value:raise ValueError('An editing instruction is required')
    lower=value.lower()
    if re.search(r'(?:;|\band\b|\bthen\b)\s*(?:then\s+)?(?:(?:add|insert|remove|delete|replace|generate)\s+(?:the\s+)?sound\b|(?:increase|reduce|raise|lower)\s+(?:the\s+)?(?:pitch|volume)\b)',lower):
        raise ValueError('Use one editing operation per instruction')
    task=None
    if re.match(r'^(generate|synthesi[sz]e|create)\b',lower):task='generate'
    elif re.match(r'^(add|insert)\b',lower):task='add'
    elif re.match(r'^(remove|delete|eliminate)\b',lower):task='remove'
    elif re.match(r'^(replace|substitute|swap)\b',lower):task='replace'
    else:
        pitch=bool(re.search(r'\bpitch\b',lower))
        volume=bool(re.search(r'\b(volume|louder|quieter)\b',lower))
        if pitch and volume:raise ValueError('Use one task per instruction')
        if pitch:task='pitch'
        elif volume:task='volume'
    if task is None:raise ValueError('Cannot determine task. Use one of the six documented instruction templates.')
    if not re.search(r'\b(sound|pitch|volume|louder|quieter)\b',lower):
        raise ValueError('Describe the sound and operation explicitly; see the examples')
    if task=='replace' and not re.search(r'\b(by|with|into|for)\b',lower):
        raise ValueError('Replacement requires both the source and destination sound')
    if task in ('pitch','volume'):
        match=re.search(r'\bby\s*([+-]?\d+(?:\.\d+)?)\s*(dB|decibels?|semitones?|octaves?)?(?=[\s.,!?]|$)',value,re.I)
        if match is None:raise ValueError('Pitch/volume instructions require an explicit change after "by"')
        amount=float(match.group(1));unit=(match.group(2) or '').lower()
        if value[match.end():].strip(' .!?'):
            raise ValueError('Unsupported amount/unit; put the change at the end, as in the documented templates')
        if not math.isfinite(amount):raise ValueError('Change amount must be finite')
        positive=bool(re.search(r'\b(increase|raise|higher|up|louder|boost)\b',lower))
        negative=bool(re.search(r'\b(reduce|decrease|lower|down|quieter)\b',lower))
        if positive==negative:raise ValueError('Specify one unambiguous increase or decrease direction')
        if amount<0:raise ValueError('Use a nonnegative magnitude with increase/reduce, not a signed magnitude')
        amount=amount if positive else -amount
        if task=='pitch':
            if unit in ('db','decibel','decibels'):raise ValueError('Pitch changes use semitones, not dB')
            if unit in ('octave','octaves'):amount*=12
            if abs(amount)>4:raise ValueError('The released checkpoint supports pitch changes within ±4 semitones')
            if abs(amount*2-round(amount*2))>1e-6:raise ValueError('Pitch changes must be multiples of 0.5 semitone')
        else:
            if unit in ('semitone','semitones','octave','octaves'):raise ValueError('Volume changes use dB, not pitch units')
            if abs(amount)>5:raise ValueError('The released checkpoint supports volume changes within ±5 dB')
    else:amount=0.0
    start=end=None
    times=re.search(r'\bfrom\s+(\d+(?:\.\d+)?)(?:st|nd|rd|th)?\s*(?:s|seconds?)?\s+to\s+(\d+(?:\.\d+)?)(?:st|nd|rd|th)?\s*(?:s|seconds?)',lower)
    if times:
        start,end=map(float,times.groups())
        if not 0<=start<end<=10:raise ValueError('Time range must satisfy 0 ≤ start < end ≤ 10 seconds')
    return Instruction(value,task,TASK_IDS[task],amount if task=='volume' else 0.0,
                       amount if task=='pitch' else 0.0,start,end)
