"""Internal PaSST model factory, adapted from hear21passt 0.0.26.

Only the inference dependencies used by KL are included. Original attribution
in the model and helper modules is retained. See the package readme.
"""
from .model import get_model as get_model_passt
from .preprocess import AugmentMelSTFT
from .wrapper import PasstBasicWrapper


def get_basic_model(**kwargs):
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)
    net = get_model_passt(arch=kwargs.get("arch", "passt_s_swa_p16_128_ap476"))
    model = PasstBasicWrapper(mel=mel, net=net, **kwargs)
    return model
