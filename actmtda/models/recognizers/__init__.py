from .general_recognizer import GeneralRecognizer

from actmtda.utils.registry import RECOGNIZER


def build_recognizer(model_cfg):
	assert model_cfg.TYPE in RECOGNIZER, \
		"cfg.MODEL.TYPE: {} are not registered in registry".format(
			model_cfg.TYPE
		)
	model = RECOGNIZER[model_cfg.TYPE](model_cfg)
	return model

__all__ = [
	'GeneralRecognizer'
]