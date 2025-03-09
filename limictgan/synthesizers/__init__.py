"""Synthesizers module."""

from limictgan.synthesizers.ctgan import CTGANSynthesizer
from limictgan.synthesizers.tvae import TVAESynthesizer

__all__ = ('CTGANSynthesizer', 'TVAESynthesizer')


def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
