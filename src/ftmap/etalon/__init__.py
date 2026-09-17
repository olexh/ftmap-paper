"""The gold standard: what a source SHOULD have produced, and the scoring of
it.
"""

from ftmap.etalon.document import Etalon, EtalonError, load
from ftmap.etalon.score import Score, score

__all__ = ["Etalon", "EtalonError", "Score", "load", "score"]
