import pytest

from JetHadronAnalysis.Analysis import Analysis
from JetHadronAnalysis.Types import AnalysisType

def test_fillSparseFromFile():
    analysis = Analysis(AnalysisType.PP, ["/mnt/d/pp/17p.root"])
    assert analysis.JetHadron.getNumberOfSparses() == 1
    assert analysis.Trigger.getNumberOfSparses() == 1
    assert analysis.MixedEvent.getNumberOfSparses() == 1