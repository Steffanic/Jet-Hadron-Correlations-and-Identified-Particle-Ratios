import pytest
from math import pi
from JetHadronAnalysis.Analysis import Analysis, Region
from JetHadronAnalysis.Types import AnalysisType, ParticleType, NormalizationMethod


@pytest.fixture
def analysis_pp():
    return Analysis(AnalysisType.PP, ["/mnt/d/pp/17p.root"])

def test_fillSparseFromFile(analysis_pp):
    assert analysis_pp.JetHadron.getNumberOfSparses() == 1
    assert analysis_pp.Trigger.getNumberOfSparses() == 1
    assert analysis_pp.MixedEvent.getNumberOfSparses() == 1

def test_setRegionForSparses(analysis_pp):
    analysis_pp.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    assert analysis_pp.JetHadron.minDeltaPhi == - pi/2
    assert analysis_pp.JetHadron.maxDeltaPhi == pi/2
    assert analysis_pp.JetHadron.minDeltaEta == - 0.6
    assert analysis_pp.JetHadron.maxDeltaEta == 0.6
    assert analysis_pp.MixedEvent.minDeltaPhi == - pi/2
    assert analysis_pp.MixedEvent.maxDeltaPhi == pi/2
    assert analysis_pp.MixedEvent.minDeltaEta == - 0.6
    assert analysis_pp.MixedEvent.maxDeltaEta == 0.6
    
    analysis_pp.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    assert analysis_pp.JetHadron.minDeltaPhi == pi/2
    assert analysis_pp.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis_pp.JetHadron.minDeltaEta == - 1.2
    assert analysis_pp.JetHadron.maxDeltaEta == 1.2
    assert analysis_pp.MixedEvent.minDeltaPhi == pi/2
    assert analysis_pp.MixedEvent.maxDeltaPhi == 3*pi/2
    assert analysis_pp.MixedEvent.minDeltaEta == - 1.2
    assert analysis_pp.MixedEvent.maxDeltaEta == 1.2

    analysis_pp.setRegionForSparses(Region.INCLUSIVE)
    assert analysis_pp.JetHadron.minDeltaPhi == - pi/2
    assert analysis_pp.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis_pp.JetHadron.minDeltaEta == - 1.4
    assert analysis_pp.JetHadron.maxDeltaEta == 1.4
    assert analysis_pp.MixedEvent.minDeltaPhi == - pi/2
    assert analysis_pp.MixedEvent.maxDeltaPhi == 3*pi/2
    assert analysis_pp.MixedEvent.minDeltaEta == - 1.4
    assert analysis_pp.MixedEvent.maxDeltaEta == 1.4

    with pytest.raises(NotImplementedError):
       analysis_pp.setRegionForSparses(Region.BACKGROUND)

def test_setRegionThenSetInclusiveRegion(analysis_pp):
    analysis_pp.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    analysis_pp.setRegionForSparses(Region.INCLUSIVE)
    assert analysis_pp.JetHadron.minDeltaPhi == - pi/2
    assert analysis_pp.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis_pp.JetHadron.minDeltaEta == - 1.4
    assert analysis_pp.JetHadron.maxDeltaEta == 1.4
    assert analysis_pp.MixedEvent.minDeltaPhi == - pi/2
    assert analysis_pp.MixedEvent.maxDeltaPhi == 3*pi/2
    assert analysis_pp.MixedEvent.minDeltaEta == - 1.4
    assert analysis_pp.MixedEvent.maxDeltaEta == 1.4

    

def test_getDifferentialCorrelationFunction(analysis_pp):
    correlationFunction = analysis_pp.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 72

def test_getDifferentialCorrelationFunctionInNearSide(analysis_pp):
    analysis_pp.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis_pp.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 36

def test_getDifferentialCorrelationFunctionInAwaySide(analysis_pp):
    analysis_pp.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis_pp.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 36

def test_getDifferentialCorrelationFunctionForPions(analysis_pp):
    analysis_pp.setParticleSelectionForSparses(ParticleType.PION)
    correlationFunction = analysis_pp.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 72

def test_getNormalizedDifferentialMixedEventCorrelationFunction(analysis_pp):
    analysis_pp.setRegionForSparses(Region.INCLUSIVE)
    correlationFunction = analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW)
    assert correlationFunction.GetNbinsX() == 72

    analysis_pp.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    assert correlationFunction.GetNbinsX() == 36

    analysis_pp.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    assert correlationFunction.GetNbinsX() == 36
    
    analysis_pp.setRegionForSparses(Region.INCLUSIVE)
    correlationFunction = analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 72

    analysis_pp.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 36

    analysis_pp.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 36

def test_getNormalizedDifferentialMixedEventCorrelationFunctionFailsOnWindowSizeTooLarge(analysis_pp):
    analysis_pp.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    with pytest.raises(AssertionError):
        analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi)

    analysis_pp.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    with pytest.raises(AssertionError):
        analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi)

    analysis_pp.setRegionForSparses(Region.INCLUSIVE)
    with pytest.raises(AssertionError):
        analysis_pp.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=2*pi)