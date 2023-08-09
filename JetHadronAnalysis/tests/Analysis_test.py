import pytest
from pytest_lazyfixture import lazy_fixture
from math import pi
from JetHadronAnalysis.Analysis import Analysis, Region
from JetHadronAnalysis.Types import AnalysisType, ParticleType, NormalizationMethod


@pytest.fixture
def analysis_pp():
    return Analysis(AnalysisType.PP, ["/mnt/d/pp/17p.root"])

@pytest.fixture
def analysis_PbPb():
    return Analysis(AnalysisType.SEMICENTRAL, ["/mnt/d/18q/296510.root"])

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_fillSparseFromFile(analysis):
    assert analysis.JetHadron.getNumberOfSparses() == 1
    assert analysis.Trigger.getNumberOfSparses() == 1
    assert analysis.MixedEvent.getNumberOfSparses() == 1

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_setRegionForSparses(analysis):
    analysis.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi == pi/2
    assert analysis.JetHadron.minDeltaEta == - 0.6
    assert analysis.JetHadron.maxDeltaEta == 0.6
    assert analysis.MixedEvent.minDeltaPhi == - pi/2
    assert analysis.MixedEvent.maxDeltaPhi == pi/2
    assert analysis.MixedEvent.minDeltaEta == - 0.6
    assert analysis.MixedEvent.maxDeltaEta == 0.6
    
    analysis.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    assert analysis.JetHadron.minDeltaPhi == pi/2
    assert analysis.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis.JetHadron.minDeltaEta == - 1.2
    assert analysis.JetHadron.maxDeltaEta == 1.2
    assert analysis.MixedEvent.minDeltaPhi == pi/2
    assert analysis.MixedEvent.maxDeltaPhi == 3*pi/2
    assert analysis.MixedEvent.minDeltaEta == - 1.2
    assert analysis.MixedEvent.maxDeltaEta == 1.2

    analysis.setRegionForSparses(Region.INCLUSIVE)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis.JetHadron.minDeltaEta == - 1.4
    assert analysis.JetHadron.maxDeltaEta == 1.4
    assert analysis.MixedEvent.minDeltaPhi == - pi/2
    assert analysis.MixedEvent.maxDeltaPhi == 3*pi/2
    assert analysis.MixedEvent.minDeltaEta == - 1.4
    assert analysis.MixedEvent.maxDeltaEta == 1.4

    with pytest.raises(NotImplementedError):
       analysis.setRegionForSparses(Region.BACKGROUND)

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_setRegionThenSetInclusiveRegion(analysis):
    analysis.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    analysis.setRegionForSparses(Region.INCLUSIVE)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis.JetHadron.minDeltaEta == - 1.4
    assert analysis.JetHadron.maxDeltaEta == 1.4
    assert analysis.MixedEvent.minDeltaPhi == - pi/2
    assert analysis.MixedEvent.maxDeltaPhi == 3*pi/2
    assert analysis.MixedEvent.minDeltaEta == - 1.4
    assert analysis.MixedEvent.maxDeltaEta == 1.4

    

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunction(analysis):
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 72

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunctionInNearSide(analysis):
    analysis.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 36

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunctionInAwaySide(analysis):
    analysis.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 36

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunctionForPions(analysis):
    analysis.setParticleSelectionForSparses(ParticleType.PION)
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 72

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getNormalizedDifferentialMixedEventCorrelationFunction(analysis):
    analysis.setRegionForSparses(Region.INCLUSIVE)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW)
    assert correlationFunction.GetNbinsX() == 72

    analysis.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    assert correlationFunction.GetNbinsX() == 36

    analysis.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    assert correlationFunction.GetNbinsX() == 36
    
    analysis.setRegionForSparses(Region.INCLUSIVE)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 72

    analysis.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 36

    analysis.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 36

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getNormalizedDifferentialMixedEventCorrelationFunctionFailsOnWindowSizeTooLarge(analysis):
    analysis.setRegionForSparses(Region.NEAR_SIDE_SIGNAL)
    with pytest.raises(AssertionError):
        analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi)

    analysis.setRegionForSparses(Region.AWAY_SIDE_SIGNAL)
    with pytest.raises(AssertionError):
        analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi)

    analysis.setRegionForSparses(Region.INCLUSIVE)
    with pytest.raises(AssertionError):
        analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=2*pi)