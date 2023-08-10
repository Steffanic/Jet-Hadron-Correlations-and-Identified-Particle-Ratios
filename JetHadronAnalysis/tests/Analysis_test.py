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
def test_setRegion(analysis):
    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi == pi/2
    assert analysis.JetHadron.minDeltaEta == - 0.6
    assert analysis.JetHadron.maxDeltaEta == 0.6
    # assert analysis.MixedEvent.minDeltaPhi == - pi/2
    # assert analysis.MixedEvent.maxDeltaPhi == pi/2
    # assert analysis.MixedEvent.minDeltaEta == - 0.6
    # assert analysis.MixedEvent.maxDeltaEta == 0.6
    
    analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
    assert analysis.JetHadron.minDeltaPhi == pi/2
    assert analysis.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis.JetHadron.minDeltaEta == - 1.2
    assert analysis.JetHadron.maxDeltaEta == 1.2
    # assert analysis.MixedEvent.minDeltaPhi == pi/2
    # assert analysis.MixedEvent.maxDeltaPhi == 3*pi/2
    # assert analysis.MixedEvent.minDeltaEta == - 1.2
    # assert analysis.MixedEvent.maxDeltaEta == 1.2

    analysis.setRegion(Region.INCLUSIVE)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis.JetHadron.minDeltaEta == - 1.4
    assert analysis.JetHadron.maxDeltaEta == 1.4
    # assert analysis.MixedEvent.minDeltaPhi == - pi/2
    # assert analysis.MixedEvent.maxDeltaPhi == 3*pi/2
    # assert analysis.MixedEvent.minDeltaEta == - 1.4
    # assert analysis.MixedEvent.maxDeltaEta == 1.4

    analysis.setRegion(Region.BACKGROUND_ETAPOS)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi ==  pi/2
    assert analysis.JetHadron.minDeltaEta == 0.8
    assert analysis.JetHadron.maxDeltaEta == 1.2
    # assert analysis.MixedEvent.minDeltaPhi == - pi/2
    # assert analysis.MixedEvent.maxDeltaPhi == pi/2
    # assert analysis.MixedEvent.minDeltaEta == 0.8
    # assert analysis.MixedEvent.maxDeltaEta == 1.2

    analysis.setRegion(Region.BACKGROUND_ETANEG)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi ==  pi/2
    assert analysis.JetHadron.minDeltaEta == - 1.2
    assert analysis.JetHadron.maxDeltaEta == - 0.8
    # assert analysis.MixedEvent.minDeltaPhi == - pi/2
    # assert analysis.MixedEvent.maxDeltaPhi == pi/2
    # assert analysis.MixedEvent.minDeltaEta == - 1.2
    # assert analysis.MixedEvent.maxDeltaEta == - 0.8


@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_setRegionThenSetInclusiveRegion(analysis):
    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    analysis.setRegion(Region.INCLUSIVE)
    assert analysis.JetHadron.minDeltaPhi == - pi/2
    assert analysis.JetHadron.maxDeltaPhi == 3 *pi/2
    assert analysis.JetHadron.minDeltaEta == - 1.4
    assert analysis.JetHadron.maxDeltaEta == 1.4
    # assert analysis.MixedEvent.minDeltaPhi == - pi/2
    # assert analysis.MixedEvent.maxDeltaPhi == 3*pi/2
    # assert analysis.MixedEvent.minDeltaEta == - 1.4
    # assert analysis.MixedEvent.maxDeltaEta == 1.4

    

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunction(analysis):
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 72

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunctionInNearSide(analysis):
    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 36

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunctionInAwaySide(analysis):
    analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 36

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialCorrelationFunctionForPions(analysis):
    analysis.setParticleSelectionForJetHadron(ParticleType.PION)
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    assert correlationFunction.GetNbinsX() == 72

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getNormalizedDifferentialMixedEventCorrelationFunction(analysis):
    analysis.setRegion(Region.INCLUSIVE)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW)
    assert correlationFunction.GetNbinsX() == 72

    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    assert correlationFunction.GetNbinsX() == 36

    analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    assert correlationFunction.GetNbinsX() == 36
    
    analysis.setRegion(Region.INCLUSIVE)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 72

    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 36

    analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
    correlationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.MAX)
    assert correlationFunction.GetNbinsX() == 36

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getNormalizedDifferentialMixedEventCorrelationFunctionFailsOnWindowSizeTooLarge(analysis):
    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    with pytest.raises(AssertionError):
        analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi)

    analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
    with pytest.raises(AssertionError):
        analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi)

    analysis.setRegion(Region.INCLUSIVE)
    with pytest.raises(AssertionError):
        analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=2*pi)

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getAcceptanceCorrectedDifferentialCorrelationFunction(analysis: Analysis):
    analysis.setRegion(Region.INCLUSIVE)
    differentialCorrelationFunction = analysis.getDifferentialCorrelationFunction()
    normalizedMixedEventCorrelationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW)
    correlationFunction = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(differentialCorrelationFunction, normalizedMixedEventCorrelationFunction)
    assert correlationFunction.GetNbinsX() == 72

    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    differentialCorrelationFunction = analysis.getDifferentialCorrelationFunction()
    normalizedMixedEventCorrelationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    correlationFunction = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(differentialCorrelationFunction, normalizedMixedEventCorrelationFunction)
    assert correlationFunction.GetNbinsX() == 36

    analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
    differentialCorrelationFunction = analysis.getDifferentialCorrelationFunction()
    normalizedMixedEventCorrelationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW, windowSize=pi/2)
    correlationFunction = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(differentialCorrelationFunction, normalizedMixedEventCorrelationFunction)
    assert correlationFunction.GetNbinsX() == 36


@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_number_of_associated_particles_is_less_than_inclusive(analysis):
    analysis.setRegion(Region.INCLUSIVE)
    analysis.setParticleSelectionForJetHadron(ParticleType.INCLUSIVE)
    number_of_inclusive_particles = analysis.getNumberOfAssociatedParticles()

    analysis.setParticleSelectionForJetHadron(ParticleType.PION)
    number_of_pions = analysis.getNumberOfAssociatedParticles()

    assert number_of_pions < number_of_inclusive_particles

    analysis.setParticleSelectionForJetHadron(ParticleType.KAON)
    number_of_kaons = analysis.getNumberOfAssociatedParticles()

    assert number_of_kaons < number_of_inclusive_particles

    analysis.setParticleSelectionForJetHadron(ParticleType.PROTON)
    number_of_protons = analysis.getNumberOfAssociatedParticles()

    assert number_of_protons < number_of_inclusive_particles
