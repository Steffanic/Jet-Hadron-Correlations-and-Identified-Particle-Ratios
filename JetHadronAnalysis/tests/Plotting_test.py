import pytest
from pytest_lazyfixture import lazy_fixture
from math import pi
from JetHadronAnalysis.Analysis import Analysis, Region
from JetHadronAnalysis.Types import AnalysisType, ParticleType, NormalizationMethod
from JetHadronAnalysis.Plotting import plotArray, plotTH1

@pytest.fixture
def analysis_pp():
    return Analysis(AnalysisType.PP, ["/mnt/d/pp/17p.root"])

@pytest.fixture
def analysis_PbPb():
    return Analysis(AnalysisType.SEMICENTRAL, ["/mnt/d/18q/296510.root"])

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialAzimuthalCorrelationFunctionAndPlot(analysis):
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    mixedEventCorrelationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW)
    acceptanceCorrectedCorrelationFunction = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(correlationFunction, mixedEventCorrelationFunction)
    azimuthalCorrelationFunction = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptanceCorrectedCorrelationFunction)
    plotTH1(histograms=azimuthalCorrelationFunction, data_labels="test", error_bands=None, error_band_labels=None, xtitle="#Delta#eta", ytitle="#Delta#phi", title="test", output_path=f"test_{analysis.analysisType}.png")

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getDifferentialAzimuthalCorrelationFunctionAndPlotWithErrorbands(analysis):
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    mixedEventCorrelationFunction = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(NormalizationMethod.SLIDING_WINDOW)
    acceptanceCorrectedCorrelationFunction = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(correlationFunction, mixedEventCorrelationFunction)
    azimuthalCorrelationFunction = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptanceCorrectedCorrelationFunction)
    error_bands = [500.0, 2000.0]
    error_band_labels = ["test1", "test2"]
    plotTH1(histograms=azimuthalCorrelationFunction, data_labels="test", error_bands=error_bands, error_band_labels=error_band_labels, xtitle="#Delta#eta", ytitle="#Delta#phi", title="test", output_path=f"test_{analysis.analysisType}.png")

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_plotTH2DRasiesAssertionError(analysis):
    correlationFunction = analysis.getDifferentialCorrelationFunction()
    with pytest.raises(AssertionError):
        plotTH1(histograms=correlationFunction, data_labels="test", error_bands=None, error_band_labels=None, xtitle="#Delta#eta", ytitle="#Delta#phi", title="test", output_path="test.png")
