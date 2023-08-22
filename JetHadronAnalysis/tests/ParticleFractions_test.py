from math import pi
from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt
from JetHadronAnalysis.Plotting import plotTH1
import pytest
from pytest_lazyfixture import lazy_fixture
import logging

@pytest.fixture
def analysis_pp():
    return Analysis(AnalysisType.PP, ["/mnt/d/pp/17p.root"])

@pytest.fixture
def analysis_PbPb():
    return Analysis(AnalysisType.SEMICENTRAL, ["/mnt/d/18q/296510.root"])

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getParticleFractionsRuns(analysis):
    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    analysis.setAssociatedHadronMomentumBin(AssociatedHadronMomentumBin.PT_1_15)
    particle_fractions, particle_fraction_errors, chi2OverNDF = analysis.getPIDFractions(makeIntermediatePlots=True)
    logging.info(analysis.numberOfAssociatedHadronsDictionary)
    logging.info(particle_fractions)
    logging.info(particle_fraction_errors)
    assert len(particle_fractions) == len(particle_fraction_errors)

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getParticleFractionsSubRegions(analysis):
    analysis.JetHadron.setDeltaEtaRange(-0.6, -0.2)
    analysis.JetHadron.setDeltaPhiRange(-0.6, 0.6)
    analysis.setAssociatedHadronMomentumBin(AssociatedHadronMomentumBin.PT_15_2)
    particle_fractions, particle_fraction_errors, chi2OverNDF = analysis.getPIDFractions(makeIntermediatePlots=True)
    logging.info("Negative Eta")
    logging.info(particle_fractions)
    logging.info(particle_fraction_errors)
    logging.info(chi2OverNDF)

    analysis.JetHadron.setDeltaEtaRange(0.2, 0.6)
    analysis.JetHadron.setDeltaPhiRange(-0.6, 0.6)
    analysis.setAssociatedHadronMomentumBin(AssociatedHadronMomentumBin.PT_15_2)
    particle_fractions, particle_fraction_errors, chi2OverNDF = analysis.getPIDFractions(makeIntermediatePlots=True)
    logging.info("Positive Eta")
    logging.info(particle_fractions)
    logging.info(particle_fraction_errors)
    logging.info(chi2OverNDF)
    assert len(particle_fractions) == len(particle_fraction_errors)

@pytest.mark.parametrize("analysis", [lazy_fixture("analysis_pp"), lazy_fixture("analysis_PbPb")])
def test_getPerSpeciesAzimuthalCorrelationFunctions(analysis):
    analysis.setAssociatedHadronMomentumBin(AssociatedHadronMomentumBin.PT_2_3)
    # first get the particle fractions for each region 
    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    near_side_particle_fractions, near_side_particle_fraction_errors, chi2OverNDF = analysis.getPIDFractions(makeIntermediatePlots=True)
    logging.info(f"{near_side_particle_fractions=}")
    logging.info(f"{near_side_particle_fraction_errors=}")

    analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
    away_side_particle_fractions, away_side_particle_fraction_errors, chi2OverNDF = analysis.getPIDFractions(makeIntermediatePlots=True)
    logging.info(f"{away_side_particle_fractions=}")
    logging.info(f"{away_side_particle_fraction_errors=}")

    analysis.setRegion(Region.BACKGROUND_ETANEG)
    background_eta_neg_particle_fractions, background_eta_neg_particle_fraction_errors, chi2OverNDF = analysis.getPIDFractions(makeIntermediatePlots=True)
    logging.info(f"{background_eta_neg_particle_fractions=}")
    logging.info(f"{background_eta_neg_particle_fraction_errors=}")

    analysis.setRegion(Region.BACKGROUND_ETAPOS)
    background_eta_pos_particle_fractions, background_eta_pos_particle_fraction_errors, chi2OverNDF = analysis.getPIDFractions(makeIntermediatePlots=True)
    logging.info(f"{background_eta_pos_particle_fractions=}")
    logging.info(f"{background_eta_pos_particle_fraction_errors=}")

    # now get the azimuthal correlation functions for each region
    analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
    near_side_correlation_function = analysis.getDifferentialCorrelationFunction()
    near_side_mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
    near_side_acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(near_side_correlation_function, near_side_mixed_event_correlation_function)
    near_side_azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(near_side_acceptance_corrected_correlation_function)
    # now get the per species azimuthal correlation functions for each region by scaling
    near_side_pion_azimuthal_correlation_function = near_side_azimuthal_correlation_function.Clone()
    near_side_pion_azimuthal_correlation_function.Scale(near_side_particle_fractions[0])
    near_side_kaon_azimuthal_correlation_function = near_side_azimuthal_correlation_function.Clone()
    near_side_kaon_azimuthal_correlation_function.Scale(near_side_particle_fractions[2])
    near_side_proton_azimuthal_correlation_function = near_side_azimuthal_correlation_function.Clone()
    near_side_proton_azimuthal_correlation_function.Scale(near_side_particle_fractions[1])
    plotTH1(histograms=[near_side_azimuthal_correlation_function,near_side_pion_azimuthal_correlation_function, near_side_kaon_azimuthal_correlation_function, near_side_proton_azimuthal_correlation_function], data_labels=["inclusive", "#pi", "K", "p"], error_bands=[0, *[err_fraction*near_side_azimuthal_correlation_function.GetMaximum() for err_fraction in near_side_particle_fraction_errors]], error_band_labels=["inclusive", "#pi", "K", "p"], xtitle="#Delta#eta", ytitle="#Delta#phi", title="test", output_path=f"test_{analysis.analysisType}_near_side.png")
