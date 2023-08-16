from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt
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
    analysis.setRegion(Region.BACKGROUND_ETANEG)
    analysis.setAssociatedHadronMomentumBin(AssociatedHadronMomentumBin.PT_1_15)
    particle_fractions, particle_fraction_errors = analysis.getPIDFractions()
    logging.info(analysis.numberOfAssociatedHadronsDictionary)
    logging.info(particle_fractions)
    logging.info(particle_fraction_errors)
    assert len(particle_fractions) == len(particle_fraction_errors)