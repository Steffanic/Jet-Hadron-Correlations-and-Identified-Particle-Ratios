from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt

if __name__=="__main__":
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])
    ana_pp.setAssociatedHadronMomentumBin(AssociatedHadronMomentumBin.PT_1_15)
    ana_pp.setRegion(Region.NEAR_SIDE_SIGNAL)
    print(f"Near-side signal, {ana_pp.getPIDFractions()}")
    ana_pp.setRegion(Region.AWAY_SIDE_SIGNAL)
    print(f"Away-side signal, {ana_pp.getPIDFractions()}")
    ana_pp.setAssociatedHadronMomentumBin(AssociatedHadronMomentumBin.PT_15_2)
    ana_pp.setRegion(Region.NEAR_SIDE_SIGNAL)
    print(f"Near-side signal, {ana_pp.getPIDFractions()}")
    ana_pp.setRegion(Region.AWAY_SIDE_SIGNAL)
    print(f"Away-side signal, {ana_pp.getPIDFractions()}")
    
    