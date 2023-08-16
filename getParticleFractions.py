from JetHadronAnalysis.Analysis import Analysis, Region
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt

if __name__=="__main__":
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])

    print(ana_pp.getPIDFractions())
    