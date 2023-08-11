from JetHadronAnalysis.Analysis import Analysis, Region
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt

from ROOT import TCanvas

def pause_for_input():
    programPause = input("Press the <ENTER> key to continue...")
    
if __name__=="__main__":
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])

    ana_pp.setParticleSelectionForJetHadron(pt.PION)

    corr = ana_pp.getDifferentialCorrelationFunction(True)
    corrCanvas = TCanvas("corrCanvas", "corrCanvas", 800, 600)
    corr.Draw("lego")

    
    norm_me_corr = ana_pp.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW)
    norm_me_corrCanvas = TCanvas("norm_me_corrCanvas", "norm_me_corrCanvas", 800, 600)
    norm_me_corr.Draw("lego")

    acc_corr = ana_pp.getAcceptanceCorrectedDifferentialCorrelationFunction(corr, norm_me_corr)
    acc_corrCanvas = TCanvas("acc_corrCanvas", "acc_corrCanvas", 800, 600)
    acc_corr.Draw("lego")

    acc_corr_dphi = ana_pp.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acc_corr)
    acc_corr_dphiCanvas = TCanvas("acc_corr_dphiCanvas", "acc_corr_dphiCanvas", 800, 600)
    acc_corr_dphi.Draw()

    bg_corr_func = ana_pp.getBackgroundCorrelationFunction(True)
    bg_corr_funcCanvas = TCanvas("bg_corr_funcCanvas", "bg_corr_funcCanvas", 800, 600)
    bg_corr_func.Draw()

    bg_func = ana_pp.getAzimuthalBackgroundFunction(bg_corr_func)
    bg_funcCanvas = TCanvas("bg_funcCanvas", "bg_funcCanvas", 800, 600)
    bg_func.Draw()

    bg_corr = ana_pp.getAcceptanceCorrectedBackgroundSubtractedDifferentialAzimuthalCorrelationFunction(acc_corr_dphi, bg_func)
    bg_corrCanvas = TCanvas("bg_corrCanvas", "bg_corrCanvas", 800, 600)
    bg_corr.Draw()

    

    while input("Press q to quit: ")!="q":
            pass