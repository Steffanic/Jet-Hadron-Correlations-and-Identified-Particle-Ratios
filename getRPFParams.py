from functools import partial
from math import pi
from multiprocessing import Pool
from matplotlib import pyplot as plt

import numpy as np
from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt
from JetHadronAnalysis.Types import ReactionPlaneBin as rpa
from JetHadronAnalysis.Types import TriggerJetMomentumBin as tjmb
from JetHadronAnalysis.Plotting import plotTH1, plotArrays
from JetHadronAnalysis.RPFFit import RPFFit
from JetHadronAnalysis.Fitting.RPF import resolution_parameters


def get_azimuthal_correlation_function(analysis):
    correlation_function = analysis.getDifferentialCorrelationFunction(True)
    mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2, TOF=True)
    acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
    azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)
    return azimuthal_correlation_function

def plot_in_mid_out_and_inclusive_with_fit(x, y, yerr, optimal_params, covariance, reduced_chi2, fit_function, analysis):
    plt.figure(figsize=(20, 5))
    for i, rpa_bin in enumerate(rpa):
        plt.subplot(1, 4, i+1)
        plt.errorbar(x, y[i], yerr[i], label=f"{rpa_bin.name}", fmt=".")
        if i<3:
            plt.plot(x, fit_function(None, *resolution_parameters[analysis.analysisType].values(), x, *optimal_params)[i*len(x):(i+1)*len(x)], "b--", label=f"fit {rpa_bin.name}")
        else:
            plt.plot(x, np.sum([fit_function(None, *resolution_parameters[analysis.analysisType].values(), x, *optimal_params)[rp_bin*len(x):(rp_bin+1)*len(x)] for rp_bin in range(3)], axis=0), label=f"fit {rpa_bin.name}")
        plt.title(f"{rpa_bin.name}")
        plt.legend()
    plt.suptitle(f"{analysis.analysisType} {analysis.currentTriggerJetMomentumBin} {analysis.currentAssociatedHadronMomentumBin} {analysis.current_species} reduced chi2: {reduced_chi2}")
    plt.savefig(f"RPFFits/RPF_fit_{analysis.analysisType}_{analysis.currentTriggerJetMomentumBin}_{analysis.currentAssociatedHadronMomentumBin}_{analysis.current_species}.png")

def fit_and_plot(analysis, species):
    region = Region.BACKGROUND
    for trig_bin in tjmb:
        for assoc_bin in AssociatedHadronMomentumBin:
            print(f"fitting {analysis.analysisType} {trig_bin} {assoc_bin} {species}")
            analysis.setTriggerJetMomentumBin(trig_bin)
            analysis.setAssociatedHadronMomentumBin(assoc_bin)
            analysis.setRegion(region)
            analysis.setParticleSelectionForJetHadron(species)
            
            # first get the in- mid- and out-of-plane correlation functions
            analysis.setReactionPlaneAngleBin(rpa.INCLUSIVE)
            inclusive_azimuthal_correlation_function = get_azimuthal_correlation_function(analysis)
            analysis.setReactionPlaneAngleBin(rpa.IN_PLANE)
            inplane_azimuthal_correlation_function = get_azimuthal_correlation_function(analysis)
            analysis.setReactionPlaneAngleBin(rpa.MID_PLANE)
            midplane_azimuthal_correlation_function = get_azimuthal_correlation_function(analysis)
            analysis.setReactionPlaneAngleBin(rpa.OUT_PLANE)
            outplane_azimuthal_correlation_function = get_azimuthal_correlation_function(analysis)

            # now lets make the RPF fit object
            rpf_fitter = RPFFit(analysis.analysisType, analysis.currentTriggerJetMomentumBin, analysis.currentAssociatedHadronMomentumBin, analysis.current_species)

            rpf_fitter.setDefaultParameters()

            x, y, yerr = RPFFit.prepareData(inplane_azimuthal_correlation_function, midplane_azimuthal_correlation_function, outplane_azimuthal_correlation_function)

            optimal_params, covariance, reduced_chi2 = rpf_fitter.performFit(x, y, yerr)

            #augment y and yerr with the bincontentsa and binerrors of the inclusive correlation function
            y.append(np.array([inclusive_azimuthal_correlation_function.GetBinContent(i) for i in range(1, inclusive_azimuthal_correlation_function.GetNbinsX()+1)]))
            yerr.append(np.array([inclusive_azimuthal_correlation_function.GetBinError(i) for i in range(1, inclusive_azimuthal_correlation_function.GetNbinsX()+1)]))

            plot_in_mid_out_and_inclusive_with_fit(x, y, yerr, optimal_params, covariance, reduced_chi2, rpf_fitter.fittingFunction, analysis)


if __name__=="__main__":
    ana_SEMICENTRAL = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/296510.root", "/mnt/d/18q/296191.root", "/mnt/d/18q/296379.root", "/mnt/d/18q/296551.root", "/mnt/d/18q/296550.root", "/mnt/d/18q/296472.root", "/mnt/d/18q/296433.root", "/mnt/d/18q/296423.root", "/mnt/d/18q/296377.root", "/mnt/d/18q/296133.root", "/mnt/d/18q/296068.root", "/mnt/d/18q/296065.root", "/mnt/d/18q/295754.root", "/mnt/d/18q/295673.root", "/mnt/d/18r/297129.root", "/mnt/d/18r/297372.root", "/mnt/d/18r/297415.root", "/mnt/d/18r/297441.root", "/mnt/d/18r/297446.root", "/mnt/d/18r/297479.root", "/mnt/d/18r/297544.root", "/mnt/d/18r/296690.root", "/mnt/d/18r/296794.root", "/mnt/d/18r/296894.root", "/mnt/d/18r/296941.root", "/mnt/d/18r/297031.root", "/mnt/d/18r/297085.root", "/mnt/d/18r/297118.root"])
    analysis = ana_SEMICENTRAL
    region = Region.BACKGROUND
    for species in [pt.PROTON, pt.INCLUSIVE]:
        fit_and_plot(analysis, species)
                


