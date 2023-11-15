from math import pi
from matplotlib import pyplot as plt

import numpy as np
from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt
from JetHadronAnalysis.Plotting import plotTH1, plotArrays

def getYieldAndError(analysis, particle_type, azimuthal_correlation_function, loadFractionsFromDB=True):
    
    # now get the per species azimuthal correlation functions for each region by scaling
    if particle_type == pt.INCLUSIVE:
        particle_azimuthal_correlation_function = azimuthal_correlation_function
    else:
        particle_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(particle_type, azimuthal_correlation_function, loadFractionsFromDB)
    yield_ = analysis.getYieldFromAzimuthalCorrelationFunction(particle_azimuthal_correlation_function)
    return yield_[0], yield_[1]

if __name__=="__main__":
    loadFractionsFromDB=True
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])
    pp_yields = {}
    pp_yield_errors = {}
    ana_SEMICENTRAl = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/296510.root", "/mnt/d/18q/296191.root", "/mnt/d/18q/296379.root", "/mnt/d/18q/296551.root"])
    PbPb_yields = {}
    PbPb_yield_errors = {}
    for analysis in [ana_pp, ana_SEMICENTRAl]:
        for region in [Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL, Region.BACKGROUND]:
            for assoc_bin in AssociatedHadronMomentumBin:
                analysis.setAssociatedHadronMomentumBin(assoc_bin)
                analysis.setRegion(region)
                
                correlation_function = analysis.getDifferentialCorrelationFunction(True)
                mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
                acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
                azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)

                inclusive_yield, inclusive_yield_error = getYieldAndError(analysis, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
                pion_yield, pion_yield_error = getYieldAndError(analysis, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
                kaon_yield, kaon_yield_error = getYieldAndError(analysis, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
                proton_yield, proton_yield_error = getYieldAndError(analysis, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

                print(f"Yield for {assoc_bin} in {analysis.analysisType} is {inclusive_yield}")
                print(f"Pion yield for {assoc_bin} in {analysis.analysisType} is {pion_yield}")
                print(f"Kaon yield for {assoc_bin} in {analysis.analysisType} is {kaon_yield}")
                print(f"Proton yield for {assoc_bin} in {analysis.analysisType} is {proton_yield}")
                if analysis.analysisType == at.PP:
                    pp_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
                    pp_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}
                else:
                    PbPb_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
                    PbPb_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}

    # make the arrays of yields
    pp_inclusive_yields = {}
    pp_inclusive_yield_errors = {}
    pp_pion_yields = {}
    pp_pion_yield_errors = {}
    pp_kaon_yields = {}
    pp_kaon_yield_errors = {}
    pp_proton_yields = {}
    pp_proton_yield_errors = {}

    PbPb_inclusive_yields = {}
    PbPb_inclusive_yield_errors = {}
    PbPb_pion_yields = {}
    PbPb_pion_yield_errors = {}
    PbPb_kaon_yields = {}
    PbPb_kaon_yield_errors = {}
    PbPb_proton_yields = {}
    PbPb_proton_yield_errors = {}


    for region in [Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL, Region.BACKGROUND]:
        pp_inclusive_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        pp_inclusive_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        pp_pion_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        pp_pion_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        pp_kaon_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        pp_kaon_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        pp_proton_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])
        pp_proton_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])

        PbPb_inclusive_yields[region] = np.array([PbPb_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        PbPb_inclusive_yield_errors[region] = np.array([PbPb_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        PbPb_pion_yields[region] = np.array([PbPb_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        PbPb_pion_yield_errors[region] = np.array([PbPb_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        PbPb_kaon_yields[region] = np.array([PbPb_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        PbPb_kaon_yield_errors[region] = np.array([PbPb_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        PbPb_proton_yields[region] = np.array([PbPb_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])
        PbPb_proton_yield_errors[region] = np.array([PbPb_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])

        # now plot the arrays
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_inclusive_yields[region], pp_inclusive_yield_errors[region], label="pp inclusive")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_pion_yields[region], pp_pion_yield_errors[region], label="pp pion")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_kaon_yields[region], pp_kaon_yield_errors[region], label="pp kaon")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_proton_yields[region], pp_proton_yield_errors[region], label="pp proton")
        plt.semilogy()
        plt.legend()
        plt.savefig(f"pp_{region}_yields.png")

        plt.close()

        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_inclusive_yields[region], PbPb_inclusive_yield_errors[region], label="PbPb inclusive")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_pion_yields[region], PbPb_pion_yield_errors[region], label="PbPb pion")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_kaon_yields[region], PbPb_kaon_yield_errors[region], label="PbPb kaon")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_proton_yields[region], PbPb_proton_yield_errors[region], label="PbPb proton")
        plt.semilogy()
        plt.legend()
        plt.savefig(f"PbPb_{region}_yields.png")

        plt.close()

    for region in [Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        # now plot the near_side yields minus the background yields
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_inclusive_yields[region]-pp_inclusive_yields[Region.BACKGROUND], np.sqrt(pp_inclusive_yield_errors[region]**2+pp_inclusive_yield_errors[Region.BACKGROUND]**2), label="pp inclusive")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_pion_yields[region]-pp_pion_yields[Region.BACKGROUND], np.sqrt(pp_pion_yield_errors[region]**2+pp_pion_yield_errors[Region.BACKGROUND]**2), label="pp pion")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_kaon_yields[region]-pp_kaon_yields[Region.BACKGROUND], np.sqrt(pp_kaon_yield_errors[region]**2+pp_kaon_yield_errors[Region.BACKGROUND]**2), label="pp kaon")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], pp_proton_yields[region]-pp_proton_yields[Region.BACKGROUND], np.sqrt(pp_proton_yield_errors[region]**2+pp_proton_yield_errors[Region.BACKGROUND]**2), label="pp proton")
        plt.semilogy()
        plt.legend()
        plt.savefig(f"pp_{region}_yields_minus_background.png")

        plt.close()

        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_inclusive_yields[region]-PbPb_inclusive_yields[Region.BACKGROUND], np.sqrt(PbPb_inclusive_yield_errors[region]**2+PbPb_inclusive_yield_errors[Region.BACKGROUND]**2), label="PbPb inclusive")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_pion_yields[region]-PbPb_pion_yields[Region.BACKGROUND], np.sqrt(PbPb_pion_yield_errors[region]**2+PbPb_pion_yield_errors[Region.BACKGROUND]**2), label="PbPb pion")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_kaon_yields[region]-PbPb_kaon_yields[Region.BACKGROUND], np.sqrt(PbPb_kaon_yield_errors[region]**2+PbPb_kaon_yield_errors[Region.BACKGROUND]**2), label="PbPb kaon")
        plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], PbPb_proton_yields[region]-PbPb_proton_yields[Region.BACKGROUND], np.sqrt(PbPb_proton_yield_errors[region]**2+PbPb_proton_yield_errors[Region.BACKGROUND]**2), label="PbPb proton")
        plt.semilogy()
        plt.legend()
        plt.savefig(f"PbPb_{region}_yields_minus_background.png")

        plt.close()

    

        