from math import pi
from matplotlib import pyplot as plt

import numpy as np
from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt
from JetHadronAnalysis.Plotting import plotTH1, plotArrays

def getYieldAndError(analysis, particle_type, azimuthal_correlation_function, loadFractionsFromDB=True): # I have to add the yield error band calculation
    # now get the per species azimuthal correlation functions for each region by scaling
    particle_pid_fit_sys_err = None
    if particle_type == pt.INCLUSIVE:
        particle_azimuthal_correlation_function = azimuthal_correlation_function
    else:
        particle_azimuthal_correlation_function, particle_pid_fit_sys_err = analysis.getAzimuthalCorrelationFunctionforParticleType(particle_type, azimuthal_correlation_function, loadFractionsFromDB)
    yield_ = analysis.getYieldFromAzimuthalCorrelationFunction(particle_azimuthal_correlation_function, particle_pid_fit_sys_err)
    return yield_[0], yield_[1], yield_[2]

def getAzimuthalCorrelationFunction(analysis):
    correlation_function = analysis.getDifferentialCorrelationFunction(True)
    mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
    acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
    azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)
    return azimuthal_correlation_function

if __name__=="__main__":

    assoc_pt_bin_centers = [1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 8.0]
    assoc_pt_bin_widths = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 4.0]
    loadFractionsFromDB=True

    # ++++++++++++++++++++++++++++++++++
    # PP 
    # ++++++++++++++++++++++++++++++++++
    print("Loading PP files")
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])
    pp_yields = {}
    pp_yield_errors = {}
    pp_yield_pid_fit_sys_errors = {}

    # make the arrays of yields
    pp_inclusive_yields = {}
    pp_inclusive_yield_errors = {}
    pp_pion_yields = {}
    pp_pion_yield_errors = {}
    pp_pion_yield_pid_fit_sys_errors = {}
    pp_kaon_yields = {}
    pp_kaon_yield_errors = {}
    pp_kaon_yield_pid_fit_sys_errors = {}
    pp_proton_yields = {}
    pp_proton_yield_errors = {}
    pp_proton_yield_pid_fit_sys_errors = {}

    pp_inclusive_bgsub_yields = {}
    pp_inclusive_bgsub_yield_errors = {}
    pp_pion_bgsub_yields = {}
    pp_pion_bgsub_yield_errors = {}
    pp_pion_bgsub_pid_fit_sys_errors = {}
    pp_kaon_bgsub_yields = {}
    pp_kaon_bgsub_yield_errors = {}
    pp_kaon_bgsub_pid_fit_sys_errors = {}
    pp_proton_bgsub_yields = {}
    pp_proton_bgsub_yield_errors = {}
    pp_proton_bgsub_pid_fit_sys_errors = {}

    for region in [Region.BACKGROUND, Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        for assoc_bin in AssociatedHadronMomentumBin:
            print(f"Starting {assoc_bin} in {region} for pp")
            ana_pp.setAssociatedHadronMomentumBin(assoc_bin)
            ana_pp.setRegion(region)
            
            azimuthal_correlation_function = getAzimuthalCorrelationFunction(ana_pp)

            inclusive_yield, inclusive_yield_error, inclusive_pid_fit_err = getYieldAndError(ana_pp, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
            pion_yield, pion_yield_error, pion_pid_fit_err = getYieldAndError(ana_pp, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
            kaon_yield, kaon_yield_error, kaon_pid_fit_err = getYieldAndError(ana_pp, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
            proton_yield, proton_yield_error, proton_pid_fit_err = getYieldAndError(ana_pp, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

            print(f"Yield for {assoc_bin} in {ana_pp.analysisType} is {inclusive_yield}")
            print(f"Pion yield for {assoc_bin} in {ana_pp.analysisType} is {pion_yield}")
            print(f"Kaon yield for {assoc_bin} in {ana_pp.analysisType} is {kaon_yield}")
            print(f"Proton yield for {assoc_bin} in {ana_pp.analysisType} is {proton_yield}")
            pp_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
            pp_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}
            pp_yield_pid_fit_sys_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_pid_fit_err, pt.PION:pion_pid_fit_err, pt.KAON:kaon_pid_fit_err, pt.PROTON:proton_pid_fit_err}


        pp_inclusive_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_inclusive_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_pion_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_pion_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_pion_yield_pid_fit_sys_errors[region] = np.array([pp_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_kaon_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_kaon_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_kaon_yield_pid_fit_sys_errors[region] = np.array([pp_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_proton_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_proton_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        pp_proton_yield_pid_fit_sys_errors[region] = np.array([pp_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            pp_inclusive_bgsub_yields[region] = pp_inclusive_yields[region]-pp_inclusive_yields[Region.BACKGROUND]
            pp_inclusive_bgsub_yield_errors[region] = np.sqrt(pp_inclusive_yield_errors[region]**2+pp_inclusive_yield_errors[Region.BACKGROUND]**2)
            pp_pion_bgsub_yields[region] = pp_pion_yields[region]-pp_pion_yields[Region.BACKGROUND]
            pp_pion_bgsub_yield_errors[region] = np.sqrt(pp_pion_yield_errors[region]**2+pp_pion_yield_errors[Region.BACKGROUND]**2)
            pp_pion_bgsub_pid_fit_sys_errors[region] = np.sqrt(pp_pion_yield_pid_fit_sys_errors[region]**2+pp_pion_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)
            pp_kaon_bgsub_yields[region] = pp_kaon_yields[region]-pp_kaon_yields[Region.BACKGROUND]
            pp_kaon_bgsub_yield_errors[region] = np.sqrt(pp_kaon_yield_errors[region]**2+pp_kaon_yield_errors[Region.BACKGROUND]**2)
            pp_kaon_bgsub_pid_fit_sys_errors[region] = np.sqrt(pp_kaon_yield_pid_fit_sys_errors[region]**2+pp_kaon_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)
            pp_proton_bgsub_yields[region] = pp_proton_yields[region]-pp_proton_yields[Region.BACKGROUND]
            pp_proton_bgsub_yield_errors[region] = np.sqrt(pp_proton_yield_errors[region]**2+pp_proton_yield_errors[Region.BACKGROUND]**2)
            pp_proton_bgsub_pid_fit_sys_errors[region] = np.sqrt(pp_proton_yield_pid_fit_sys_errors[region]**2+pp_proton_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)

        # now plot the arrays
        plt.errorbar(assoc_pt_bin_centers, pp_inclusive_yields[region], pp_inclusive_yield_errors[region], label="pp inclusive", color="black")
        plt.errorbar(assoc_pt_bin_centers, pp_pion_yields[region], pp_pion_yield_errors[region], label="pp pion", color="blue")
        plt.errorbar(assoc_pt_bin_centers, pp_kaon_yields[region], pp_kaon_yield_errors[region], label="pp kaon", color="green")
        plt.errorbar(assoc_pt_bin_centers, pp_proton_yields[region], pp_proton_yield_errors[region], label="pp proton", color="red")


        plt.fill_between(assoc_pt_bin_centers, pp_pion_yields[region]-pp_pion_yield_pid_fit_sys_errors[region], pp_pion_yields[region]+pp_pion_yield_pid_fit_sys_errors[region], alpha=0.2, color="blue")

        plt.fill_between(assoc_pt_bin_centers, pp_kaon_yields[region]-pp_kaon_yield_pid_fit_sys_errors[region], pp_kaon_yields[region]+pp_kaon_yield_pid_fit_sys_errors[region], alpha=0.2, color="green")

        plt.fill_between(assoc_pt_bin_centers, pp_proton_yields[region]-pp_proton_yield_pid_fit_sys_errors[region], pp_proton_yields[region]+pp_proton_yield_pid_fit_sys_errors[region], alpha=0.2, color="red")
        
        plt.title(f"PP {region.name} yields")
        plt.xlabel("Associated hadron pT[GeV/c]")
        plt.ylabel("Yield")
        plt.semilogy()
        plt.xticks(assoc_pt_bin_centers)
        plt.ylim(1e-5, 1e0)
        plt.legend()
        plt.savefig(f"Plots/PP/{region}_yields.png")

        plt.close()

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            # now plot the near_side yields minus the background yields
            plt.errorbar(assoc_pt_bin_centers, pp_inclusive_bgsub_yields[region], pp_inclusive_bgsub_yield_errors[region], label="pp inclusive background subtracted", color="black")
            plt.errorbar(assoc_pt_bin_centers, pp_pion_bgsub_yields[region], pp_pion_bgsub_yield_errors[region], label="pp pion background subtracted", color="blue")
            plt.errorbar(assoc_pt_bin_centers, pp_kaon_bgsub_yields[region], pp_kaon_bgsub_yield_errors[region], label="pp kaon background subtracted", color="green")
            plt.errorbar(assoc_pt_bin_centers, pp_proton_bgsub_yields[region], pp_proton_bgsub_yield_errors[region], label="pp proton", color="red")


            plt.fill_between(assoc_pt_bin_centers, pp_pion_bgsub_yields[region]-pp_pion_bgsub_pid_fit_sys_errors[region], pp_pion_bgsub_yields[region]+pp_pion_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="blue")

            plt.fill_between(assoc_pt_bin_centers, pp_kaon_bgsub_yields[region]-pp_kaon_bgsub_pid_fit_sys_errors[region], pp_kaon_bgsub_yields[region]+pp_kaon_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="green")

            plt.fill_between(assoc_pt_bin_centers, pp_proton_bgsub_yields[region]-pp_proton_bgsub_pid_fit_sys_errors[region], pp_proton_bgsub_yields[region]+pp_proton_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="red")

            plt.title(f"PP {region.name} background subtracted yields")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Yield")
            plt.semilogy()
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(1e-5, 1e0)
            plt.legend()
            plt.savefig(f"Plots/PP/{region}_background_subtracted_yields.png")

            plt.close()

            # now the ratios proton to pion and kaon to pion 
            plt.errorbar(assoc_pt_bin_centers, pp_proton_bgsub_yields[region]/pp_pion_bgsub_yields[region], pp_proton_bgsub_yield_errors[region]/pp_pion_bgsub_yields[region], label="pp proton to pion ratio", color="red")
            plt.fill_between(assoc_pt_bin_centers, pp_proton_bgsub_yields[region]/pp_pion_bgsub_yields[region]-pp_proton_bgsub_yield_errors[region]/pp_pion_bgsub_yields[region], pp_proton_bgsub_yields[region]/pp_pion_bgsub_yields[region]+pp_proton_bgsub_yield_errors[region]/pp_pion_bgsub_yields[region], alpha=0.2, color="red")
            plt.title(f"PP {region.name} proton to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/PP/{region}_proton_to_pion_ratio.png")

            plt.close()

            plt.errorbar(assoc_pt_bin_centers, pp_kaon_bgsub_yields[region]/pp_pion_bgsub_yields[region], pp_kaon_bgsub_yield_errors[region]/pp_pion_bgsub_yields[region], label="pp kaon to pion ratio", color="green")
            plt.fill_between(assoc_pt_bin_centers, pp_kaon_bgsub_yields[region]/pp_pion_bgsub_yields[region]-pp_kaon_bgsub_yield_errors[region]/pp_pion_bgsub_yields[region], pp_kaon_bgsub_yields[region]/pp_pion_bgsub_yields[region]+pp_kaon_bgsub_yield_errors[region]/pp_pion_bgsub_yields[region], alpha=0.2, color="green")

            plt.title(f"PP {region.name} kaon to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/PP/{region}_kaon_to_pion_ratio.png")

            plt.close()

            
    # +++++++++++++++++++++++
    # SemiCentral
    # +++++++++++++++++++++++
            
    ana_semicentral = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root",])
    semicentral_yields = {}
    semicentral_yield_errors = {}
    semicentral_yield_pid_fit_sys_errors = {}
    semicentral_inclusive_yields = {}
    semicentral_inclusive_yield_errors = {}
    semicentral_pion_yields = {}
    semicentral_pion_yield_errors = {}
    semicentral_pion_yield_pid_fit_sys_errors = {}
    semicentral_kaon_yields = {}
    semicentral_kaon_yield_errors = {}
    semicentral_kaon_yield_pid_fit_sys_errors = {}
    semicentral_proton_yields = {}
    semicentral_proton_yield_errors = {}
    semicentral_proton_yield_pid_fit_sys_errors = {}

    semicentral_inclusive_bgsub_yields = {}
    semicentral_inclusive_bgsub_yield_errors = {}
    semicentral_pion_bgsub_yields = {}
    semicentral_pion_bgsub_yield_errors = {}
    semicentral_pion_bgsub_pid_fit_sys_errors = {}
    semicentral_kaon_bgsub_yields = {}
    semicentral_kaon_bgsub_yield_errors = {}
    semicentral_kaon_bgsub_pid_fit_sys_errors = {}
    semicentral_proton_bgsub_yields = {}
    semicentral_proton_bgsub_yield_errors = {}
    semicentral_proton_bgsub_pid_fit_sys_errors = {}

    for region in [Region.BACKGROUND, Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        for assoc_bin in AssociatedHadronMomentumBin:
            print(f"Starting {assoc_bin} in {region} for semicentral")
            ana_semicentral.setAssociatedHadronMomentumBin(assoc_bin)
            ana_semicentral.setRegion(region)
            
            azimuthal_correlation_function = getAzimuthalCorrelationFunction(ana_semicentral)

            inclusive_yield, inclusive_yield_error, inclusive_pid_fit_err = getYieldAndError(ana_semicentral, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
            pion_yield, pion_yield_error, pion_pid_fit_err = getYieldAndError(ana_semicentral, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
            kaon_yield, kaon_yield_error, kaon_pid_fit_err = getYieldAndError(ana_semicentral, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
            proton_yield, proton_yield_error, proton_pid_fit_err = getYieldAndError(ana_semicentral, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

            print(f"Yield for {assoc_bin} in {ana_semicentral.analysisType} is {inclusive_yield}")
            print(f"Pion yield for {assoc_bin} in {ana_semicentral.analysisType} is {pion_yield}")
            print(f"Kaon yield for {assoc_bin} in {ana_semicentral.analysisType} is {kaon_yield}")
            print(f"Proton yield for {assoc_bin} in {ana_semicentral.analysisType} is {proton_yield}")
            semicentral_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
            semicentral_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}
            semicentral_yield_pid_fit_sys_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_pid_fit_err, pt.PION:pion_pid_fit_err, pt.KAON:kaon_pid_fit_err, pt.PROTON:proton_pid_fit_err}
            # i have to add the same plots that I added from the pp analysis
        semicentral_inclusive_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_inclusive_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_pion_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_pion_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_pion_yield_pid_fit_sys_errors[region] = np.array([semicentral_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_kaon_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_kaon_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_kaon_yield_pid_fit_sys_errors[region] = np.array([semicentral_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_proton_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_proton_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        semicentral_proton_yield_pid_fit_sys_errors[region] = np.array([semicentral_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            semicentral_inclusive_bgsub_yields[region] = semicentral_inclusive_yields[region]-semicentral_inclusive_yields[Region.BACKGROUND]
            semicentral_inclusive_bgsub_yield_errors[region] = np.sqrt(semicentral_inclusive_yield_errors[region]**2+semicentral_inclusive_yield_errors[Region.BACKGROUND]**2)
            semicentral_pion_bgsub_yields[region] = semicentral_pion_yields[region]-semicentral_pion_yields[Region.BACKGROUND]
            semicentral_pion_bgsub_yield_errors[region] = np.sqrt(semicentral_pion_yield_errors[region]**2+semicentral_pion_yield_errors[Region.BACKGROUND]**2)
            semicentral_pion_bgsub_pid_fit_sys_errors[region] = np.sqrt(semicentral_pion_yield_pid_fit_sys_errors[region]**2+semicentral_pion_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)
            semicentral_kaon_bgsub_yields[region] = semicentral_kaon_yields[region]-semicentral_kaon_yields[Region.BACKGROUND]
            semicentral_kaon_bgsub_yield_errors[region] = np.sqrt(semicentral_kaon_yield_errors[region]**2+semicentral_kaon_yield_errors[Region.BACKGROUND]**2)
            semicentral_kaon_bgsub_pid_fit_sys_errors[region] = np.sqrt(semicentral_kaon_yield_pid_fit_sys_errors[region]**2+semicentral_kaon_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)
            semicentral_proton_bgsub_yields[region] = semicentral_proton_yields[region]-semicentral_proton_yields[Region.BACKGROUND]
            semicentral_proton_bgsub_yield_errors[region] = np.sqrt(semicentral_proton_yield_errors[region]**2+semicentral_proton_yield_errors[Region.BACKGROUND]**2)
            semicentral_proton_bgsub_pid_fit_sys_errors[region] = np.sqrt(semicentral_proton_yield_pid_fit_sys_errors[region]**2+semicentral_proton_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)

        
        # now plot the arrays
        plt.errorbar(assoc_pt_bin_centers, semicentral_inclusive_yields[region], semicentral_inclusive_yield_errors[region], label="semicentral inclusive", color="black")
        plt.errorbar(assoc_pt_bin_centers, semicentral_pion_yields[region], semicentral_pion_yield_errors[region], label="semicentral pion", color="blue")
        plt.errorbar(assoc_pt_bin_centers, semicentral_kaon_yields[region], semicentral_kaon_yield_errors[region], label="semicentral kaon", color="green")
        plt.errorbar(assoc_pt_bin_centers, semicentral_proton_yields[region], semicentral_proton_yield_errors[region], label="semicentral proton", color="red")


        plt.fill_between(assoc_pt_bin_centers, semicentral_pion_yields[region]-semicentral_pion_yield_pid_fit_sys_errors[region], semicentral_pion_yields[region]+semicentral_pion_yield_pid_fit_sys_errors[region], alpha=0.2, color="blue")

        plt.fill_between(assoc_pt_bin_centers, semicentral_kaon_yields[region]-semicentral_kaon_yield_pid_fit_sys_errors[region], semicentral_kaon_yields[region]+semicentral_kaon_yield_pid_fit_sys_errors[region], alpha=0.2, color="green")

        plt.fill_between(assoc_pt_bin_centers, semicentral_proton_yields[region]-semicentral_proton_yield_pid_fit_sys_errors[region], semicentral_proton_yields[region]+semicentral_proton_yield_pid_fit_sys_errors[region], alpha=0.2, color="red")

        plt.title(f"Semicentral {region.name} yields")
        plt.xlabel("Associated hadron pT[GeV/c]")
        plt.ylabel("Yield")
        plt.xticks(assoc_pt_bin_centers)
        plt.semilogy()
        plt.ylim(1e-5, 1e0)
        plt.legend()
        plt.savefig(f"Plots/SEMICENTRAL/{region}_yields.png")

        plt.close()

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            plt.errorbar(assoc_pt_bin_centers, semicentral_inclusive_bgsub_yields[region], semicentral_inclusive_bgsub_yield_errors[region], label="semicentral inclusive background subtracted", color="black")
            plt.errorbar(assoc_pt_bin_centers, semicentral_pion_bgsub_yields[region], semicentral_pion_bgsub_yield_errors[region], label="semicentral pion background subtracted", color="blue")
            plt.errorbar(assoc_pt_bin_centers, semicentral_kaon_bgsub_yields[region], semicentral_kaon_bgsub_yield_errors[region], label="semicentral kaon background subtracted", color="green")
            plt.errorbar(assoc_pt_bin_centers, semicentral_proton_bgsub_yields[region], semicentral_proton_bgsub_yield_errors[region], label="semicentral proton background subtracted", color="red")


            plt.fill_between(assoc_pt_bin_centers, semicentral_pion_bgsub_yields[region]-semicentral_pion_bgsub_pid_fit_sys_errors[region], semicentral_pion_bgsub_yields[region]+semicentral_pion_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="blue")

            plt.fill_between(assoc_pt_bin_centers, semicentral_kaon_bgsub_yields[region]-semicentral_kaon_bgsub_pid_fit_sys_errors[region], semicentral_kaon_bgsub_yields[region]+semicentral_kaon_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="green")

            plt.fill_between(assoc_pt_bin_centers, semicentral_proton_bgsub_yields[region]-semicentral_proton_bgsub_pid_fit_sys_errors[region], semicentral_proton_bgsub_yields[region]+semicentral_proton_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="red")

            plt.title(f"Semicentral {region.name} background subtracted yields")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Yield")
            plt.xticks(assoc_pt_bin_centers)
            plt.semilogy()
            plt.ylim(1e-5, 1e0)
            plt.legend()
            plt.savefig(f"Plots/SEMICENTRAL/{region}_background_subtracted_yields.png")

            plt.close()

            # now the ratios proton to pion and kaon to pion
            plt.errorbar(assoc_pt_bin_centers, semicentral_proton_bgsub_yields[region]/semicentral_pion_bgsub_yields[region], semicentral_proton_bgsub_yield_errors[region]/semicentral_pion_bgsub_yields[region], label="semicentral proton to pion ratio", color="red")
            sys_err = semicentral_proton_bgsub_yield_errors[region]/semicentral_pion_bgsub_yields[region]+semicentral_pion_bgsub_yield_errors[region]*semicentral_proton_bgsub_yields[region]/semicentral_pion_bgsub_yields[region]**2
            plt.fill_between(assoc_pt_bin_centers, semicentral_proton_bgsub_yields[region]/semicentral_pion_bgsub_yields[region]-sys_err, semicentral_proton_bgsub_yields[region]/semicentral_pion_bgsub_yields[region]+sys_err, alpha=0.2, color="red")

            plt.title(f"Semicentral {region.name} proton to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/SEMICENTRAL/{region}_proton_to_pion_ratio.png")

            plt.close()

            plt.errorbar(assoc_pt_bin_centers, semicentral_kaon_bgsub_yields[region]/semicentral_pion_bgsub_yields[region], semicentral_kaon_bgsub_yield_errors[region]/semicentral_pion_bgsub_yields[region], label="semicentral kaon to pion ratio", color="green")
            sys_err = semicentral_kaon_bgsub_yield_errors[region]/semicentral_pion_bgsub_yields[region]+semicentral_pion_bgsub_yield_errors[region]*semicentral_kaon_bgsub_yields[region]/semicentral_pion_bgsub_yields[region]**2
            plt.fill_between(assoc_pt_bin_centers, semicentral_kaon_bgsub_yields[region]/semicentral_pion_bgsub_yields[region]-sys_err, semicentral_kaon_bgsub_yields[region]/semicentral_pion_bgsub_yields[region]+sys_err, alpha=0.2, color="green")

            plt.title(f"Semicentral {region.name} kaon to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/SEMICENTRAL/{region}_kaon_to_pion_ratio.png")

            plt.close()
    
    # +++++++++++++++++++++++
    # Central
    # +++++++++++++++++++++++
            

    ana_central = Analysis(at.CENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root",])
    central_yields = {}
    central_yield_errors = {}
    central_yield_pid_fit_sys_errors = {}
    central_inclusive_yields = {}
    central_inclusive_yield_errors = {}
    central_pion_yields = {}
    central_pion_yield_errors = {}
    central_pion_yield_pid_fit_sys_errors = {}
    central_kaon_yields = {}
    central_kaon_yield_errors = {}
    central_kaon_yield_pid_fit_sys_errors = {}
    central_proton_yields = {}
    central_proton_yield_errors = {}
    central_proton_yield_pid_fit_sys_errors = {}

    central_inclusive_bgsub_yields = {}
    central_inclusive_bgsub_yield_errors = {}
    central_pion_bgsub_yields = {}
    central_pion_bgsub_yield_errors = {}
    central_pion_bgsub_pid_fit_sys_errors = {}
    central_kaon_bgsub_yields = {}
    central_kaon_bgsub_yield_errors = {}
    central_kaon_bgsub_pid_fit_sys_errors = {}
    central_proton_bgsub_yields = {}
    central_proton_bgsub_yield_errors = {}
    central_proton_bgsub_pid_fit_sys_errors = {}


    for region in [Region.BACKGROUND, Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        for assoc_bin in AssociatedHadronMomentumBin:
            print(f"Starting {assoc_bin} in {region} for central")
            ana_central.setAssociatedHadronMomentumBin(assoc_bin)
            ana_central.setRegion(region)
            
            azimuthal_correlation_function = getAzimuthalCorrelationFunction(ana_central)

            inclusive_yield, inclusive_yield_error, inclusive_pid_fit_err = getYieldAndError(ana_central, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
            pion_yield, pion_yield_error, pion_pid_fit_err = getYieldAndError(ana_central, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
            kaon_yield, kaon_yield_error, kaon_pid_fit_err = getYieldAndError(ana_central, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
            proton_yield, proton_yield_error, proton_pid_fit_err = getYieldAndError(ana_central, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

            print(f"Yield for {assoc_bin} in {ana_central.analysisType} is {inclusive_yield}")
            print(f"Pion yield for {assoc_bin} in {ana_central.analysisType} is {pion_yield}")
            print(f"Kaon yield for {assoc_bin} in {ana_central.analysisType} is {kaon_yield}")
            print(f"Proton yield for {assoc_bin} in {ana_central.analysisType} is {proton_yield}")
            central_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
            central_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}
            central_yield_pid_fit_sys_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_pid_fit_err, pt.PION:pion_pid_fit_err, pt.KAON:kaon_pid_fit_err, pt.PROTON:proton_pid_fit_err}
        central_inclusive_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_inclusive_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_pion_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_pion_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_pion_yield_pid_fit_sys_errors[region] = np.array([central_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_kaon_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_kaon_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_kaon_yield_pid_fit_sys_errors[region] = np.array([central_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_proton_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_proton_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
        central_proton_yield_pid_fit_sys_errors[region] = np.array([central_yield_pid_fit_sys_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            central_inclusive_bgsub_yields[region] = central_inclusive_yields[region]-central_inclusive_yields[Region.BACKGROUND]
            central_inclusive_bgsub_yield_errors[region] = np.sqrt(central_inclusive_yield_errors[region]**2+central_inclusive_yield_errors[Region.BACKGROUND]**2)
            central_pion_bgsub_yields[region] = central_pion_yields[region]-central_pion_yields[Region.BACKGROUND]
            central_pion_bgsub_yield_errors[region] = np.sqrt(central_pion_yield_errors[region]**2+central_pion_yield_errors[Region.BACKGROUND]**2)
            central_pion_bgsub_pid_fit_sys_errors[region] = np.sqrt(central_pion_yield_pid_fit_sys_errors[region]**2+central_pion_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)
            central_kaon_bgsub_yields[region] = central_kaon_yields[region]-central_kaon_yields[Region.BACKGROUND]
            central_kaon_bgsub_yield_errors[region] = np.sqrt(central_kaon_yield_errors[region]**2+central_kaon_yield_errors[Region.BACKGROUND]**2)
            central_kaon_bgsub_pid_fit_sys_errors[region] = np.sqrt(central_kaon_yield_pid_fit_sys_errors[region]**2+central_kaon_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)
            central_proton_bgsub_yields[region] = central_proton_yields[region]-central_proton_yields[Region.BACKGROUND]
            central_proton_bgsub_yield_errors[region] = np.sqrt(central_proton_yield_errors[region]**2+central_proton_yield_errors[Region.BACKGROUND]**2)
            central_proton_bgsub_pid_fit_sys_errors[region] = np.sqrt(central_proton_yield_pid_fit_sys_errors[region]**2+central_proton_yield_pid_fit_sys_errors[Region.BACKGROUND]**2)


        # now plot the arrays
        plt.errorbar(assoc_pt_bin_centers, central_inclusive_yields[region], central_inclusive_yield_errors[region], label="central inclusive", color="black")
        plt.errorbar(assoc_pt_bin_centers, central_pion_yields[region], central_pion_yield_errors[region], label="central pion",    color="blue")  
        plt.errorbar(assoc_pt_bin_centers, central_kaon_yields[region], central_kaon_yield_errors[region], label="central kaon",   color="green")
        plt.errorbar(assoc_pt_bin_centers, central_proton_yields[region], central_proton_yield_errors[region], label="central proton", color="red")


        plt.fill_between(assoc_pt_bin_centers, central_pion_yields[region]-central_pion_yield_pid_fit_sys_errors[region], central_pion_yields[region]+central_pion_yield_pid_fit_sys_errors[region], alpha=0.2, color="blue")

        plt.fill_between(assoc_pt_bin_centers, central_kaon_yields[region]-central_kaon_yield_pid_fit_sys_errors[region], central_kaon_yields[region]+central_kaon_yield_pid_fit_sys_errors[region], alpha=0.2, color="green")

        plt.fill_between(assoc_pt_bin_centers, central_proton_yields[region]-central_proton_yield_pid_fit_sys_errors[region], central_proton_yields[region]+central_proton_yield_pid_fit_sys_errors[region], alpha=0.2, color="red")

        plt.title(f"Central {region.name} yields")
        plt.xlabel("Associated hadron pT[GeV/c]")
        plt.ylabel("Yield")
        plt.xticks(assoc_pt_bin_centers)
        plt.ylim(1e-5, 1e0)
        plt.semilogy()
        plt.legend()
        plt.savefig(f"Plots/CENTRAL/{region}_yields.png")

        plt.close()

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            plt.errorbar(assoc_pt_bin_centers, central_inclusive_yields[region]-central_inclusive_yields[Region.BACKGROUND], np.sqrt(central_inclusive_yield_errors[region]**2+central_inclusive_yield_errors[Region.BACKGROUND]**2), label="central inclusive", color="black")
            plt.errorbar(assoc_pt_bin_centers, central_pion_yields[region]-central_pion_yields[Region.BACKGROUND], np.sqrt(central_pion_yield_errors[region]**2+central_pion_yield_errors[Region.BACKGROUND]**2), label="central pion", color="blue")
            plt.errorbar(assoc_pt_bin_centers, central_kaon_yields[region]-central_kaon_yields[Region.BACKGROUND], np.sqrt(central_kaon_yield_errors[region]**2+central_kaon_yield_errors[Region.BACKGROUND]**2), label="central kaon", color="green")
            plt.errorbar(assoc_pt_bin_centers, central_proton_yields[region]-central_proton_yields[Region.BACKGROUND], np.sqrt(central_proton_yield_errors[region]**2+central_proton_yield_errors[Region.BACKGROUND]**2), label="central proton", color="red")

            
            plt.fill_between(assoc_pt_bin_centers, central_pion_bgsub_yields[region]-central_pion_bgsub_pid_fit_sys_errors[region], central_pion_yields[region]+central_pion_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="blue")

            plt.fill_between(assoc_pt_bin_centers, central_kaon_bgsub_yields[region]-central_kaon_bgsub_pid_fit_sys_errors[region], central_kaon_yields[region]+central_kaon_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="green")

            plt.fill_between(assoc_pt_bin_centers, central_proton_bgsub_yields[region]-central_proton_bgsub_pid_fit_sys_errors[region], central_proton_yields[region]+central_proton_bgsub_pid_fit_sys_errors[region], alpha=0.2, color="red")

            plt.title(f"Central {region.name} background subtracted yields")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Yield")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(1e-5, 1e0)
            plt.semilogy()
            plt.legend()
            plt.savefig(f"Plots/CENTRAL/{region}_background_subtracted_yields.png")

            plt.close()

            # now the ratios proton to pion and kaon to pion
            plt.errorbar(assoc_pt_bin_centers, central_proton_bgsub_yields[region]/central_pion_bgsub_yields[region], central_proton_bgsub_yield_errors[region]/central_pion_bgsub_yields[region], label="central proton to pion ratio", color="red")
            sys_err = central_proton_bgsub_yield_errors[region]/central_pion_bgsub_yields[region]+central_pion_bgsub_yield_errors[region]*central_proton_bgsub_yields[region]/central_pion_bgsub_yields[region]**2
            plt.fill_between(assoc_pt_bin_centers, central_proton_bgsub_yields[region]/central_pion_bgsub_yields[region]-sys_err, central_proton_bgsub_yields[region]/central_pion_bgsub_yields[region]+sys_err, alpha=0.2, color="red")

            plt.title(f"Central {region.name} proton to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/CENTRAL/{region}_proton_to_pion_ratio.png")

            plt.close()

            plt.errorbar(assoc_pt_bin_centers, central_kaon_bgsub_yields[region]/central_pion_bgsub_yields[region], central_kaon_bgsub_yield_errors[region]/central_pion_bgsub_yields[region], label="central kaon to pion ratio", color="green")
            sys_err = central_kaon_bgsub_yield_errors[region]/central_pion_bgsub_yields[region]+central_pion_bgsub_yield_errors[region]*central_kaon_bgsub_yields[region]/central_pion_bgsub_yields[region]**2
            plt.fill_between(assoc_pt_bin_centers, central_kaon_bgsub_yields[region]/central_pion_bgsub_yields[region]-sys_err, central_kaon_bgsub_yields[region]/central_pion_bgsub_yields[region]+sys_err, alpha=0.2, color="green")

            plt.title(f"Central {region.name} kaon to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/CENTRAL/{region}_kaon_to_pion_ratio.png")

            plt.close()