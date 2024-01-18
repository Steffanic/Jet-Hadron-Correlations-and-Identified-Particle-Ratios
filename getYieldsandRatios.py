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
    if particle_type == pt.INCLUSIVE:
        particle_azimuthal_correlation_function = azimuthal_correlation_function
    else:
        particle_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(particle_type, azimuthal_correlation_function, loadFractionsFromDB)
    yield_ = analysis.getYieldFromAzimuthalCorrelationFunction(particle_azimuthal_correlation_function)
    return yield_[0], yield_[1]

if __name__=="__main__":

    assoc_pt_bin_centers = [1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 8.0]
    assoc_pt_bin_widths = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 4.0]

    # ++++++++++++++++++++++++++++++++++
    # PP 
    # ++++++++++++++++++++++++++++++++++
    loadFractionsFromDB=True
    print("Loading PP files")
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])
    pp_yields = {}
    pp_yield_errors = {}

    # make the arrays of yields
    pp_inclusive_yields = {}
    pp_inclusive_yield_errors = {}
    pp_pion_yields = {}
    pp_pion_yield_errors = {}
    pp_kaon_yields = {}
    pp_kaon_yield_errors = {}
    pp_proton_yields = {}
    pp_proton_yield_errors = {}

    pp_inclusive_bgsub_yields = {}
    pp_inclusive_bgsub_yield_errors = {}
    pp_pion_bgsub_yields = {}
    pp_pion_bgsub_yield_errors = {}
    pp_kaon_bgsub_yields = {}
    pp_kaon_bgsub_yield_errors = {}
    pp_proton_bgsub_yields = {}
    pp_proton_bgsub_yield_errors = {}

    for region in [Region.BACKGROUND, Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        for assoc_bin in AssociatedHadronMomentumBin:
            print(f"Starting {assoc_bin} in {region} for pp")
            ana_pp.setAssociatedHadronMomentumBin(assoc_bin)
            ana_pp.setRegion(region)
            
            correlation_function = ana_pp.getDifferentialCorrelationFunction(True)
            mixed_event_correlation_function = ana_pp.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
            acceptance_corrected_correlation_function = ana_pp.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
            azimuthal_correlation_function = ana_pp.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)

            inclusive_yield, inclusive_yield_error = getYieldAndError(ana_pp, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
            pion_yield, pion_yield_error = getYieldAndError(ana_pp, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
            kaon_yield, kaon_yield_error = getYieldAndError(ana_pp, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
            proton_yield, proton_yield_error = getYieldAndError(ana_pp, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

            print(f"Yield for {assoc_bin} in {ana_pp.analysisType} is {inclusive_yield}")
            print(f"Pion yield for {assoc_bin} in {ana_pp.analysisType} is {pion_yield}")
            print(f"Kaon yield for {assoc_bin} in {ana_pp.analysisType} is {kaon_yield}")
            print(f"Proton yield for {assoc_bin} in {ana_pp.analysisType} is {proton_yield}")
            pp_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
            pp_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}


        pp_inclusive_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        pp_inclusive_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        pp_pion_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        pp_pion_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        pp_kaon_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        pp_kaon_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        pp_proton_yields[region] = np.array([pp_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])
        pp_proton_yield_errors[region] = np.array([pp_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            pp_inclusive_bgsub_yields[region] = pp_inclusive_yields[region]-pp_inclusive_yields[Region.BACKGROUND]
            pp_inclusive_bgsub_yield_errors[region] = np.sqrt(pp_inclusive_yield_errors[region]**2+pp_inclusive_yield_errors[Region.BACKGROUND]**2)
            pp_pion_bgsub_yields[region] = pp_pion_yields[region]-pp_pion_yields[Region.BACKGROUND]
            pp_pion_bgsub_yield_errors[region] = np.sqrt(pp_pion_yield_errors[region]**2+pp_pion_yield_errors[Region.BACKGROUND]**2)
            pp_kaon_bgsub_yields[region] = pp_kaon_yields[region]-pp_kaon_yields[Region.BACKGROUND]
            pp_kaon_bgsub_yield_errors[region] = np.sqrt(pp_kaon_yield_errors[region]**2+pp_kaon_yield_errors[Region.BACKGROUND]**2)
            pp_proton_bgsub_yields[region] = pp_proton_yields[region]-pp_proton_yields[Region.BACKGROUND]
            pp_proton_bgsub_yield_errors[region] = np.sqrt(pp_proton_yield_errors[region]**2+pp_proton_yield_errors[Region.BACKGROUND]**2)

        # now plot the arrays
        plt.errorbar(assoc_pt_bin_centers, pp_inclusive_yields[region], pp_inclusive_yield_errors[region], label="pp inclusive")
        plt.errorbar(assoc_pt_bin_centers, pp_pion_yields[region], pp_pion_yield_errors[region], label="pp pion")
        plt.errorbar(assoc_pt_bin_centers, pp_kaon_yields[region], pp_kaon_yield_errors[region], label="pp kaon")
        plt.errorbar(assoc_pt_bin_centers, pp_proton_yields[region], pp_proton_yield_errors[region], label="pp proton")
        
        plt.semilogy()
        plt.legend()
        plt.savefig(f"pp_{region}_yields.png")

        plt.close()

        if region in [Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            # now plot the near_side yields minus the background yields
            plt.errorbar(assoc_pt_bin_centers, pp_inclusive_bgsub_yields[region], pp_inclusive_bgsub_yield_errors[region], label="pp inclusive background subtracted")
            plt.errorbar(assoc_pt_bin_centers, pp_pion_bgsub_yields[region], pp_pion_bgsub_yield_errors[region], label="pp pion background subtracted")
            plt.errorbar(assoc_pt_bin_centers, pp_kaon_bgsub_yields[region], pp_kaon_bgsub_yield_errors[region], label="pp kaon background subtracted")
            plt.errorbar(assoc_pt_bin_centers, pp_proton_bgsub_yields[region], pp_proton_bgsub_yield_errors[region], label="pp proton")
            plt.semilogy()
            plt.legend()
            plt.savefig(f"pp_{region}_yields_minus_background.png")

            plt.close()
            
    # +++++++++++++++++++++++
    # SemiCentral
    # +++++++++++++++++++++++
            
    ana_semicentral = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root",])
    semicentral_yields = {}
    semicentral_yield_errors = {}
    semicentral_inclusive_yields = {}
    semicentral_inclusive_yield_errors = {}
    semicentral_pion_yields = {}
    semicentral_pion_yield_errors = {}
    semicentral_kaon_yields = {}
    semicentral_kaon_yield_errors = {}
    semicentral_proton_yields = {}
    semicentral_proton_yield_errors = {}

    for region in [Region.BACKGROUND, Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        for assoc_bin in AssociatedHadronMomentumBin:
            print(f"Starting {assoc_bin} in {region} for semicentral")
            ana_semicentral.setAssociatedHadronMomentumBin(assoc_bin)
            ana_semicentral.setRegion(region)
            
            correlation_function = ana_semicentral.getDifferentialCorrelationFunction(True)
            mixed_event_correlation_function = ana_semicentral.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
            acceptance_corrected_correlation_function = ana_semicentral.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
            azimuthal_correlation_function = ana_semicentral.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)

            inclusive_yield, inclusive_yield_error = getYieldAndError(ana_semicentral, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
            pion_yield, pion_yield_error = getYieldAndError(ana_semicentral, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
            kaon_yield, kaon_yield_error = getYieldAndError(ana_semicentral, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
            proton_yield, proton_yield_error = getYieldAndError(ana_semicentral, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

            print(f"Yield for {assoc_bin} in {ana_semicentral.analysisType} is {inclusive_yield}")
            print(f"Pion yield for {assoc_bin} in {ana_semicentral.analysisType} is {pion_yield}")
            print(f"Kaon yield for {assoc_bin} in {ana_semicentral.analysisType} is {kaon_yield}")
            print(f"Proton yield for {assoc_bin} in {ana_semicentral.analysisType} is {proton_yield}")
            semicentral_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
            semicentral_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}
            # i have to add the same plots that I added from the pp analysis
        semicentral_inclusive_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        semicentral_inclusive_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        semicentral_pion_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        semicentral_pion_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        semicentral_kaon_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        semicentral_kaon_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        semicentral_proton_yields[region] = np.array([semicentral_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])
        semicentral_proton_yield_errors[region] = np.array([semicentral_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])
        
        # now plot the arrays
        plt.errorbar(assoc_pt_bin_centers, semicentral_inclusive_yields[region], semicentral_inclusive_yield_errors[region], label="semicentral inclusive")
        plt.errorbar(assoc_pt_bin_centers, semicentral_pion_yields[region], semicentral_pion_yield_errors[region], label="semicentral pion")
        plt.errorbar(assoc_pt_bin_centers, semicentral_kaon_yields[region], semicentral_kaon_yield_errors[region], label="semicentral kaon")
        plt.errorbar(assoc_pt_bin_centers, semicentral_proton_yields[region], semicentral_proton_yield_errors[region], label="semicentral proton")
        plt.semilogy()
        plt.legend()
        plt.savefig(f"semicentral_{region}_yields.png")

        plt.close()

        if region in [Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], semicentral_inclusive_yields[region]-semicentral_inclusive_yields[Region.BACKGROUND], np.sqrt(semicentral_inclusive_yield_errors[region]**2+semicentral_inclusive_yield_errors[Region.BACKGROUND]**2), label="semicentral inclusive")
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], semicentral_pion_yields[region]-semicentral_pion_yields[Region.BACKGROUND], np.sqrt(semicentral_pion_yield_errors[region]**2+semicentral_pion_yield_errors[Region.BACKGROUND]**2), label="semicentral pion")
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], semicentral_kaon_yields[region]-semicentral_kaon_yields[Region.BACKGROUND], np.sqrt(semicentral_kaon_yield_errors[region]**2+semicentral_kaon_yield_errors[Region.BACKGROUND]**2), label="semicentral kaon")
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], semicentral_proton_yields[region]-semicentral_proton_yields[Region.BACKGROUND], np.sqrt(semicentral_proton_yield_errors[region]**2+semicentral_proton_yield_errors[Region.BACKGROUND]**2), label="semicentral proton")
            plt.semilogy()
            plt.legend()
            plt.savefig(f"semicentral_{region}_yields_minus_background.png")

            plt.close()
    
    # +++++++++++++++++++++++
    # Central
    # +++++++++++++++++++++++
            

    ana_central = Analysis(at.CENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root",])
    central_yields = {}
    central_yield_errors = {}
    central_inclusive_yields = {}
    central_inclusive_yield_errors = {}
    central_pion_yields = {}
    central_pion_yield_errors = {}
    central_kaon_yields = {}
    central_kaon_yield_errors = {}
    central_proton_yields = {}
    central_proton_yield_errors = {}

    for region in [Region.BACKGROUND, Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        for assoc_bin in AssociatedHadronMomentumBin:
            print(f"Starting {assoc_bin} in {region} for central")
            ana_central.setAssociatedHadronMomentumBin(assoc_bin)
            ana_central.setRegion(region)
            
            correlation_function = ana_central.getDifferentialCorrelationFunction(True)
            mixed_event_correlation_function = ana_central.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
            acceptance_corrected_correlation_function = ana_central.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
            azimuthal_correlation_function = ana_central.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)

            inclusive_yield, inclusive_yield_error = getYieldAndError(ana_central, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
            pion_yield, pion_yield_error = getYieldAndError(ana_central, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
            kaon_yield, kaon_yield_error = getYieldAndError(ana_central, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
            proton_yield, proton_yield_error = getYieldAndError(ana_central, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

            print(f"Yield for {assoc_bin} in {ana_central.analysisType} is {inclusive_yield}")
            print(f"Pion yield for {assoc_bin} in {ana_central.analysisType} is {pion_yield}")
            print(f"Kaon yield for {assoc_bin} in {ana_central.analysisType} is {kaon_yield}")
            print(f"Proton yield for {assoc_bin} in {ana_central.analysisType} is {proton_yield}")
            central_yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
            central_yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}
        central_inclusive_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        central_inclusive_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])
        central_pion_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        central_pion_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])
        central_kaon_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        central_kaon_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])
        central_proton_yields[region] = np.array([central_yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])
        central_proton_yield_errors[region] = np.array([central_yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])

        # now plot the arrays
        plt.errorbar(assoc_pt_bin_centers, central_inclusive_yields[region], central_inclusive_yield_errors[region], label="central inclusive")
        plt.errorbar(assoc_pt_bin_centers, central_pion_yields[region], central_pion_yield_errors[region], label="central pion")
        plt.errorbar(assoc_pt_bin_centers, central_kaon_yields[region], central_kaon_yield_errors[region], label="central kaon")
        plt.errorbar(assoc_pt_bin_centers, central_proton_yields[region], central_proton_yield_errors[region], label="central proton")
        plt.semilogy()
        plt.legend()
        plt.savefig(f"central_{region}_yields.png")

        plt.close()

        if region in [Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], central_inclusive_yields[region]-central_inclusive_yields[Region.BACKGROUND], np.sqrt(central_inclusive_yield_errors[region]**2+central_inclusive_yield_errors[Region.BACKGROUND]**2), label="central inclusive")
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], central_pion_yields[region]-central_pion_yields[Region.BACKGROUND], np.sqrt(central_pion_yield_errors[region]**2+central_pion_yield_errors[Region.BACKGROUND]**2), label="central pion")
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], central_kaon_yields[region]-central_kaon_yields[Region.BACKGROUND], np.sqrt(central_kaon_yield_errors[region]**2+central_kaon_yield_errors[Region.BACKGROUND]**2), label="central kaon")
            plt.errorbar([assoc_bin.value for assoc_bin in AssociatedHadronMomentumBin], central_proton_yields[region]-central_proton_yields[Region.BACKGROUND], np.sqrt(central_proton_yield_errors[region]**2+central_proton_yield_errors[Region.BACKGROUND]**2), label="central proton")
            plt.semilogy()
            plt.legend()
            plt.savefig(f"central_{region}_yields_minus_background.png")

            plt.close()
