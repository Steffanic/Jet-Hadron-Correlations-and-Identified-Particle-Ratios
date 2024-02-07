import csv
from math import pi
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin, TriggerJetMomentumBin
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt
from JetHadronAnalysis.Plotting import plotTH1, plotArrays

def getYieldAndError(analysis, particle_type, azimuthal_correlation_function, loadFractionsFromDB=True): # I have to add the yield error band calculation
    # now get the per species azimuthal correlation functions for each region by scaling
    particle_pid_fit_shape_sys_err = None
    particle_pid_fit_yield_sys_err = None
    
    if analysis.currentRegion == Region.BACKGROUND:
        background_azimuthal_correlation_function = azimuthal_correlation_function
        background_function = analysis.getBackgroundFunction(background_azimuthal_correlation_function)
        azimuthal_correlation_function = background_function

    if particle_type == pt.INCLUSIVE:
        particle_azimuthal_correlation_function = azimuthal_correlation_function
    else:
        particle_azimuthal_correlation_function, particle_pid_fit_shape_sys_err, particle_pid_fit_yield_sys_err = analysis.getAzimuthalCorrelationFunctionforParticleType(particle_type, azimuthal_correlation_function, loadFractionsFromDB)
    yield_ = analysis.getYieldFromAzimuthalCorrelationFunction(particle_azimuthal_correlation_function, particle_pid_fit_shape_sys_err,particle_pid_fit_yield_sys_err)
    return yield_[0], yield_[1], yield_[2], yield_[3]

def getAzimuthalCorrelationFunction(analysis):
    correlation_function = analysis.getDifferentialCorrelationFunction(True)
    mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
    acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
    azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)
    return azimuthal_correlation_function

def run_analysis(ana: Analysis, loadFractionsFromDB=False):
    print(f"Loading {ana.analysisType.name} files")
    load_from_file=False
    yields = {}
    yield_errors = {}
    yield_pid_fit_shape_sys_errors = {}
    yield_pid_fit_yield_sys_errors = {}

    # make the arrays of yields
    inclusive_yields = {}
    inclusive_yield_errors = {}
    pion_yields = {}
    pion_yield_errors = {}
    pion_yield_pid_fit_shape_sys_errors = {}
    pion_yield_pid_fit_yield_sys_errors = {}
    kaon_yields = {}
    kaon_yield_errors = {}
    kaon_yield_pid_fit_shape_sys_errors = {}
    kaon_yield_pid_fit_yield_sys_errors = {}
    proton_yields = {}
    proton_yield_errors = {}
    proton_yield_pid_fit_shape_sys_errors = {}
    proton_yield_pid_fit_yield_sys_errors = {}

    inclusive_bgsub_yields = {}
    inclusive_bgsub_yield_errors = {}
    pion_bgsub_yields = {}
    pion_bgsub_yield_errors = {}
    pion_bgsub_pid_fit_shape_sys_errors = {}
    pion_bgsub_pid_fit_yield_sys_errors = {}
    kaon_bgsub_yields = {}
    kaon_bgsub_yield_errors = {}
    kaon_bgsub_pid_fit_shape_sys_errors = {}
    kaon_bgsub_pid_fit_yield_sys_errors = {}
    proton_bgsub_yields = {}
    proton_bgsub_yield_errors = {}
    proton_bgsub_pid_fit_shape_sys_errors = {}
    proton_bgsub_pid_fit_yield_sys_errors = {}

    for region in [Region.BACKGROUND, Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
        if load_from_file:
            region_for_file = region.name.lower().replace("_signal", "")
            with open(f"{ana.analysisType.name.lower()}_{region_for_file}_yields.csv", "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                data = list(reader)
                data = np.array(data).astype(float)
                
                inclusive_yields[region] = data[:,1]
                inclusive_yield_errors[region] = data[:,2]
                pion_yields[region] = data[:,3]
                pion_yield_errors[region] = data[:,4]
                pion_yield_pid_fit_shape_sys_errors[region] = data[:,5]
                pion_yield_pid_fit_yield_sys_errors[region] = data[:,6]
                kaon_yields[region] = data[:,7]
                kaon_yield_errors[region] = data[:,8]
                kaon_yield_pid_fit_shape_sys_errors[region] = data[:,9]
                kaon_yield_pid_fit_yield_sys_errors[region] = data[:,10]
                proton_yields[region] = data[:,11]
                proton_yield_errors[region] = data[:,12]
                proton_yield_pid_fit_shape_sys_errors[region] = data[:,13]
                proton_yield_pid_fit_yield_sys_errors[region] = data[:,14]
                inclusive_bgsub_yields[region] = data[:,15]
                inclusive_bgsub_yield_errors[region] = data[:,16]
                pion_bgsub_yields[region] = data[:,17]
                pion_bgsub_yield_errors[region] = data[:,18]
                pion_bgsub_pid_fit_shape_sys_errors[region] = data[:,19]
                pion_bgsub_pid_fit_yield_sys_errors[region] = data[:,20]
                kaon_bgsub_yields[region] = data[:,21]
                kaon_bgsub_yield_errors[region] = data[:,22]
                kaon_bgsub_pid_fit_shape_sys_errors[region] = data[:,23]
                kaon_bgsub_pid_fit_yield_sys_errors[region] = data[:,24]
                proton_bgsub_yields[region] = data[:,25]
                proton_bgsub_yield_errors[region] = data[:,26]
                proton_bgsub_pid_fit_shape_sys_errors[region] = data[:,27]
                proton_bgsub_pid_fit_yield_sys_errors[region] = data[:,28]
        else:
            ana.setTriggerJetMomentumBin(TriggerJetMomentumBin.PT_20_40)
            for assoc_bin in AssociatedHadronMomentumBin:
                print(f"Starting {assoc_bin} in {region} for {ana.analysisType.name}")
                ana.setAssociatedHadronMomentumBin(assoc_bin)
                ana.setRegion(region)
                
                azimuthal_correlation_function = getAzimuthalCorrelationFunction(ana)

                inclusive_yield, inclusive_yield_error, inclusive_pid_fit_shape_err, inclusive_pid_fit_yield_err = getYieldAndError(ana, pt.INCLUSIVE, azimuthal_correlation_function, loadFractionsFromDB)
                pion_yield, pion_yield_error, pion_pid_fit_shape_err, pion_pid_fit_yield_err = getYieldAndError(ana, pt.PION, azimuthal_correlation_function, loadFractionsFromDB)
                kaon_yield, kaon_yield_error, kaon_pid_fit_shape_err, kaon_pid_fit_yield_err  = getYieldAndError(ana, pt.KAON, azimuthal_correlation_function, loadFractionsFromDB)
                proton_yield, proton_yield_error, proton_pid_fit_shape_err, proton_pid_fit_yield_err = getYieldAndError(ana, pt.PROTON, azimuthal_correlation_function, loadFractionsFromDB)

                print(f"Yield for {assoc_bin} in {ana.analysisType} is {inclusive_yield}")
                print(f"Pion yield for {assoc_bin} in {ana.analysisType} is {pion_yield}")
                print(f"Kaon yield for {assoc_bin} in {ana.analysisType} is {kaon_yield}")
                print(f"Proton yield for {assoc_bin} in {ana.analysisType} is {proton_yield}")
                yields[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield, pt.PION:pion_yield, pt.KAON:kaon_yield, pt.PROTON:proton_yield}
                yield_errors[(region, assoc_bin)] = {pt.INCLUSIVE:inclusive_yield_error, pt.PION:pion_yield_error, pt.KAON:kaon_yield_error, pt.PROTON:proton_yield_error}
                yield_pid_fit_shape_sys_errors[(region, assoc_bin)] = {pt.INCLUSIVE:np.zeros_like(pion_pid_fit_shape_err), pt.PION:pion_pid_fit_shape_err, pt.KAON:kaon_pid_fit_shape_err, pt.PROTON:proton_pid_fit_shape_err}
                yield_pid_fit_yield_sys_errors[(region, assoc_bin)] = {pt.INCLUSIVE:np.zeros_like(pion_pid_fit_yield_err), pt.PION:pion_pid_fit_yield_err, pt.KAON:kaon_pid_fit_yield_err, pt.PROTON:proton_pid_fit_yield_err}


            inclusive_yields[region] = np.array([yields[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            inclusive_yield_errors[region] = np.array([yield_errors[(region,assoc_bin)][pt.INCLUSIVE] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)

            pion_yields[region] = np.array([yields[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            pion_yield_errors[region] = np.array([yield_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            pion_yield_pid_fit_shape_sys_errors[region] = np.array([yield_pid_fit_shape_sys_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            pion_yield_pid_fit_yield_sys_errors[region] = np.array([yield_pid_fit_yield_sys_errors[(region,assoc_bin)][pt.PION] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)

            kaon_yields[region] = np.array([yields[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            kaon_yield_errors[region] = np.array([yield_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            kaon_yield_pid_fit_shape_sys_errors[region] = np.array([yield_pid_fit_shape_sys_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            kaon_yield_pid_fit_yield_sys_errors[region] = np.array([yield_pid_fit_yield_sys_errors[(region,assoc_bin)][pt.KAON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)

            proton_yields[region] = np.array([yields[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            proton_yield_errors[region] = np.array([yield_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            proton_yield_pid_fit_shape_sys_errors[region] = np.array([yield_pid_fit_shape_sys_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)
            proton_yield_pid_fit_yield_sys_errors[region] = np.array([yield_pid_fit_yield_sys_errors[(region,assoc_bin)][pt.PROTON] for assoc_bin in AssociatedHadronMomentumBin])/np.array(assoc_pt_bin_widths)

            if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
                inclusive_bgsub_yields[region] = inclusive_yields[region]-inclusive_yields[Region.BACKGROUND]
                inclusive_bgsub_yield_errors[region] = np.sqrt(inclusive_yield_errors[region]**2+inclusive_yield_errors[Region.BACKGROUND]**2)

                pion_bgsub_yields[region] = pion_yields[region]-pion_yields[Region.BACKGROUND]
                pion_bgsub_yield_errors[region] = np.sqrt(pion_yield_errors[region]**2+pion_yield_errors[Region.BACKGROUND]**2)
                pion_bgsub_pid_fit_shape_sys_errors[region] = np.sqrt(pion_yield_pid_fit_shape_sys_errors[region]**2+pion_yield_pid_fit_shape_sys_errors[Region.BACKGROUND]**2)
                pion_bgsub_pid_fit_yield_sys_errors[region] = np.sqrt(pion_yield_pid_fit_yield_sys_errors[region]**2+pion_yield_pid_fit_yield_sys_errors[Region.BACKGROUND]**2)

                kaon_bgsub_yields[region] = kaon_yields[region]-kaon_yields[Region.BACKGROUND]
                kaon_bgsub_yield_errors[region] = np.sqrt(kaon_yield_errors[region]**2+kaon_yield_errors[Region.BACKGROUND]**2)
                kaon_bgsub_pid_fit_shape_sys_errors[region] = np.sqrt(kaon_yield_pid_fit_shape_sys_errors[region]**2+kaon_yield_pid_fit_shape_sys_errors[Region.BACKGROUND]**2)
                kaon_bgsub_pid_fit_yield_sys_errors[region] = np.sqrt(kaon_yield_pid_fit_yield_sys_errors[region]**2+kaon_yield_pid_fit_yield_sys_errors[Region.BACKGROUND]**2)

                proton_bgsub_yields[region] = proton_yields[region]-proton_yields[Region.BACKGROUND]
                proton_bgsub_yield_errors[region] = np.sqrt(proton_yield_errors[region]**2+proton_yield_errors[Region.BACKGROUND]**2)
                proton_bgsub_pid_fit_shape_sys_errors[region] = np.sqrt(proton_yield_pid_fit_shape_sys_errors[region]**2+proton_yield_pid_fit_shape_sys_errors[Region.BACKGROUND]**2)
                proton_bgsub_pid_fit_yield_sys_errors[region] = np.sqrt(proton_yield_pid_fit_yield_sys_errors[region]**2+proton_yield_pid_fit_yield_sys_errors[Region.BACKGROUND]**2)


        # now plot the arrays
        plt.errorbar(assoc_pt_bin_centers, inclusive_yields[region], inclusive_yield_errors[region], label=f"{ana.analysisType.name.lower()} inclusive", color="black")
        plt.errorbar(assoc_pt_bin_centers, pion_yields[region], pion_yield_errors[region], label=f"{ana.analysisType.name.lower()} pion", color="blue")
        plt.errorbar(assoc_pt_bin_centers, kaon_yields[region], kaon_yield_errors[region], label=f"{ana.analysisType.name.lower()} kaon", color="green")
        plt.errorbar(assoc_pt_bin_centers, proton_yields[region], proton_yield_errors[region], label=f"{ana.analysisType.name.lower()} proton", color="red")


        plt.fill_between(assoc_pt_bin_centers, pion_yields[region]-pion_yield_pid_fit_shape_sys_errors[region], pion_yields[region]+pion_yield_pid_fit_shape_sys_errors[region], alpha=0.2, color="blue", linestyle="dashed", label="PID fit shape systematic error")
        plt.fill_between(assoc_pt_bin_centers, pion_yields[region]-pion_yield_pid_fit_yield_sys_errors[region], pion_yields[region]+pion_yield_pid_fit_yield_sys_errors[region], alpha=0.2, color="blue", linestyle="dotted", label="PID fit yield systematic error")

        plt.fill_between(assoc_pt_bin_centers, kaon_yields[region]-kaon_yield_pid_fit_shape_sys_errors[region], kaon_yields[region]+kaon_yield_pid_fit_shape_sys_errors[region], alpha=0.2, color="green", linestyle="dashed")
        plt.fill_between(assoc_pt_bin_centers, kaon_yields[region]-kaon_yield_pid_fit_yield_sys_errors[region], kaon_yields[region]+kaon_yield_pid_fit_yield_sys_errors[region], alpha=0.2, color="green", linestyle="dotted")

        plt.fill_between(assoc_pt_bin_centers, proton_yields[region]-proton_yield_pid_fit_shape_sys_errors[region], proton_yields[region]+proton_yield_pid_fit_shape_sys_errors[region], alpha=0.2, color="red", linestyle="dashed")
        plt.fill_between(assoc_pt_bin_centers, proton_yields[region]-proton_yield_pid_fit_yield_sys_errors[region], proton_yields[region]+proton_yield_pid_fit_yield_sys_errors[region], alpha=0.2, color="red", linestyle="dotted")
        
        plt.title(f"{ana.analysisType.name.lower()} {region.name} yields")
        plt.xlabel("Associated hadron pT[GeV/c]")
        plt.ylabel("Yield")
        plt.semilogy()
        plt.xticks(assoc_pt_bin_centers)
        if ana.analysisType == at.PP:
            plt.ylim(1e-6, 1e-1)
        else:
            plt.ylim(1e-5, 1e0)
        plt.legend()
        plt.savefig(f"Plots/{ana.analysisType.name}/{region}_yields.png")

        plt.close()

        if region in [Region.INCLUSIVE, Region.NEAR_SIDE_SIGNAL, Region.AWAY_SIDE_SIGNAL]:
            # now plot the near_side yields minus the background yields
            plt.errorbar(assoc_pt_bin_centers, inclusive_bgsub_yields[region], inclusive_bgsub_yield_errors[region], label=f"{ana.analysisType.name.lower()} inclusive background subtracted", color="black")
            plt.errorbar(assoc_pt_bin_centers, pion_bgsub_yields[region], pion_bgsub_yield_errors[region], label=f"{ana.analysisType.name.lower()} pion background subtracted", color="blue")
            plt.errorbar(assoc_pt_bin_centers, kaon_bgsub_yields[region], kaon_bgsub_yield_errors[region], label=f"{ana.analysisType.name.lower()} kaon background subtracted", color="green")
            plt.errorbar(assoc_pt_bin_centers, proton_bgsub_yields[region], proton_bgsub_yield_errors[region], label=f"{ana.analysisType.name.lower()} proton", color="red")


            plt.fill_between(assoc_pt_bin_centers, pion_bgsub_yields[region]-pion_bgsub_pid_fit_shape_sys_errors[region], pion_bgsub_yields[region]+pion_bgsub_pid_fit_shape_sys_errors[region], alpha=0.2, color="blue", linestyle="dashed", label="PID fit shape systematic error")
            plt.fill_between(assoc_pt_bin_centers, pion_bgsub_yields[region]-pion_bgsub_pid_fit_yield_sys_errors[region], pion_bgsub_yields[region]+pion_bgsub_pid_fit_yield_sys_errors[region], alpha=0.2, color="blue", linestyle="dotted", label="PID fit yield systematic error")

            plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[region]-kaon_bgsub_pid_fit_shape_sys_errors[region], kaon_bgsub_yields[region]+kaon_bgsub_pid_fit_shape_sys_errors[region], alpha=0.2, color="green", linestyle="dashed")
            plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[region]-kaon_bgsub_pid_fit_yield_sys_errors[region], kaon_bgsub_yields[region]+kaon_bgsub_pid_fit_yield_sys_errors[region], alpha=0.2, color="green", linestyle="dotted")

            plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[region]-proton_bgsub_pid_fit_shape_sys_errors[region], proton_bgsub_yields[region]+proton_bgsub_pid_fit_shape_sys_errors[region], alpha=0.2, color="red", linestyle="dashed")
            plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[region]-proton_bgsub_pid_fit_yield_sys_errors[region], proton_bgsub_yields[region]+proton_bgsub_pid_fit_yield_sys_errors[region], alpha=0.2, color="red", linestyle="dotted")

            plt.title(f"{ana.analysisType.name.lower()} {region.name} background subtracted yields")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Yield")
            plt.semilogy()
            plt.xticks(assoc_pt_bin_centers)
            if ana.analysisType == at.PP:
                plt.ylim(1e-6, 1e-1)
            else:
                plt.ylim(1e-5, 1e0)
            plt.legend()
            plt.savefig(f"Plots/{ana.analysisType.name}/{region}_background_subtracted_yields.png")

            plt.close()

            if ana.analysisType == at.PP:
                published_p_to_pi_ratios = pd.read_csv("ppPtoPi.csv") 
                published_k_to_pi_ratios = pd.read_csv("ppKtoPi.csv")
            if ana.analysisType == at.SEMICENTRAL:
                published_p_to_pi_ratios = pd.read_csv("30-50PbPbPtoPi.csv") 
                published_k_to_pi_ratios = pd.read_csv("30-50PbPbKtoPi.csv")
            if ana.analysisType == at.CENTRAL:
                published_p_to_pi_ratios = pd.read_csv("0-10PbPbPtoPi.csv") 
                published_k_to_pi_ratios = pd.read_csv("0-10PbPbKtoPi.csv")


            p_to_pi_pT_bin_centers = published_p_to_pi_ratios['pT']
            p_to_pi_ratios = published_p_to_pi_ratios['ratio']
            p_to_pi_ratio_errors = published_p_to_pi_ratios['abserror']

            k_to_pi_pT_bin_centers = published_k_to_pi_ratios['pT']
            k_to_pi_ratios = published_k_to_pi_ratios['ratio']
            k_to_pi_ratio_errors = published_k_to_pi_ratios['abserror']

            # now the ratios proton to pion and kaon to pion 
            plt.errorbar(assoc_pt_bin_centers, proton_bgsub_yields[region]/pion_bgsub_yields[region], proton_bgsub_yield_errors[region]/pion_bgsub_yields[region], color="red", label="Data")

            plt.errorbar(p_to_pi_pT_bin_centers, p_to_pi_ratios, p_to_pi_ratio_errors, label=f"Published", color="red", marker="o", linestyle="none")

            shape_sys_err = np.sqrt((proton_bgsub_pid_fit_shape_sys_errors[region]/pion_bgsub_yields[region])**2+(pion_bgsub_pid_fit_shape_sys_errors[region]*proton_bgsub_yields[region]/pion_bgsub_yields[region]**2)**2)
            plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[region]/pion_bgsub_yields[region] - shape_sys_err, proton_bgsub_yields[region]/pion_bgsub_yields[region] + shape_sys_err, alpha=0.2, color="red", linestyle="dashed", label="PID shape systematic")
            yield_sys_err = np.sqrt((proton_bgsub_pid_fit_yield_sys_errors[region]/pion_bgsub_yields[region])**2+(pion_bgsub_pid_fit_yield_sys_errors[region]*proton_bgsub_yields[region]/pion_bgsub_yields[region]**2)**2)
            plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[region]/pion_bgsub_yields[region] - yield_sys_err, proton_bgsub_yields[region]/pion_bgsub_yields[region] + yield_sys_err, alpha=0.2, color="red", linestyle="dotted", label="PID yield systematic")
            plt.title(f"{ana.analysisType.name.lower()} {region.name} proton to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.xlim(0, 10)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/{ana.analysisType.name}/{region}_proton_to_pion_ratio.png")

            plt.close()

            plt.errorbar(assoc_pt_bin_centers, kaon_bgsub_yields[region]/pion_bgsub_yields[region], kaon_bgsub_yield_errors[region]/pion_bgsub_yields[region], label="Data", color="green")

            plt.errorbar(k_to_pi_pT_bin_centers, k_to_pi_ratios, k_to_pi_ratio_errors, label="Published", color="green", marker="o", linestyle="none")

            shape_sys_err = np.sqrt((kaon_bgsub_pid_fit_shape_sys_errors[region]/pion_bgsub_yields[region])**2+(pion_bgsub_pid_fit_shape_sys_errors[region]*kaon_bgsub_yields[region]/pion_bgsub_yields[region]**2)**2)
            plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[region]/pion_bgsub_yields[region] - shape_sys_err, kaon_bgsub_yields[region]/pion_bgsub_yields[region] + shape_sys_err, alpha=0.2, color="green", linestyle="dashed", label="PID shape systematic")
            yield_sys_err = np.sqrt((kaon_bgsub_pid_fit_yield_sys_errors[region]/pion_bgsub_yields[region])**2+(pion_bgsub_pid_fit_yield_sys_errors[region]*kaon_bgsub_yields[region]/pion_bgsub_yields[region]**2)**2)
            plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[region]/pion_bgsub_yields[region] - yield_sys_err, kaon_bgsub_yields[region]/pion_bgsub_yields[region] + yield_sys_err, alpha=0.2, color="green", linestyle="dotted", label="PID yield systematic")

            plt.title(f"{ana.analysisType.name.lower()} {region.name} kaon to pion ratio")
            plt.xlabel("Associated hadron pT[GeV/c]")
            plt.ylabel("Ratio")
            plt.xticks(assoc_pt_bin_centers)
            plt.ylim(0, 1.2)
            plt.xlim(0, 10)
            plt.hlines(1.0, 0, 10, color="black")
            plt.legend()
            plt.savefig(f"Plots/{ana.analysisType.name}/{region}_kaon_to_pion_ratio.png")

            plt.close()

    # load the published ratios
    if ana.analysisType == at.PP:
        published_p_to_pi_ratios = pd.read_csv("ppPtoPi.csv") 
        published_k_to_pi_ratios = pd.read_csv("ppKtoPi.csv")
    if ana.analysisType == at.SEMICENTRAL:
        published_p_to_pi_ratios = pd.read_csv("30-50PbPbPtoPi.csv") 
        published_k_to_pi_ratios = pd.read_csv("30-50PbPbKtoPi.csv")
    if ana.analysisType == at.CENTRAL:
        published_p_to_pi_ratios = pd.read_csv("0-10PbPbPtoPi.csv") 
        published_k_to_pi_ratios = pd.read_csv("0-10PbPbKtoPi.csv")

    p_to_pi_pT_bin_centers = published_p_to_pi_ratios['pT']
    p_to_pi_ratios = published_p_to_pi_ratios['ratio']
    p_to_pi_ratio_errors = published_p_to_pi_ratios['abserror']

    k_to_pi_pT_bin_centers = published_k_to_pi_ratios['pT']
    k_to_pi_ratios = published_k_to_pi_ratios['ratio']
    k_to_pi_ratio_errors = published_k_to_pi_ratios['abserror']


    
    # now plot all three region ratios on the same plot
    plt.errorbar(assoc_pt_bin_centers, proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL], proton_bgsub_yield_errors[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL], label=f"near side", color="red", linestyle="dashed")
    plt.errorbar(assoc_pt_bin_centers, proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL], proton_bgsub_yield_errors[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL], label=f"away side", color="red", linestyle="dotted")
    plt.errorbar(assoc_pt_bin_centers, proton_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE], proton_bgsub_yield_errors[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE], label=f"inclusive", color="red")

    near_side_shape_sys_err = np.sqrt((proton_bgsub_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL]*proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] - near_side_shape_sys_err, proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] + near_side_shape_sys_err, alpha=0.2, color="red", linestyle="dashed", label="PID shape systematic")
    near_side_yield_sys_err = np.sqrt((proton_bgsub_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL]*proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] - near_side_yield_sys_err, proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] + near_side_yield_sys_err, alpha=0.2, color="red", linestyle="dotted", label="PID yield systematic")

    away_side_shape_sys_err = np.sqrt((proton_bgsub_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL]*proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] - away_side_shape_sys_err, proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] + away_side_shape_sys_err, alpha=0.2, color="red", linestyle="dashed")
    away_side_yield_sys_err = np.sqrt((proton_bgsub_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL]*proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] - away_side_yield_sys_err, proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] + away_side_yield_sys_err, alpha=0.2, color="red", linestyle="dotted")

    inclusive_shape_sys_err = np.sqrt((proton_bgsub_pid_fit_shape_sys_errors[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE])**2+(pion_bgsub_pid_fit_shape_sys_errors[Region.INCLUSIVE]*proton_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] - inclusive_shape_sys_err, proton_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] + inclusive_shape_sys_err, alpha=0.2, color="red", linestyle="dashed")
    inclusive_yield_sys_err = np.sqrt((proton_bgsub_pid_fit_yield_sys_errors[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE])**2+(pion_bgsub_pid_fit_yield_sys_errors[Region.INCLUSIVE]*proton_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, proton_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] - inclusive_yield_sys_err, proton_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] + inclusive_yield_sys_err, alpha=0.2, color="red", linestyle="dotted")

    # plot the published ratios
    plt.errorbar(p_to_pi_pT_bin_centers, p_to_pi_ratios, p_to_pi_ratio_errors, label="Published Inclusive", color="red", marker="o", linestyle="none")

    plt.title(f"{ana.analysisType.name.lower()} proton to pion ratio")
    plt.xlabel("Associated hadron pT[GeV/c]")
    plt.ylabel("Ratio")
    plt.xticks(assoc_pt_bin_centers)
    plt.ylim(0, 1.2)
    plt.xlim(0, 10)
    plt.legend()
    plt.savefig(f"Plots/{ana.analysisType.name}/proton_to_pion_ratio_all_regions.png")

    plt.close()

    plt.errorbar(assoc_pt_bin_centers, kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL], kaon_bgsub_yield_errors[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL], label="near side", color="green", linestyle="dashed")
    plt.errorbar(assoc_pt_bin_centers, kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL], kaon_bgsub_yield_errors[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL], label="away side", color="green", linestyle="dotted")
    plt.errorbar(assoc_pt_bin_centers, kaon_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE], kaon_bgsub_yield_errors[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE], label="inclusive", color="green")

    near_side_shape_sys_err = np.sqrt((kaon_bgsub_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL]*kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] - near_side_shape_sys_err, kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] + near_side_shape_sys_err, alpha=0.2, color="green", linestyle="dashed", label="PID shape systematic")
    near_side_yield_sys_err = np.sqrt((kaon_bgsub_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL]*kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] - near_side_yield_sys_err, kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL]/pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL] + near_side_yield_sys_err, alpha=0.2, color="green", linestyle="dotted", label="PID yield systematic")

    away_side_shape_sys_err = np.sqrt((kaon_bgsub_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL]*kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] - away_side_shape_sys_err, kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] + away_side_shape_sys_err, alpha=0.2, color="green", linestyle="dashed")
    away_side_yield_sys_err = np.sqrt((kaon_bgsub_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL])**2+(pion_bgsub_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL]*kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] - away_side_yield_sys_err, kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL]/pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL] + away_side_yield_sys_err, alpha=0.2, color="green", linestyle="dotted")

    inclusive_shape_sys_err = np.sqrt((kaon_bgsub_pid_fit_shape_sys_errors[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE])**2+(pion_bgsub_pid_fit_shape_sys_errors[Region.INCLUSIVE]*kaon_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] - inclusive_shape_sys_err, kaon_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] + inclusive_shape_sys_err, alpha=0.2, color="green", linestyle="dashed")
    inclusive_yield_sys_err = np.sqrt((kaon_bgsub_pid_fit_yield_sys_errors[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE])**2+(pion_bgsub_pid_fit_yield_sys_errors[Region.INCLUSIVE]*kaon_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE]**2)**2)
    plt.fill_between(assoc_pt_bin_centers, kaon_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] - inclusive_yield_sys_err, kaon_bgsub_yields[Region.INCLUSIVE]/pion_bgsub_yields[Region.INCLUSIVE] + inclusive_yield_sys_err, alpha=0.2, color="green", linestyle="dotted")

    # plot the published ratios
    plt.errorbar(k_to_pi_pT_bin_centers, k_to_pi_ratios, k_to_pi_ratio_errors, label="Published Inclusive", color="green", marker="o", linestyle="none")
    
    plt.title(f"{ana.analysisType.name.lower()} kaon to pion ratio")
    plt.xlabel("Associated hadron pT[GeV/c]")
    plt.ylabel("Ratio")
    plt.xticks(assoc_pt_bin_centers)
    plt.ylim(0, 1.2)
    plt.xlim(0, 10)
    plt.legend()
    plt.savefig(f"Plots/{ana.analysisType.name}/kaon_to_pion_ratio_all_regions.png")

    plt.close()



        
    #     # save the yields to a csv file
    #     header = ["assoc_pt_bin_center", "inclusive_yield", "inclusive_yield_error", "pion_yield", "pion_yield_error", "pion_yield_pid_fit_shape_sys_error", "pion_yield_pid_fit_yield_sys_error", "kaon_yield", "kaon_yield_error", "kaon_yield_pid_fit_shape_sys_error", "kaon_yield_pid_fit_yield_sys_error", "proton_yield", "proton_yield_error", "proton_yield_pid_fit_shape_sys_error", "proton_yield_pid_fit_yield_sys_error", "inclusive_bgsub_yield", "inclusive_bgsub_yield_error", "pion_bgsub_yield", "pion_bgsub_yield_error", "pion_bgsub_yield_pid_fit_shape_sys_error", "pion_bgsub_yield_pid_fit_yield_sys_error", "kaon_bgsub_yield", "kaon_bgsub_yield_error", "kaon_bgsub_yield_pid_fit_shape_sys_error", "kaon_bgsub_yield_pid_fit_yield_sys_error", "proton_bgsub_yield", "proton_bgsub_yield_error", "proton_bgsub_yield_pid_fit_shape_sys_error", "proton_bgsub_yield_pid_fit_yield_sys_error"]
    #     background_data = [assoc_pt_bin_centers, pp_inclusive_yields[Region.BACKGROUND], pp_inclusive_yield_errors[Region.BACKGROUND], pp_pion_yields[Region.BACKGROUND], pp_pion_yield_errors[Region.BACKGROUND], pp_pion_yield_pid_fit_shape_sys_errors[Region.BACKGROUND], pp_pion_yield_pid_fit_yield_sys_errors[Region.BACKGROUND], pp_kaon_yields[Region.BACKGROUND], pp_kaon_yield_errors[Region.BACKGROUND], pp_kaon_yield_pid_fit_shape_sys_errors[Region.BACKGROUND], pp_kaon_yield_pid_fit_yield_sys_errors[Region.BACKGROUND], pp_proton_yields[Region.BACKGROUND], pp_proton_yield_errors[Region.BACKGROUND], pp_proton_yield_pid_fit_shape_sys_errors[Region.BACKGROUND], pp_proton_yield_pid_fit_yield_sys_errors[Region.BACKGROUND], pp_inclusive_bgsub_yields[Region.BACKGROUND], pp_inclusive_bgsub_yield_errors[Region.BACKGROUND], pp_pion_bgsub_yields[Region.BACKGROUND], pp_pion_bgsub_yield_errors[Region.BACKGROUND], pp_pion_bgsub_pid_fit_shape_sys_errors[Region.BACKGROUND], pp_pion_bgsub_pid_fit_yield_sys_errors[Region.BACKGROUND], pp_kaon_bgsub_yields[Region.BACKGROUND], pp_kaon_bgsub_yield_errors[Region.BACKGROUND], pp_kaon_bgsub_pid_fit_shape_sys_errors[Region.BACKGROUND], pp_kaon_bgsub_pid_fit_yield_sys_errors[Region.BACKGROUND], pp_proton_bgsub_yields[Region.BACKGROUND], pp_proton_bgsub_yield_errors[Region.BACKGROUND], pp_proton_bgsub_pid_fit_shape_sys_errors[Region.BACKGROUND], pp_proton_bgsub_pid_fit_yield_sys_errors[Region.BACKGROUND]]
    #     inclusive_data = [assoc_pt_bin_centers, pp_inclusive_yields[Region.INCLUSIVE], pp_inclusive_yield_errors[Region.INCLUSIVE], pp_pion_yields[Region.INCLUSIVE], pp_pion_yield_errors[Region.INCLUSIVE], pp_pion_yield_pid_fit_shape_sys_errors[Region.INCLUSIVE], pp_pion_yield_pid_fit_yield_sys_errors[Region.INCLUSIVE], pp_kaon_yields[Region.INCLUSIVE], pp_kaon_yield_errors[Region.INCLUSIVE], pp_kaon_yield_pid_fit_shape_sys_errors[Region.INCLUSIVE], pp_kaon_yield_pid_fit_yield_sys_errors[Region.INCLUSIVE], pp_proton_yields[Region.INCLUSIVE], pp_proton_yield_errors[Region.INCLUSIVE], pp_proton_yield_pid_fit_shape_sys_errors[Region.INCLUSIVE], pp_proton_yield_pid_fit_yield_sys_errors[Region.INCLUSIVE], pp_inclusive_bgsub_yields[Region.INCLUSIVE], pp_inclusive_bgsub_yield_errors[Region.INCLUSIVE], pp_pion_bgsub_yields[Region.INCLUSIVE], pp_pion_bgsub_yield_errors[Region.INCLUSIVE], pp_pion_bgsub_pid_fit_shape_sys_errors[Region.INCLUSIVE], pp_pion_bgsub_pid_fit_yield_sys_errors[Region.INCLUSIVE], pp_kaon_bgsub_yields[Region.INCLUSIVE], pp_kaon_bgsub_yield_errors[Region.INCLUSIVE], pp_kaon_bgsub_pid_fit_shape_sys_errors[Region.INCLUSIVE], pp_kaon_bgsub_pid_fit_yield_sys_errors[Region.INCLUSIVE], pp_proton_bgsub_yields[Region.INCLUSIVE], pp_proton_bgsub_yield_errors[Region.INCLUSIVE], pp_proton_bgsub_pid_fit_shape_sys_errors[Region.INCLUSIVE], pp_proton_bgsub_pid_fit_yield_sys_errors[Region.INCLUSIVE]]
    #     near_side_data = [assoc_pt_bin_centers, pp_inclusive_yields[Region.NEAR_SIDE_SIGNAL], pp_inclusive_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_pion_yields[Region.NEAR_SIDE_SIGNAL], pp_pion_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_pion_yield_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_pion_yield_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_kaon_yields[Region.NEAR_SIDE_SIGNAL], pp_kaon_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_kaon_yield_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_kaon_yield_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_proton_yields[Region.NEAR_SIDE_SIGNAL], pp_proton_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_proton_yield_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_proton_yield_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_inclusive_bgsub_yields[Region.NEAR_SIDE_SIGNAL], pp_inclusive_bgsub_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_pion_bgsub_yields[Region.NEAR_SIDE_SIGNAL], pp_pion_bgsub_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_pion_bgsub_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_pion_bgsub_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_kaon_bgsub_yields[Region.NEAR_SIDE_SIGNAL], pp_kaon_bgsub_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_kaon_bgsub_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_kaon_bgsub_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_proton_bgsub_yields[Region.NEAR_SIDE_SIGNAL], pp_proton_bgsub_yield_errors[Region.NEAR_SIDE_SIGNAL], pp_proton_bgsub_pid_fit_shape_sys_errors[Region.NEAR_SIDE_SIGNAL], pp_proton_bgsub_pid_fit_yield_sys_errors[Region.NEAR_SIDE_SIGNAL]]
    #     away_side_data = [assoc_pt_bin_centers, pp_inclusive_yields[Region.AWAY_SIDE_SIGNAL], pp_inclusive_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_pion_yields[Region.AWAY_SIDE_SIGNAL], pp_pion_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_pion_yield_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_pion_yield_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_kaon_yields[Region.AWAY_SIDE_SIGNAL], pp_kaon_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_kaon_yield_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_kaon_yield_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_proton_yields[Region.AWAY_SIDE_SIGNAL], pp_proton_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_proton_yield_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_proton_yield_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_inclusive_bgsub_yields[Region.AWAY_SIDE_SIGNAL], pp_inclusive_bgsub_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_pion_bgsub_yields[Region.AWAY_SIDE_SIGNAL], pp_pion_bgsub_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_pion_bgsub_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_pion_bgsub_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_kaon_bgsub_yields[Region.AWAY_SIDE_SIGNAL], pp_kaon_bgsub_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_kaon_bgsub_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_kaon_bgsub_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_proton_bgsub_yields[Region.AWAY_SIDE_SIGNAL], pp_proton_bgsub_yield_errors[Region.AWAY_SIDE_SIGNAL], pp_proton_bgsub_pid_fit_shape_sys_errors[Region.AWAY_SIDE_SIGNAL], pp_proton_bgsub_pid_fit_yield_sys_errors[Region.AWAY_SIDE_SIGNAL]]
    # 
    #     with open("pp_background_yields.csv", "w") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(header)
    #         writer.writerows(zip(*background_data))
    # 
    #     with open("pp_inclusive_yields.csv", "w") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(header)
    #         writer.writerows(zip(*inclusive_data))
    # 
    #     with open("pp_near_side_yields.csv", "w") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(header)
    #         writer.writerows(zip(*near_side_data))
    # 
    #     with open("pp_away_side_yields.csv", "w") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(header)
    #         writer.writerows(zip(*away_side_data))
    
if __name__=="__main__":

    assoc_pt_bin_centers = [1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 8.0]
    assoc_pt_bin_widths = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 4.0]
    loadFractionsFromDB=True
    run_pp = False
    run_semicentral = False
    run_central = True

    # ++++++++++++++++++++++++++++++++++
    # PP 
    # ++++++++++++++++++++++++++++++++++
    if run_pp:
        ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])
        run_analysis(ana_pp, loadFractionsFromDB=loadFractionsFromDB)
    # +++++++++++++++++++++++
    # SemiCentral
    # +++++++++++++++++++++++
    if run_semicentral:
        ana_semicentral = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root", "/mnt/d/18r/new_root/296690.root", "/mnt/d/18r/new_root/296794.root", "/mnt/d/18r/new_root/296894.root", "/mnt/d/18r/new_root/296941.root", "/mnt/d/18r/new_root/297031.root", "/mnt/d/18r/new_root/297085.root", "/mnt/d/18r/new_root/297118.root", "/mnt/d/18r/new_root/297129.root", "/mnt/d/18r/new_root/297372.root", "/mnt/d/18r/new_root/297415.root", "/mnt/d/18r/new_root/297441.root", "/mnt/d/18r/new_root/297446.root", "/mnt/d/18r/new_root/297479.root", "/mnt/d/18r/new_root/297544.root", ])
        run_analysis(ana_semicentral, loadFractionsFromDB=loadFractionsFromDB)
    # +++++++++++++++++++++++
    # Central
    # +++++++++++++++++++++++
    if run_central:
        ana_central = Analysis(at.CENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root", "/mnt/d/18r/new_root/296690.root", "/mnt/d/18r/new_root/296794.root", "/mnt/d/18r/new_root/296894.root", "/mnt/d/18r/new_root/296941.root", "/mnt/d/18r/new_root/297031.root", "/mnt/d/18r/new_root/297085.root", "/mnt/d/18r/new_root/297118.root", "/mnt/d/18r/new_root/297129.root", "/mnt/d/18r/new_root/297372.root", "/mnt/d/18r/new_root/297415.root", "/mnt/d/18r/new_root/297441.root", "/mnt/d/18r/new_root/297446.root", "/mnt/d/18r/new_root/297479.root", "/mnt/d/18r/new_root/297544.root",     ])
        run_analysis(ana_central, loadFractionsFromDB=loadFractionsFromDB)
        

    