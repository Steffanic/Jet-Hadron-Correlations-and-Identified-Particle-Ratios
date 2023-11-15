import numpy as np
from JetHadronAnalysis.Types import AnalysisType, Region, AssociatedHadronMomentumBin, ParticleType, TriggerJetMomentumBin
import sqlite3
import matplotlib.pyplot as plt

from JetHadronAnalysis.RPFDB import getParameterByTriggerAndHadronMomentumForParticleSpecies

def plotFitParametersByTriggerAndHadronMomentum(analysisType, parameter,  expected_range:list, dbCursor, with_reduced_chi2_panel=False):
    inclusive_parameter = getParameterByTriggerAndHadronMomentumForParticleSpecies(analysisType = analysisType, parameter_name = parameter, particle_species=ParticleType.INCLUSIVE, dbCursor=dbCursor, with_reduced_chi2=with_reduced_chi2_panel)
    pion_parameter = getParameterByTriggerAndHadronMomentumForParticleSpecies(analysisType = analysisType, parameter_name = parameter, particle_species=ParticleType.PION, dbCursor=dbCursor, with_reduced_chi2=with_reduced_chi2_panel)
    proton_parameter = getParameterByTriggerAndHadronMomentumForParticleSpecies(analysisType = analysisType, parameter_name = parameter, particle_species=ParticleType.PROTON, dbCursor=dbCursor, with_reduced_chi2=with_reduced_chi2_panel)
    kaon_parameter = getParameterByTriggerAndHadronMomentumForParticleSpecies(analysisType = analysisType, parameter_name = parameter, particle_species=ParticleType.KAON, dbCursor=dbCursor, with_reduced_chi2=with_reduced_chi2_panel)

    # split the data into the different trigger momentum bins and plot them
    # The first 7 elements of each list are the first trigger bin and so on

    inclusive_by_triggerbin = [inclusive_parameter[:7], inclusive_parameter[7:]]
    pion_by_triggerbin = [pion_parameter[:7], pion_parameter[7:]]
    proton_by_triggerbin = [proton_parameter[:7], proton_parameter[7:]]
    kaon_by_triggerbin = [kaon_parameter[:7], kaon_parameter[7:]]
    plt.figure(figsize=(20,10))
    num_rows = 2 if with_reduced_chi2_panel else 1
    fig, ax = plt.subplots(num_rows, 2, figsize=(20,10), sharey="row", sharex=True, gridspec_kw={'height_ratios': [2,1]})
    for i in range(2):
        current_ax = ax[0,i] if with_reduced_chi2_panel else ax[i]
        current_ax.errorbar(x=[x[1] for x in inclusive_by_triggerbin[i]], y=[x[2] for x in inclusive_by_triggerbin[i]], yerr=[x[3] for x in inclusive_by_triggerbin[i]], fmt='o', label=f"Inclusive {parameter}")
        current_ax.errorbar(x=[x[1]+0.1 for x in pion_by_triggerbin[i]], y=[x[2] for x in pion_by_triggerbin[i]], yerr=[x[3] for x in pion_by_triggerbin[i]], fmt='o', label=f"Pion {parameter}")
        current_ax.errorbar(x=[x[1]+0.2 for x in proton_by_triggerbin[i]], y=[x[2] for x in proton_by_triggerbin[i]], yerr=[x[3] for x in proton_by_triggerbin[i]], fmt='o', label=f"Proton {parameter}")
        current_ax.errorbar(x=[x[1]+0.3 for x in kaon_by_triggerbin[i]], y=[x[2] for x in kaon_by_triggerbin[i]], yerr=[x[3] for x in kaon_by_triggerbin[i]], fmt='o', label=f"Kaon {parameter}")
        current_ax.hlines(y=[x[2] for x in inclusive_by_triggerbin[i]], xmin=[x[1] for x in inclusive_by_triggerbin[i]], xmax=[x[1]+0.3 for x in inclusive_by_triggerbin[i]], color='black')
        current_ax.set_ylim(*expected_range)
        current_ax.set_title(f"{parameter} by Hadron Momentum for {analysisType.name} With Trigger Momentum {TriggerJetMomentumBin(i+1).name}")
        current_ax.legend()
        if with_reduced_chi2_panel:
            current_ax = ax[1,i]
            current_ax.plot([x[1] for x in inclusive_by_triggerbin[i]], [x[4] for x in inclusive_by_triggerbin[i]], '-', label=f"Inclusive reduced chi2 statistic")
            current_ax.plot([x[1]+0.1 for x in pion_by_triggerbin[i]], [x[4] for x in pion_by_triggerbin[i]], '-', label=f"Pion reduced chi2 statistic")
            current_ax.plot([x[1]+0.2 for x in proton_by_triggerbin[i]], [x[4] for x in proton_by_triggerbin[i]], '-', label=f"Proton reduced chi2 statistic")
            current_ax.plot([x[1]+0.3 for x in kaon_by_triggerbin[i]], [x[4] for x in kaon_by_triggerbin[i]], '-', label=f"Kaon reduced chi2 statistic")
            current_ax.hlines(y=1, xmin=1, xmax=7,  color='black')
            current_ax.set_ylim(0, 2)
            current_ax.set_ylabel("Reduced Chi2 Statistic")
            current_ax.legend()
            # adjust the lower panel to be flush with the upper poanel
            plt.subplots_adjust(hspace=0.0)
            # make the lower panel 20% of the height 

            


    plt.savefig(f"RPFPlots/{parameter}ByTriggerAndHadronMomentum{analysisType.name}.png")

def main():
    conn = sqlite3.connect("RPF.db")
    c = conn.cursor()

    for ana_type in AnalysisType:
        if ana_type != AnalysisType.SEMICENTRAL:
            continue
        plotFitParametersByTriggerAndHadronMomentum(analysisType = ana_type, parameter = "background_level", expected_range=[-0.0001, 0.002], dbCursor=c, with_reduced_chi2_panel=True)
        plotFitParametersByTriggerAndHadronMomentum(analysisType = ana_type, parameter = "v2", expected_range=[-0.2, 0.2], dbCursor=c, with_reduced_chi2_panel=True)
        plotFitParametersByTriggerAndHadronMomentum(analysisType = ana_type, parameter = "v3", expected_range=[-0.1, 0.15], dbCursor=c, with_reduced_chi2_panel=True)
        plotFitParametersByTriggerAndHadronMomentum(analysisType = ana_type, parameter = "v4", expected_range=[-0.3, 0.3], dbCursor=c, with_reduced_chi2_panel=True)
        plotFitParametersByTriggerAndHadronMomentum(analysisType = ana_type, parameter = "va2", expected_range=[-0.15, 0.35], dbCursor=c, with_reduced_chi2_panel=True)
        plotFitParametersByTriggerAndHadronMomentum(analysisType = ana_type, parameter = "va4", expected_range=[-0.3, 0.35], dbCursor=c, with_reduced_chi2_panel=True)
    conn.close()

if __name__ == "__main__":
    main()