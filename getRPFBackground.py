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

customRegion = {"DeltaEta": [0.8, 1.2], "DeltaPhi": [-pi/2, 3*pi/2]}
def get_azimuthal_correlation_function(analysis):
    correlation_function = analysis.getDifferentialCorrelationFunction(True)
    mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2, customRegion=customRegion, TOF=True)
    acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(correlation_function, mixed_event_correlation_function)
    azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(acceptance_corrected_correlation_function)
    return azimuthal_correlation_function

if __name__=="__main__":
    ana_SEMICENTRAL = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/new_root/296510.root", "/mnt/d/18q/new_root/296191.root", "/mnt/d/18q/new_root/296379.root", "/mnt/d/18q/new_root/296551.root", "/mnt/d/18q/new_root/296550.root", "/mnt/d/18q/new_root/296472.root", "/mnt/d/18q/new_root/296433.root", "/mnt/d/18q/new_root/296423.root", "/mnt/d/18q/new_root/296377.root", "/mnt/d/18q/new_root/296133.root", "/mnt/d/18q/new_root/296068.root", "/mnt/d/18q/new_root/296065.root", "/mnt/d/18q/new_root/295754.root", "/mnt/d/18q/new_root/295673.root", "/mnt/d/18r/new_root/297129.root", "/mnt/d/18r/new_root/297372.root", "/mnt/d/18r/new_root/297415.root", "/mnt/d/18r/new_root/297441.root", "/mnt/d/18r/new_root/297446.root", "/mnt/d/18r/new_root/297479.root", "/mnt/d/18r/new_root/297544.root", "/mnt/d/18r/new_root/296690.root", "/mnt/d/18r/new_root/296794.root", "/mnt/d/18r/new_root/296894.root", "/mnt/d/18r/new_root/296941.root", "/mnt/d/18r/new_root/297031.root", "/mnt/d/18r/new_root/297085.root", "/mnt/d/18r/new_root/297118.root"])
    analysis = ana_SEMICENTRAL
    species = pt.PION
    region = Region.BACKGROUND
    for trig_bin in tjmb:
        for assoc_bin in [AssociatedHadronMomentumBin.PT_2_3]:
            print(f"fitting {analysis.analysisType} {trig_bin} {assoc_bin} {species}")
            analysis.setTriggerJetMomentumBin(trig_bin)
            analysis.setAssociatedHadronMomentumBin(assoc_bin)
            #analysis.setRegion(region)
            analysis.JetHadron.setDeltaEtaRange(0.8, 1.2)
            analysis.JetHadron.setDeltaPhiRange(-pi/2, 3*pi/2)
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

            background_correlation_function = analysis.getRPInclusiveBackgroundCorrelationFunctionUsingRPF(inplane_azimuthal_correlation_function, midplane_azimuthal_correlation_function, outplane_azimuthal_correlation_function, loadFunctionFromDB=False)

            inclusive_azimuthal_correlation_function.Draw()
            background_correlation_function.Draw("same")
