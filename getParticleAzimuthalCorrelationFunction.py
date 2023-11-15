from math import pi

import numpy as np
from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt
from JetHadronAnalysis.Plotting import plotTH1

if __name__=="__main__":
    loadFractionsFromDB=True
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])
    ana_SEMICENTRAl = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/296510.root"])
    for analysis in [ana_pp, ana_SEMICENTRAl]:
        for assoc_bin in AssociatedHadronMomentumBin:
            analysis.setAssociatedHadronMomentumBin(assoc_bin)
            analysis.setRegion(Region.NEAR_SIDE_SIGNAL)
            
            near_side_correlation_function = analysis.getDifferentialCorrelationFunction(True)
            near_side_mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
            near_side_acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(near_side_correlation_function, near_side_mixed_event_correlation_function)
            near_side_azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(near_side_acceptance_corrected_correlation_function)
            # now get the per species azimuthal correlation functions for each region by scaling
            near_side_pion_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.PION, near_side_azimuthal_correlation_function, loadFractionsFromDB)
  
            near_side_kaon_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.KAON, near_side_azimuthal_correlation_function, loadFractionsFromDB)
            near_side_proton_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.PROTON, near_side_azimuthal_correlation_function, loadFractionsFromDB)
            
            plotTH1(histograms=[
                near_side_azimuthal_correlation_function,
                near_side_pion_azimuthal_correlation_function,
                near_side_kaon_azimuthal_correlation_function,
                near_side_proton_azimuthal_correlation_function],
                data_labels=["inclusive", "#pi", "K", "p"],
                error_bands=None,
                            error_band_labels=None, xtitle="#Delta#phi", ytitle="1/N_{trig} dN/dDeltaphi", title=f"Per-trigger normalized azimuthal correlation function for near-side and {assoc_bin}", output_path=f"{analysis.analysisType}_{assoc_bin}_near_side.png")
            analysis.setRegion(Region.AWAY_SIDE_SIGNAL)
            
            away_side_correlation_function = analysis.getDifferentialCorrelationFunction(True)
            away_side_mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
            away_side_acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(away_side_correlation_function, away_side_mixed_event_correlation_function)
            away_side_azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(away_side_acceptance_corrected_correlation_function)
            # now get the per species azimuthal correlation functions for each region by scaling
            away_side_pion_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.PION, away_side_azimuthal_correlation_function, loadFractionsFromDB)
            away_side_kaon_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.KAON, away_side_azimuthal_correlation_function, loadFractionsFromDB)
            away_side_proton_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.PROTON, away_side_azimuthal_correlation_function, loadFractionsFromDB)
           
            plotTH1(histograms=[
                away_side_azimuthal_correlation_function,
                away_side_pion_azimuthal_correlation_function,
                away_side_kaon_azimuthal_correlation_function,
                away_side_proton_azimuthal_correlation_function],
                data_labels=["inclusive", "#pi", "K", "p"],
                error_bands=None,
                            error_band_labels=None, xtitle="#Delta#phi", ytitle="1/N_{trig} dN/dDeltaphi", title=f"Per-trigger normalized azimuthal correlation function for away-side and {assoc_bin}", output_path=f"{analysis.analysisType}_{assoc_bin}_away_side.png")
            
            analysis.setRegion(Region.BACKGROUND)
            
            background_correlation_function = analysis.getDifferentialCorrelationFunction(True)
            background_mixed_event_correlation_function = analysis.getNormalizedDifferentialMixedEventCorrelationFunction(nm.SLIDING_WINDOW, windowSize=pi/2)
            background_acceptance_corrected_correlation_function = analysis.getAcceptanceCorrectedDifferentialCorrelationFunction(background_correlation_function, background_mixed_event_correlation_function)
            background_azimuthal_correlation_function = analysis.getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(background_acceptance_corrected_correlation_function)
            # now get the per species azimuthal correlation functions for each region by scaling
            background_pion_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.PION, background_azimuthal_correlation_function, loadFractionsFromDB)
            background_kaon_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.KAON, background_azimuthal_correlation_function, loadFractionsFromDB)
            background_proton_azimuthal_correlation_function = analysis.getAzimuthalCorrelationFunctionforParticleType(pt.PROTON, background_azimuthal_correlation_function, loadFractionsFromDB)
            
            plotTH1(histograms=[
                background_azimuthal_correlation_function,
                background_pion_azimuthal_correlation_function,
                background_kaon_azimuthal_correlation_function,
                background_proton_azimuthal_correlation_function],
                data_labels=["inclusive", "#pi", "K", "p"],
                error_bands=None,
                            error_band_labels=None, xtitle="#Delta#phi", ytitle="1/N_{trig} dN/dDeltaphi", title=f"Per-trigger normalized azimuthal correlation function for background-side and {assoc_bin}", output_path=f"{analysis.analysisType}_{assoc_bin}_background.png")


        