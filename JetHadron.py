from collections import defaultdict
from itertools import product
import logging
import pickle
import subprocess
from typing import Optional, Union
import numpy as np
import requests
import matplotlib.pyplot as plt
from functools import partial
from time import time


from ROOT import (
    TH1F,
    TH2F,
    TFile,
    TH1,
)


import _JetHadronPlot, _JetHadronFit, _JetHadronAnalysis

from _JetHadronUtilities import (
    return_none,
    return_true,
    init_none_dict,
    init_bool_dict,
)
from _JetHadronLogging import (
    debug_logger,
    error_logger,
    info_logger,
    log_function_call,
)

# turn latex on in matplotlib
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# increase the font size for christine
plt.rcParams.update({"font.size": 16})


class JetHadron(
    _JetHadronPlot.PlotMixin, _JetHadronFit.FitMixin, _JetHadronAnalysis.AnalysisMixin
):
    """
    Class to track and contain all the steps along the way of my analysis for one centrality class
    """

    def __init__(
        self,
        rootFileNames: list,
        analysisType,
        fill_on_init=True,
        pickle_on_init=True,
        plot_on_init=True,
    ):
        """
        Initializes the location to save plots and grabs the main components of the rootFile
        """
        # turn off ROOT's automatic garbage collection
        TH1.AddDirectory(False)
        assert analysisType in ["central", "semicentral", "pp"]
        self.analysisType = analysisType
        self.base_save_path = (
            f"/home/steffanic/Projects/Thesis/backend_output/{analysisType}/"
        )
        # let's turn the sparses into lists of sparses to use all files
        self.JH, self.MixedEvent, self.Trigger = [], [], []
        self.EventPlaneAngleHist = []

        for filename in rootFileNames:
            debug_logger.debug(f"Loading file {filename}")
            file = TFile(filename)
            fileJH, fileME, fileT = self.get_sparses(file)
            self.JH.append(fileJH)
            self.MixedEvent.append(fileME)
            self.Trigger.append(fileT)
            if self.analysisType in ["central", "semicentral"]:
                self.EventPlaneAngleHist.append(self.get_event_plane_angle_hist(file))

        debug_logger.debug("Finished loading files")
        self.assert_sparses_filled()

        # define pT bins
        # see http://cds.cern.ch/record/1381321/plots#1 for motivation
        self.pTassocBinEdges = [
            # 0.15,
            # 0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            10.0,
        ]  # start at 500 MeV, feels like the average background level
        self.pTassocBinCenters = [
            # 0.325,
            # 0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            4.5,
            5.5,
            8.0,
        ]  # subject to change based on statistics
        self.pTassocBinWidths = [
            # 0.35,
            # 0.5,
            0.5,
            0.5,
            1,
            1,
            1,
            1,
            4,
        ]  # subject to change based on statistics
        self.pTtrigBinEdges = [
            # 10,
            20,
            40,
            #60,
        ]  # subject to change based on statistics
        self.pTtrigBinCenters = [
            # 15,
            30,
            #50,
        ]  # subject to change based on statistics
        self.pTtrigBinWidths = [
            # 10,
            20,
            #20,
        ]  # subject to change based on statistics
        self.central_p0s = {
            (i, j): [
                1,
                0.02,
                0.005,
                0.02,
                0.05,
                0.03,
            ]
            for i in range(len(self.pTtrigBinCenters))
            for j in range(len(self.pTassocBinCenters))
        }
        self.central_p0s[(0, 0)] = [
            1000042.8,
            0.0473,
            -0.000306,
            0.02,
            0.1013,
            0.03,
        ]  # pTtrig 20-40, pTassoc 1.0-1.5
        self.central_p0s[(0, 1)] = [
            40000.19,
            0.0402,
            -0.0058,
            0.02,
            0.1506,
            0.03,
        ]  # pTtrig 20-40, pTassoc 1.5-2.0
        self.central_p0s[(0, 2)] = [
            4006.86,
            0.0414,
            0.0015,
            0.02,
            0.234,
            0.03,
        ]  # pTtrig 20-40, pTassoc 2.0-3.0
        self.central_p0s[(0, 3)] = [
            56.84,
            0.0636,
            -0.00766,
            0.02,
            0.237,
            0.03,
        ]  # pTtrig 20-40, pTassoc 3.0-4.0
        self.central_p0s[(0, 4)] = [
            8.992,
            0.1721,
            -0.0987,
            0.02,
            0.233,
            0.03,
        ]  # pTtrig 20-40, pTassoc 4.0-5.0
        self.central_p0s[(0, 5)] = [
            2.318,
            -0.0508,
            -0.143,
            0.02,
            0.1876,
            0.03,
        ]  # pTtrig 20-40, pTassoc  5.0-6.0
        self.central_p0s[(0, 6)] = [
            2.076,
            -0.0886,
            0.12929,
            0.02,
            0.0692,
            0.03,
        ]  # pTtrig 20-40, pTassoc 6.0-10.0
        self.central_p0s[(1, 0)] = [
            1,
            0.02,
            0.005,
            0.02,
            0.2,
            0.03,
        ]  # pTtrig 40-60, pTassoc 1.0-1.5
        self.central_p0s[(1, 1)] = [
            1,
            0.02,
            0.005,
            0.02,
            0.2,
            0.03,
        ]  # pTtrig 40-60, pTassoc 1.5-2.0
        self.central_p0s[(1, 2)] = [
            1,
            0.02,
            0.005,
            0.02,
            0.2,
            0.03,
        ]  # pTtrig 40-60, pTassoc 2.0-3.0
        self.central_p0s[(1, 3)] = [
            1,
            0.02,
            0.005,
            0.02,
            0.2,
            0.03,
        ]  # pTtrig 40-60, pTassoc 3.0-4.0
        self.central_p0s[(1, 4)] = [
            1,
            0.02,
            0.005,
            0.02,
            0.2,
            0.03,
        ]  # pTtrig 40-60, pTassoc    4.0-5.0
        self.central_p0s[(1, 5)] = [
            1,
            0.02,
            0.005,
            0.02,
            0.2,
            0.03,
        ]  # pTtrig 40-60, pTassoc 5.0-6.0
        self.central_p0s[(1, 6)] = [
            1,
            0.02,
            0.005,
            0.02,
            0.2,
            0.03,
        ]  # pTtrig 40-60, pTassoc 6.0-10.0
        # define event plane bins
        self.eventPlaneAngleBinEdges = [0, np.pi / 6, np.pi / 3, np.pi / 2]

        self.z_vertex_bins_PbPb= [-10,-8, -6, -4, -2,  0, 2,4, 6, 8,  10]
        self.z_vertex_bins_pp= [-8, -4,  0,  4, 8]
        if self.analysisType == 'pp':
            self.z_vertex_bins = self.z_vertex_bins_pp
        else:
            self.z_vertex_bins = self.z_vertex_bins_PbPb

        self.pionID = self.get_species_id("pion")
        self.protonID = self.get_species_id("proton")
        self.kaonID = self.get_species_id("kaon")

        # 15 pt assoc bins * 7 pt trig bins * 4 event plane bins = 420 bins

        # define signal and background regions
        self.dEtaBGHi = [0.8, 1.20]
        self.dEtaBGLo = [-1.20, -0.8]
        self.dEtaSig = [-0.6, 0.6]
        self.dEtaSigAS = [-1.20, 1.20]
        self.dPhiSigNS = [-np.pi / 2, np.pi / 2]
        self.dPhiSigAS = [np.pi / 2, 3 * np.pi / 2]

        self.get_SE_correlation_function_has_changed = True
        self.get_SE_correlation_function_w_Pion_has_changed = init_bool_dict()
        self.get_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_error_has_changed = True
        self.get_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed = True
        self.get_normalized_acceptance_corrected_correlation_function_has_changed = True
        self.ME_norm_sliding_window_has_changed = True

        self.get_SE_correlation_function_result = None
        self.get_SE_correlation_function_w_Pion_result = init_none_dict()
        self.get_ME_correlation_function_result = None
        self.get_normalized_ME_correlation_function_result = None
        self.get_normalized_ME_correlation_error = None
        self.get_acceptance_corrected_correlation_function_result = None
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_result = None
        self.get_normalized_acceptance_corrected_correlation_function_result = None
        self.ME_norm_sliding_window_result = None
        self.unfoldedTruthValues = None
        self.unfoldedTruthErrors = None
        self.resetUnfoldingResults = True

        with_eventPlane = None
        if self.analysisType in ["central", "semicentral"]:
            # Define all the arrays that will hold various objects for each bin
            self.N_trigs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.eventPlaneAngleBinEdges)),
                dtype=int,
            )
            self.RPFObjs = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=object)
            self.RPFObjsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=object)
            self.RPFObjsForTrueSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=object)
            self.RPFObjsForTrueSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=object)
            self.RPFObjsForEnhancedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=object)
            self.RPFObjsForEnhancedSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=object)
            with_eventPlane = True
        elif self.analysisType == "pp":
            self.N_trigs = np.zeros(
                (len(self.pTtrigBinEdges) - 1),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=float)
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=float)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=float)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminValsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=float)
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminValsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=float)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminValsZV =  self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=float)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrsminVals = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False,dtype=float)
            self.NormalizedBGSubtracteddPhiForTrueSpeciesminVals = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
            self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminVals = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
            self.NormalizedBGSubtracteddPhiForRefoldedSpeciesminVals = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
            self.NormalizedBGSubtracteddPhiForTrueSpeciesminValsZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
            self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminValsZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
            with_eventPlane = False


        self.N_assoc =  self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=int)
        self.N_assoc_for_species = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=True, dtype=int)
        self.SEcorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH2F) 
        self.NormMEcorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH2F)
        self.AccCorrectedSEcorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH2F)
        self.AccCorrectedSEcorrsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH2F)
        self.NormAccCorrectedSEcorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH2F)
        self.NormAccCorrectedSEcorrsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH2F)
        self.ME_norm_systematics = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=float)
        self.dPhiSigcorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigcorrsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGcorrsForTrueSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigcorrsForTrueSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGcorrsForTrueSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigcorrsForTrueSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGPIDErrForTrueSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigPIDErrForTrueSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGPIDErrForTrueSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigPIDErrForTrueSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGcorrsForEnhancedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigcorrsForEnhancedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGcorrsForEnhancedSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigcorrsForEnhancedSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGPIDErrForEnhancedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigPIDErrForEnhancedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGPIDErrForEnhancedSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigPIDErrForEnhancedSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGcorrsForRefoldedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigcorrsForRefoldedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)


        self.dPhiBGcorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiBGcorrsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.dPhiSigdpionTPCnSigmacorrs = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH2F)
        self.dEtacorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=False, dtype=TH1F)
        self.PionTPCNSigmaFitObjs = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=object)
        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs =self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV =self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs =self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH2F)
        self.NormalizedBGSubtracteddPhiForTrueSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.NormalizedBGSubtracteddPhiForTrueSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.NormalizedBGSubtracteddPhiForEnhancedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.NormalizedBGSubtracteddPhiForRefoldedSpecies = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.pionTPCnSigmaInc = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=TH1F)
        self.pionTPCnSigma_pionTOFcut = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=TH1F)
        self.pionTPCnSigma_protonTOFcut = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=TH1F)
        self.pionTPCnSigma_kaonTOFcut = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=TH1F)
        self.pionTPCnSigma_otherTOFcut = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=TH1F)
        self.pionTPCnSigmaInc_vs_dphi = self.init_pTtrig_pTassoc_region_dict(with_eventPlane=with_eventPlane, with_species=False, dtype=TH1F)
        self.pionTPCnSigma_pionTOFcut_vs_dphi = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.pionTPCnSigma_protonTOFcut_vs_dphi = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)
        self.pionTPCnSigma_kaonTOFcut_vs_dphi = self.init_pTtrig_pTassoc_array(with_eventPlane=with_eventPlane, dtype=TH1F)


        self.YieldsTrue = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsTrueNS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsTrueAS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)

        self.YieldErrsTrue = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsTrueNS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsTrueAS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        
        self.YieldsTrueZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsTrueNSZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsTrueASZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)

        self.YieldErrsTrueZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsTrueNSZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsTrueASZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        
        self.YieldsEnhanced = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsEnhancedNS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsEnhancedAS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)

        self.YieldErrsEnhanced = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsEnhancedNS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsEnhancedAS = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        
        self.YieldsEnhancedZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsEnhancedNSZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldsEnhancedASZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)

        self.YieldErrsEnhancedZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsEnhancedNSZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)
        self.YieldErrsEnhancedASZV = self.init_pTtrig_pTassoc_species_dict(with_eventPlane=False, dtype=float)


        self.current_trig_bin=0
        self.current_assoc_bin=0
        if fill_on_init:
            try:
                [
                    [
                        self.fill_hist_arrays(i, j)
                        for j in range(len(self.pTassocBinEdges) - 1)
                    ]
                    for i in range(len(self.pTtrigBinEdges) - 1)
                ]
            except Exception as e:
                print("Error in fill_on_init, doing a partial pickle")
                pickle_filename = (
                    "jhAnappPartial.pickle"
                    if self.analysisType == "pp"
                    else "jhAnaCentralPartial.pickle"
                    if self.analysisType == "central"
                    else "jhAnaSemiCentralPartial.pickle"
                )
                pickle.dump(self, open(pickle_filename, "wb"))
                raise e
        if pickle_on_init:
            # pickle the object
            pickle_filename = (
                "jhAnapp.pickle"
                if self.analysisType == "pp"
                else "jhAnaCentral.pickle"
                if self.analysisType == "central"
                else "jhAnaSemiCentral.pickle"
            )
            pickle.dump(self, open(pickle_filename, "wb"))
        if plot_on_init:
            self.plot_everything()

    def pick_up_where_you_left_off(self):
        try:
            [
                [
                    self.fill_hist_arrays(i, j)
                    for j in range(self.current_assoc_bin, len(self.pTassocBinEdges) - 1)
                ]
                for i in range(self.current_trig_bin, len(self.pTtrigBinEdges) - 1)
            ]
        except Exception as e:
                print("Error in fill_on_init, doing a partial pickle")
                pickle_filename = (
                    "jhAnappPartial.pickle"
                    if self.analysisType == "pp"
                    else "jhAnaCentralPartial.pickle"
                    if self.analysisType == "central"
                    else "jhAnaSemiCentralPartial.pickle"
                )
                pickle.dump(self, open(pickle_filename, "wb"))
                raise e

    def init_pTtrig_pTassoc_region_dict(self, with_eventPlane, with_species, dtype):
        if with_species:
            return defaultdict(partial(self.init_pTtrig_pTassoc_species_dict, with_eventPlane=with_eventPlane, dtype=dtype))
        else:
            return defaultdict(partial(self.init_pTtrig_pTassoc_array, with_eventPlane=with_eventPlane, dtype=dtype))

    def init_pTtrig_pTassoc_species_dict(self, with_eventPlane, dtype):
        return defaultdict(partial(self.init_pTtrig_pTassoc_array, with_eventPlane=with_eventPlane, dtype=dtype))

    def init_pTtrig_pTassoc_array(self, with_eventPlane:Union[bool, None]=False, dtype=object):
        '''
        Initialize an array with zeros for each pTtrig, pTassoc bin
        '''
        if with_eventPlane is None:
            print("You silly goose! with_eventPlane is None, but it should be True or False! I'm going to throw a ValueError!")
            raise ValueError("with_eventPlane is None, but it should be True or False!")
        shape = (
            len(self.pTtrigBinEdges) - 1,
            len(self.pTassocBinEdges) - 1,
        )
        if with_eventPlane:
            shape += (len(self.eventPlaneAngleBinEdges),)
        return np.zeros(
            shape, dtype=dtype
        )

    @log_function_call(description="")
    def fill_hist_arrays(self, i, j, hists_to_fill: "Optional[dict[str, bool]]" = None):
        self.current_trig_bin=i
        self.current_assoc_bin=j
        if self.analysisType in ["central", "semicentral"]:
            self.set_pT_epAngle_bin(i, j, 3)
            for region in ['NS', 'AS', 'BG']:
                self.fill_N_assoc(i, j,3, region)
            
            for k in range(len(self.eventPlaneAngleBinEdges) - 1):
                print(
                    f"Getting correlation function for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}"
                )
                # set the ranges in the sparses
                self.set_pT_epAngle_bin(i, j, k)
                self.assert_sparses_filled()
                self.epString = (
                    "out-of-plane"
                    if k == 2
                    else (
                        "mid-plane"
                        if k == 1
                        else ("in-plane" if k == 0 else "inclusive")
                    )
                )
                # fill the N_trigs array
                if hists_to_fill is None or hists_to_fill.get("Ntrigs"):
                    self.fill_N_trigs(i, j, k)
                    for region in ['NS', 'AS', 'BG']:
                        self.fill_N_assoc(i, j, k, region)
                for species in ['pion', 'proton', 'kaon']:
                    for region in ['NS', 'AS', 'BG']:
                        self.fill_N_assoc_for_species(i, j, k, species, region)
                # get the SE correlation function
                if hists_to_fill is None or hists_to_fill.get("SE"):
                    self.fill_SE_correlation_function(i, j, k)
                    self.plot_SE_correlation_function(i, j, k)

                # get the ME correlation function
                if hists_to_fill is None or hists_to_fill.get("ME"):
                    self.fill_ME_correlation_function(i, j, k)
                    self.plot_ME_correlation_function(i, j, k)

                # get the acceptance corrected SE correlation function
                if hists_to_fill is None or hists_to_fill.get("AccCorrectedSE"):
                    self.fill_AccCorrected_SE_correlation_function(i, j, k)
                    self.plot_acceptance_corrected_SE_correlation_function(i, j, k)
                
                # get the acceptance corrected SE correlation function
                if hists_to_fill is None or hists_to_fill.get("AccCorrectedSEZV"):
                    self.fill_AccCorrected_SE_correlation_function_in_z_vertex_bins(i, j, k)
                    self.plot_acceptance_corrected_SE_correlation_function_in_z_vertex_bins(i, j, k)

                # Get the number of triggers to normalize
                if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
                    self.fill_NormAccCorrected_SE_correlation_function(i, j, k)
                    self.plot_normalized_acceptance_corrected_correlation_function(
                        i, j, k
                    )
                
                # Get the number of triggers to normalize
                if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSEZV"):
                    self.fill_NormAccCorrected_SE_correlation_function_in_z_vertex_bins(i, j, k)
                    self.plot_normalized_acceptance_corrected_correlation_function_in_z_vertex_bins(
                        i, j, k
                    )

                if hists_to_fill is None or hists_to_fill.get("dPhi"):
                    self.fill_dPhi_correlation_functions(i, j, k)
        
                
                if hists_to_fill is None or hists_to_fill.get("dPhiZV"):
                    self.fill_dPhi_correlation_functions_in_z_vertex_bins(i, j, k)

                for region in ["NS", "AS", "BG"]:
                    pionTPCnSigma = self.get_pion_TPC_nSigma_inclusive(i,j,k,region=region) if region!='BG' else self.get_pion_TPC_nSigma_inclusive(i,j,k,region="BGHi")
                    if region=='BG':
                        pionTPCnSigma.Add(self.get_pion_TPC_nSigma_inclusive(i,j,k,region="BGLo"))
                    self.pionTPCnSigmaInc[region][i, j,k] = pionTPCnSigma
                    del pionTPCnSigma

                    pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma(i,j,k,"pion", region=region) if region!='BG' else self.get_pion_TPC_nSigma(i,j,k,"pion", region="BGHi")
                    if region=='BG':
                        pionTPCnSigma_pionTOFcut.Add(self.get_pion_TPC_nSigma(i,j,k,"pion", region="BGLo"))
                    self.pionTPCnSigma_pionTOFcut[region][i, j, k] = pionTPCnSigma_pionTOFcut
                    del pionTPCnSigma_pionTOFcut

                    pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma(i,j,k,"kaon", region=region) if region!='BG' else self.get_pion_TPC_nSigma(i,j,k,"kaon", region="BGHi")
                    if region=='BG':
                        pionTPCnSigma_kaonTOFcut.Add(self.get_pion_TPC_nSigma(i,j,k,"kaon", region="BGLo"))
                    self.pionTPCnSigma_kaonTOFcut[region][i, j, k] = pionTPCnSigma_kaonTOFcut
                    del pionTPCnSigma_kaonTOFcut

                    pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma(i,j,k,"proton", region=region) if region!='BG' else self.get_pion_TPC_nSigma(i,j,k,"proton", region="BGHi")
                    if region=='BG':
                        pionTPCnSigma_protonTOFcut.Add(self.get_pion_TPC_nSigma(i,j,k,"proton", region="BGLo"))
                    self.pionTPCnSigma_protonTOFcut[region][i, j, k] = pionTPCnSigma_protonTOFcut
                    del pionTPCnSigma_protonTOFcut
                    
                    pionTPCnSigma_otherTOFcut = self.get_pion_TPC_nSigma(i,j,k,"other", region=region) if region!='BG' else self.get_pion_TPC_nSigma(i,j,k,"other", region="BGHi")
                    if region=='BG':
                        pionTPCnSigma_otherTOFcut.Add(self.get_pion_TPC_nSigma(i,j,k,"other", region="BGLo"))
                    self.pionTPCnSigma_otherTOFcut[region][i, j, k] = pionTPCnSigma_otherTOFcut
                    del pionTPCnSigma_otherTOFcut
                    
#                     pionTPCnSigma = self.get_pion_TPC_nSigma_inclusive_vs_dphi()
#                     self.pionTPCnSigmaInc_vs_dphi[i, j,k] = pionTPCnSigma
#                     del pionTPCnSigma
# 
#                     pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma_vs_dphi("pion")
#                     self.pionTPCnSigma_pionTOFcut_vs_dphi[i, j, k] = pionTPCnSigma_pionTOFcut
#                     del pionTPCnSigma_pionTOFcut
# 
#                     pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma_vs_dphi("kaon")
#                     self.pionTPCnSigma_kaonTOFcut_vs_dphi[i, j, k] = pionTPCnSigma_kaonTOFcut
#                     del pionTPCnSigma_kaonTOFcut
# 
#                     pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma_vs_dphi("proton")
#                     self.pionTPCnSigma_protonTOFcut_vs_dphi[i, j, k] = pionTPCnSigma_protonTOFcut
#                     del pionTPCnSigma_protonTOFcut

                self.fit_PionTPCNSigma(i, j, k, "NS")
                self.fit_PionTPCNSigma(i, j, k, "AS")
                self.fit_PionTPCNSigma(i, j, k, "BG")
                self.plot_PionTPCNSigmaFit(i,j, k, "NS")
                self.plot_PionTPCNSigmaFit(i,j, k, "AS")
                self.plot_PionTPCNSigmaFit(i,j, k, "BG")
                # self.plot_PionTPCNSigmaVsDphi(i,j, k)

                for species in ["pion", "proton", "kaon"]:
                    self.fill_dPhi_correlation_functions_for_true_species(i, j, k, species)
                    self.fill_dPhi_correlation_functions_for_true_species_in_z_vertex_bins(i, j, k, species)
                    self.fill_dPhi_correlation_functions_for_enhanced_species(i, j, k, species)
                    self.fill_dPhi_correlation_functions_for_enhanced_species_in_z_vertex_bins(i, j, k, species)
            self.fill_N_trigs(i, j, 3)
            self.fit_RPF(i, j, p0=self.central_p0s[(i, j)])
            self.plot_RPF(i, j, withSignal=True)
            self.fit_RPF_in_z_vertex_bins(i, j, p0=self.central_p0s[(i, j)])
            self.plot_RPF_in_z_vertex_bins(i, j, withSignal=True)
            for species in ["pion", "proton", "kaon"]:
                self.fit_RPF_for_true_species(i, j, species, p0=self.central_p0s[(i, j)])
                self.fit_RPF_for_true_species_in_z_vertex_bins(i, j, species, p0=self.central_p0s[(i, j)])
                self.fit_RPF_for_enhanced_species(i, j, species, p0=self.central_p0s[(i, j)])
                self.fit_RPF_for_enhanced_species_in_z_vertex_bins(i, j, species, p0=self.central_p0s[(i, j)])
              
            if j == len(self.pTassocBinCenters):
                self.plot_optimal_parameters(i)
                self.plot_optimal_parameters_in_z_vertex_bins(i)
                for species in ["pion", "proton", "kaon"]:
                    self.plot_optimal_parameters_for_true_species(i, species)
                    self.plot_optimal_parameters_for_true_species_in_z_vertex_bins(i, species)
                    self.plot_optimal_parameters_for_enhanced_species(i, species)
                    self.plot_optimal_parameters_for_enhanced_species_in_z_vertex_bins(i, species)
            jsonpayload = {
                "value1": f"RPF fitted for p_Ttrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, p_Tassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV"
            }
            url = "https://maker.ifttt.com/trigger/code_finished/with/key/bFQ9TznlbsocL7hbBL_sDyk33qkIJdDNVSIjhyJ7Mqm"
            try:
                requests.post(url, json=jsonpayload)
            except:
                error_logger.error("Status notification failed, oh well")

            # get the background subtracted correlation function
            for k in range(len(self.eventPlaneAngleBinEdges) - 1):
                self.set_pT_epAngle_bin(i, j, k)
                self.assert_sparses_filled()
                self.fill_BG_subtracted_AccCorrected_SE_correlation_functions(i, j, k)
                self.fill_BG_subtracted_AccCorrected_SE_correlation_functions_in_z_vertex_bins(i, j, k)
                self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(
                    i, j, k
                )
                self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_in_z_vertex_bins(
                    i, j, k
                )
                self.plot_normalized_acceptance_corrected_correlation_function(i,j,k)
                self.plot_normalized_acceptance_corrected_correlation_function_in_z_vertex_bins(i,j,k)
                
                for species in ["pion", "proton", "kaon"]:
                    
                    self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
                        i, j, k, species
                    )
                    self.fill_normalized_background_subtracted_dPhi_for_true_species(
                        i, j, k, species
                    )
                    self.fill_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                        i, j, k, species
                    )
                    self.fill_normalized_background_subtracted_dPhi_for_enhanced_species(
                        i, j, k, species
                    )
                    self.fill_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(
                        i, j, k, species
                    )
                    debug_logger.debug(
                        f"Inclusive yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}: {self.get_yield_for_true_species(i,j,k,species)}"
                    )
                    debug_logger.debug(
                        f"NS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}: {self.get_yield_for_true_species(i,j,k,species, 'NS')}"
                    )
                    debug_logger.debug(
                        f"AS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}: {self.get_yield_for_true_species(i,j,k,species, 'AS')}"
                    )
            

        elif self.analysisType == "pp":
            print(
                f"Getting correlation function for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV"
            )
            # set the ranges in the sparses
            self.set_pT_epAngle_bin(i, j, 3)
            self.assert_sparses_filled()
            self.epString = "inclusive"
        if self.analysisType in ["central", "semicentral"]:
            for sparse_ind in range(len(self.JH)):
                # then grab the inclusive event plane angle bin
                self.JH[sparse_ind].GetAxis(5).SetRangeUser(
                    self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[-1]
                )
                self.MixedEvent[sparse_ind].GetAxis(5).SetRangeUser(
                    self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[-1]
                )
                self.Trigger[sparse_ind].GetAxis(2).SetRangeUser(
                    self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[-1]
                )
        self.set_has_changed()
        self.assert_sparses_filled()
        # fill the N_trigs array
        if hists_to_fill is None or hists_to_fill.get("Ntrigs"):
            self.fill_N_trigs(i, j, 3)
            for region in ['NS', 'AS', 'BG']:
                self.fill_N_assoc(i, j, 3, region)
        for species in ['pion', 'proton', 'kaon']:
            for region in ["NS", "AS", "BG"]:
                self.fill_N_assoc_for_species(i, j, 3, species, region)

        # get the SE correlation function
        if hists_to_fill is None or hists_to_fill.get("SE"):
            self.fill_SE_correlation_function(i, j, 3)
            self.plot_SE_correlation_function(i, j, 3)

        # get the ME correlation function
        if hists_to_fill is None or hists_to_fill.get("ME"):
            self.fill_ME_correlation_function(i, j, 3)
            self.plot_ME_correlation_function(i, j, 3)

        # get the acceptance corrected SE correlation function
        if hists_to_fill is None or hists_to_fill.get("AccCorrectedSE"):
            self.fill_AccCorrected_SE_correlation_function(i, j, 3)
            self.plot_acceptance_corrected_SE_correlation_function(i, j, 3)
        
        # get the acceptance corrected SE correlation function
        if hists_to_fill is None or hists_to_fill.get("AccCorrectedSEZV"):
            self.fill_AccCorrected_SE_correlation_function_in_z_vertex_bins(i, j, 3)
            self.plot_acceptance_corrected_SE_correlation_function_in_z_vertex_bins(i, j, 3)

        # Get the number of triggers to normalize
        if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
            self.fill_NormAccCorrected_SE_correlation_function(i, j, 3)
            self.plot_normalized_acceptance_corrected_correlation_function(i, j, 3)
        
        # Get the number of triggers to normalize
        if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSEZV"):
            self.fill_NormAccCorrected_SE_correlation_function_in_z_vertex_bins(i, j, 3)
            self.plot_normalized_acceptance_corrected_correlation_function_in_z_vertex_bins(i, j, 3)

        if hists_to_fill is None or hists_to_fill.get("dPhi"):
            self.fill_dPhi_correlation_functions(i, j, 3)

        if hists_to_fill is None or hists_to_fill.get("dPhi"):
            self.fill_dPhi_correlation_functions_in_z_vertex_bins(i, j, 3)
        dEta = self.get_dEta_projection_NS()
        self.dEtacorrs[i, j] = dEta
        del dEta

        for region in ["NS", "AS", "BG"]:
            if self.analysisType in ["central", "semicentral"]:
                pionTPCnSigma = self.get_pion_TPC_nSigma_inclusive(i,j,3,region=region) if region != "BG" else self.get_pion_TPC_nSigma_inclusive(i,j,3,region="BGHi")
                if region=='BG':
                    pionTPCnSigma.Add(self.get_pion_TPC_nSigma_inclusive(i,j,3,region="BGLo"))
                self.pionTPCnSigmaInc[region][i, j,3] = pionTPCnSigma
                del pionTPCnSigma

                pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma(i,j,3,"pion", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"pion", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_pionTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"pion", region="BGLo"))
                self.pionTPCnSigma_pionTOFcut[region][i, j,3] = pionTPCnSigma_pionTOFcut
                del pionTPCnSigma_pionTOFcut

                pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma(i,j,3,"kaon", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"kaon", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_kaonTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"kaon", region="BGLo"))
                self.pionTPCnSigma_kaonTOFcut[region][i, j,3] = pionTPCnSigma_kaonTOFcut
                del pionTPCnSigma_kaonTOFcut

                pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma(i,j,3,"proton", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"proton", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_protonTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"proton", region="BGLo"))
                self.pionTPCnSigma_protonTOFcut[region][i, j,3] = pionTPCnSigma_protonTOFcut
                del pionTPCnSigma_protonTOFcut
                
                pionTPCnSigma_otherTOFcut = self.get_pion_TPC_nSigma(i,j,3,"other", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"other", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_otherTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"other", region="BGLo"))
                self.pionTPCnSigma_otherTOFcut[region][i, j,3] = pionTPCnSigma_otherTOFcut
                del pionTPCnSigma_otherTOFcut
            else:
                pionTPCnSigma = self.get_pion_TPC_nSigma_inclusive(i,j,3, region=region) if region != "BG" else self.get_pion_TPC_nSigma_inclusive(i,j,3,region="BGHi")
                if region=='BG':
                    pionTPCnSigma.Add(self.get_pion_TPC_nSigma_inclusive(i,j,3,region="BGLo"))
                self.pionTPCnSigmaInc[region][i, j] = pionTPCnSigma
                del pionTPCnSigma

                pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma(i,j,3,"pion", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"pion", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_pionTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"pion", region="BGLo"))
                self.pionTPCnSigma_pionTOFcut[region][i, j] = pionTPCnSigma_pionTOFcut
                del pionTPCnSigma_pionTOFcut

                pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma(i,j,3,"kaon", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"kaon", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_kaonTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"kaon", region="BGLo"))
                self.pionTPCnSigma_kaonTOFcut[region][i, j] = pionTPCnSigma_kaonTOFcut
                del pionTPCnSigma_kaonTOFcut

                pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma(i,j,3,"proton", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"proton", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_protonTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"proton", region="BGLo"))
                self.pionTPCnSigma_protonTOFcut[region][i, j] = pionTPCnSigma_protonTOFcut
                del pionTPCnSigma_protonTOFcut
                
                pionTPCnSigma_otherTOFcut = self.get_pion_TPC_nSigma(i,j,3,"other", region=region) if region != "BG" else self.get_pion_TPC_nSigma(i,j,3,"other", region="BGHi")
                if region=='BG':
                    pionTPCnSigma_otherTOFcut.Add(self.get_pion_TPC_nSigma(i,j,3,"other", region="BGLo"))
                self.pionTPCnSigma_otherTOFcut[region][i, j] = pionTPCnSigma_otherTOFcut
                del pionTPCnSigma_otherTOFcut
        
#         if self.analysisType in ["central", "semicentral"]:
#             pionTPCnSigma = self.get_pion_TPC_nSigma_inclusive_vs_dphi()
#             self.pionTPCnSigmaInc_vs_dphi[i, j,3] = pionTPCnSigma
#             del pionTPCnSigma
# 
#             pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma_vs_dphi("pion")
#             self.pionTPCnSigma_pionTOFcut_vs_dphi[i, j,3] = pionTPCnSigma_pionTOFcut
#             del pionTPCnSigma_pionTOFcut
# 
#             pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma_vs_dphi("kaon")
#             self.pionTPCnSigma_kaonTOFcut_vs_dphi[i, j,3] = pionTPCnSigma_kaonTOFcut
#             del pionTPCnSigma_kaonTOFcut
# 
#             pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma_vs_dphi("proton")
#             self.pionTPCnSigma_protonTOFcut_vs_dphi[i, j,3] = pionTPCnSigma_protonTOFcut
#             del pionTPCnSigma_protonTOFcut
#         else:
#             pionTPCnSigma = self.get_pion_TPC_nSigma_inclusive_vs_dphi()
#             self.pionTPCnSigmaInc_vs_dphi[i, j] = pionTPCnSigma
#             del pionTPCnSigma
# 
#             pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma_vs_dphi("pion")
#             self.pionTPCnSigma_pionTOFcut_vs_dphi[i, j] = pionTPCnSigma_pionTOFcut
#             del pionTPCnSigma_pionTOFcut
# 
#             pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma_vs_dphi("kaon")
#             self.pionTPCnSigma_kaonTOFcut_vs_dphi[i, j] = pionTPCnSigma_kaonTOFcut
#             del pionTPCnSigma_kaonTOFcut
# 
#             pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma_vs_dphi("proton")
#             self.pionTPCnSigma_protonTOFcut_vs_dphi[i, j] = pionTPCnSigma_protonTOFcut
#             del pionTPCnSigma_protonTOFcut

        self.fit_PionTPCNSigma(i, j, 3, "NS")
        self.fit_PionTPCNSigma(i, j, 3, "AS")
        self.fit_PionTPCNSigma(i, j, 3, "BG")
        self.plot_PionTPCNSigmaFit(i, j, 3, "NS")
        self.plot_PionTPCNSigmaFit(i, j, 3, "AS")
        self.plot_PionTPCNSigmaFit(i, j, 3, "BG")
       

        
        self.fill_inclusive_yield(i, j)

        # get the background subtracted correlation functions
        if hists_to_fill is None or hists_to_fill.get("BGSubtractedSE"):
            self.fill_BG_subtracted_AccCorrected_SE_correlation_functions(i, j, 3)
            self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(
                i, j, 3
            )
            
            self.fill_BG_subtracted_AccCorrected_SE_correlation_functions_in_z_vertex_bins(i, j, 3)
            self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_in_z_vertex_bins(
                i, j, 3
            )
            for species in ["pion", "proton", "kaon"]:
                self.fill_dPhi_correlation_functions_for_true_species(i, j, 3, species)
                self.fill_dPhi_correlation_functions_for_enhanced_species(i, j, 3, species)
                
                self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
                    i, j, 3, species
                )
                self.fill_normalized_background_subtracted_dPhi_for_true_species(
                    i, j, 3, species
                )
                self.fill_normalized_background_subtracted_dPhi_for_enhanced_species(
                    i, j, 3, species
                )
                self.fill_dPhi_correlation_functions_for_true_species_in_z_vertex_bins(i, j, 3, species)
                self.fill_dPhi_correlation_functions_for_enhanced_species_in_z_vertex_bins(i, j, 3, species)
                self.fill_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                    i, j, 3, species
                )
                self.fill_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(
                    i, j, 3, species
                )
                self.fill_yield_for_true_species(i, j, species)
                self.fill_yield_for_true_species_in_z_vertex_bins(i, j, species)
                self.fill_yield_for_enhanced_species(i, j, species)
                self.fill_yield_for_enhanced_species_in_z_vertex_bins(i, j, species)
                
                debug_logger.debug(
                    f"Inclusive yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle inclusive: {self.get_yield_for_true_species(i,j,3,species)}"
                )
                debug_logger.debug(
                    f"NS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle inclusive: {self.get_yield_for_true_species(i,j,3,species, 'NS')}"
                )
                debug_logger.debug(
                    f"AS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle inclusive: {self.get_yield_for_true_species(i,j,3,species, 'AS')}"
                )

        print("\a")
        p = subprocess.Popen(
            [
                "bash",
                "-c",
                f'. ~/.notifyme; notifyme "Finished with p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV";',
            ]
        )
        p.kill()

    @log_function_call(description="")
    def fill_N_trigs(self, i, j, k):
        Ntrigs = self.get_N_trig()
        if self.analysisType in ["central", "semicentral"]:
            self.N_trigs[i, k] = Ntrigs
        elif self.analysisType == "pp":
            self.N_trigs[i] = Ntrigs
        del Ntrigs

    @log_function_call(description="")
    def fill_N_assoc(self, i, j, k, region):
        if region == "BG":
            Nassoc = self.get_N_assoc('BGHi')+self.get_N_assoc('BGLo')
        else:
            Nassoc = self.get_N_assoc(region)

        if self.analysisType in ["central", "semicentral"]:
            self.N_assoc[region][i, j, k] = Nassoc
        elif self.analysisType == "pp":
            self.N_assoc[region][i, j] = Nassoc
        del Nassoc

    @log_function_call(description="")
    def fill_N_assoc_for_species(self, i, j, k, species, region):
        if region == "BG":
            Nassoc = self.get_N_assoc_for_species(i,j,k,species, "BGHi")+self.get_N_assoc_for_species(i,j,k,species, "BGLo")
        else:
            Nassoc = self.get_N_assoc_for_species(i,j,k,species, region)

        if self.analysisType in ["central", "semicentral"]:
            self.N_assoc_for_species[region][species][i, j, k] = Nassoc
        elif self.analysisType == "pp":
            self.N_assoc_for_species[region][species][i, j] = Nassoc
        del Nassoc

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_SE_correlation_function(self, i, j, k):
        SEcorr = self.get_SE_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.SEcorrs[i, j, k] = SEcorr
        elif self.analysisType == "pp":
            self.SEcorrs[i, j] = SEcorr
        del SEcorr

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_ME_correlation_function(self, i, j, k):
        if j >= 2:
            self.set_pT_assoc_range(2, 7)
            self.set_pT_trig_range(0, 2)
            NormMEcorr, ME_norm_error = self.get_normalized_ME_correlation_function()
            self.set_pT_epAngle_bin(i, j, k)
        else:
            NormMEcorr, ME_norm_error = self.get_normalized_ME_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.NormMEcorrs[i, j, k] = NormMEcorr
            self.ME_norm_systematics[i, j, k] = ME_norm_error
        elif self.analysisType == "pp":
            self.NormMEcorrs[i, j] = NormMEcorr
            self.ME_norm_systematics[i, j] = ME_norm_error
        del NormMEcorr

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_AccCorrected_SE_correlation_function(self, i, j, k):
        AccCorrectedSEcorr = self.get_acceptance_corrected_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.AccCorrectedSEcorrs[i, j, k] = AccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.AccCorrectedSEcorrs[i, j] = AccCorrectedSEcorr
        del AccCorrectedSEcorr

    @log_function_call(description="")
    def fill_AccCorrected_SE_correlation_function_in_z_vertex_bins(self, i, j, k):
        AccCorrectedSEcorr = self.get_acceptance_corrected_correlation_function(in_z_vertex_bins=True)
        if self.analysisType in ["central", "semicentral"]:
            self.AccCorrectedSEcorrsZV[i, j, k] = AccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.AccCorrectedSEcorrsZV[i, j] = AccCorrectedSEcorr
        del AccCorrectedSEcorr

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_NormAccCorrected_SE_correlation_function(self, i, j, k):
        NormAccCorrectedSEcorr = (
            self.get_normalized_acceptance_corrected_correlation_function()
        )
        if self.analysisType in ["central", "semicentral"]:
            self.NormAccCorrectedSEcorrs[i, j, k] = NormAccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.NormAccCorrectedSEcorrs[i, j] = NormAccCorrectedSEcorr
        del NormAccCorrectedSEcorr

    @log_function_call(description="")
    def fill_NormAccCorrected_SE_correlation_function_in_z_vertex_bins(self, i, j, k):
        NormAccCorrectedSEcorr = (
            self.get_normalized_acceptance_corrected_correlation_function(in_z_vertex_bins=True)
        )
        if self.analysisType in ["central", "semicentral"]:
            self.NormAccCorrectedSEcorrsZV[i, j, k] = NormAccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.NormAccCorrectedSEcorrsZV[i, j] = NormAccCorrectedSEcorr
        del NormAccCorrectedSEcorr

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_dPhi_correlation_functions(self, i, j, k):
        dPhiBGHi, nbins_hi, binwidth = self.get_dPhi_projection_in_dEta_range(
            self.dEtaBGHi
        )
        # print(f"dPhiBGHi is at {hex(id(dPhiBGHi))}")
        dPhiBGLo, nbins_lo, binwidth = self.get_dPhi_projection_in_dEta_range(
            self.dEtaBGLo
        )
        # print(f"dPhiBGLo is at {hex(id(dPhiBGLo))}")

        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiMid, nbins_mid, binwidth = self.get_dPhi_projection_in_dEta_range(
            self.dEtaSig,
        )

        debug_logger.debug(f"Max of dPhiBGHi is {dPhiBGHi.GetMaximum()}")
        debug_logger.debug(f"Max of dPhiBGLo is {dPhiBGLo.GetMaximum()}")
        debug_logger.debug(f"Max of dPhiMid is {dPhiMid.GetMaximum()}")
        debug_logger.debug(f"Number of bins in dPhiBGHi is {nbins_hi}")
        debug_logger.debug(f"Number of bins in dPhiBGLo is {nbins_lo}")
        debug_logger.debug(f"Number of bins in dPhiMid is {nbins_mid}")
        # now lets make a new histogram with the same binning as dPhiBG
        # but with the values of dPhiBG for  bins with dPhi<pi/2 and the sum of the values of dPhiBG and dPhiSig for bins with dPhi>pi/2
        dPhiBGcorrs = dPhiMid.Clone()
        dPhiSig = dPhiMid.Clone()
        dPhiBGcorrs.Reset()
        dPhiSig.Reset()
        for iBin in range(1, dPhiBGcorrs.GetNbinsX() + 1):
            if dPhiBGcorrs.GetBinCenter(iBin + 1) < np.pi / 2:
                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                    )
                    /2
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )
                dPhiSig.SetBinContent(
                    iBin, dPhiMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSig.SetBinError(
                    iBin, dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )
            else:
                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )
                dPhiSig.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSig.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrs[i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrs[i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiBGcorrs[i, j] = dPhiBGcorrs
            self.dPhiSigcorrs[i, j] = dPhiSig
        del dPhiBGHi, dPhiBGLo, dPhiBGcorrs, dPhiSig
    
    @log_function_call(description="")
    def fill_dPhi_correlation_functions_in_z_vertex_bins(self, i, j, k):
        dPhiBGHi, nbins_hi, binwidth = self.get_dPhi_projection_in_dEta_range(
            self.dEtaBGHi, in_z_vertex_bins=True
        )
        # print(f"dPhiBGHi is at {hex(id(dPhiBGHi))}")
        dPhiBGLo, nbins_lo, binwidth = self.get_dPhi_projection_in_dEta_range(
            self.dEtaBGLo, in_z_vertex_bins=True
        )
        # print(f"dPhiBGLo is at {hex(id(dPhiBGLo))}")

        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiMid, nbins_mid, binwidth = self.get_dPhi_projection_in_dEta_range(
            self.dEtaSig, in_z_vertex_bins=True
        )

        debug_logger.debug(f"Max of dPhiBGHi is {dPhiBGHi.GetMaximum()}")
        debug_logger.debug(f"Max of dPhiBGLo is {dPhiBGLo.GetMaximum()}")
        debug_logger.debug(f"Max of dPhiMid is {dPhiMid.GetMaximum()}")
        debug_logger.debug(f"Number of bins in dPhiBGHi is {nbins_hi}")
        debug_logger.debug(f"Number of bins in dPhiBGLo is {nbins_lo}")
        debug_logger.debug(f"Number of bins in dPhiMid is {nbins_mid}")
        # now lets make a new histogram with the same binning as dPhiBG
        # but with the values of dPhiBG for  bins with dPhi<pi/2 and the sum of the values of dPhiBG and dPhiSig for bins with dPhi>pi/2
        dPhiBGcorrs = dPhiMid.Clone()
        dPhiSig = dPhiMid.Clone()
        dPhiBGcorrs.Reset()
        dPhiSig.Reset()
        for iBin in range(1, dPhiBGcorrs.GetNbinsX() + 1):
            if dPhiBGcorrs.GetBinCenter(iBin + 1) < np.pi / 2:
                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                    )
                    /2
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )
                dPhiSig.SetBinContent(
                    iBin, dPhiMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSig.SetBinError(
                    iBin, dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )
            else:
                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )
                dPhiSig.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSig.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrsZV[i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsZV[i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiBGcorrsZV[i, j] = dPhiBGcorrs
            self.dPhiSigcorrsZV[i, j] = dPhiSig
        del dPhiBGHi, dPhiBGLo, dPhiBGcorrs, dPhiSig

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_dPhi_correlation_functions_for_true_species(self, i, j, k, species):

        dPhiMid, nbins_mid, binwidth, dPhiPIDerrMid = self.get_dPhi_projection_in_dEta_range_for_true_species(i,j,k,
            self.dEtaSig, TOFcutSpecies=species
        )

        dPhiBGLo, nbins_lo, binwidth, dPhiPIDerrLo = self.get_dPhi_projection_in_dEta_range_for_true_species(i,j,k,
            self.dEtaBGLo, TOFcutSpecies=species
        )

        dPhiBGHi, nbins_hi, binwidth, dPhiPIDerrHi = self.get_dPhi_projection_in_dEta_range_for_true_species(i,j,k,
            self.dEtaBGHi, TOFcutSpecies=species
        )

        dPhiBGcorrs = dPhiMid.Clone()
        dPhiSig = dPhiMid.Clone()
        dPhiBGcorrs.Reset()
        dPhiSig.Reset()

        dPhiBGPIDerr = dPhiPIDerrMid.Clone()
        dPhiSigPIDerr = dPhiPIDerrMid.Clone()
        dPhiBGPIDerr.Reset()
        dPhiSigPIDerr.Reset()

        for iBin in range(1, dPhiBGcorrs.GetNbinsX() + 1):
            if dPhiBGcorrs.GetBinCenter(iBin + 1) < np.pi / 2:
                # Near Side
                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth))
                    /2
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin, dPhiMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSig.SetBinError(
                    iBin, dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )

                dPhiBGPIDerr.SetBinContent(
                    iBin,
                    (dPhiPIDerrLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiPIDerrHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth))
                    /2
                )
                dPhiBGPIDerr.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiPIDerrLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiPIDerrHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )

                dPhiSigPIDerr.SetBinContent(
                    iBin, dPhiPIDerrMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSigPIDerr.SetBinError(
                    iBin, dPhiPIDerrMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )

            else:
                # Away Side

                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSig.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiBGPIDerr.SetBinContent(
                    iBin,
                    (
                        dPhiPIDerrLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiPIDerrHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiPIDerrMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGPIDerr.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiPIDerrHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiPIDerrLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiPIDerrMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiSigPIDerr.SetBinContent(
                    iBin,
                    (
                        dPhiPIDerrLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiPIDerrHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiPIDerrMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSigPIDerr.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiPIDerrHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiPIDerrLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiPIDerrMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrsForTrueSpecies[species][i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsForTrueSpecies[species][i, j, k] = dPhiSig
            self.dPhiBGPIDErrForTrueSpecies[species][i, j, k] = dPhiBGPIDerr
            self.dPhiSigPIDErrForTrueSpecies[species][i, j, k] = dPhiSigPIDerr
        elif self.analysisType == "pp":
            self.dPhiBGcorrsForTrueSpecies[species][i, j] = dPhiBGcorrs
            self.dPhiSigcorrsForTrueSpecies[species][i, j] = dPhiSig
            self.dPhiBGPIDErrForTrueSpecies[species][i, j] = dPhiBGPIDerr
            self.dPhiSigPIDErrForTrueSpecies[species][i, j] = dPhiSigPIDerr
        del dPhiSig
    
    @log_function_call(description="", logging_level=logging.INFO)
    def fill_dPhi_correlation_functions_for_refolded_species(self, i, j, k, species):

        dPhiMid, nbins_mid, binwidth,  = self.get_dPhi_projection_in_dEta_range_for_refolded_species(i,j,k,
            self.dEtaSig, TOFcutSpecies=species
        )

        dPhiBGLo, nbins_lo, binwidth,  = self.get_dPhi_projection_in_dEta_range_for_refolded_species(i,j,k,
            self.dEtaBGLo, TOFcutSpecies=species
        )

        dPhiBGHi, nbins_hi, binwidth,  = self.get_dPhi_projection_in_dEta_range_for_refolded_species(i,j,k,
            self.dEtaBGHi, TOFcutSpecies=species
        )

        dPhiBGcorrs = dPhiMid.Clone()
        dPhiSig = dPhiMid.Clone()
        dPhiBGcorrs.Reset()
        dPhiSig.Reset()


        for iBin in range(1, dPhiBGcorrs.GetNbinsX() + 1):
            if dPhiBGcorrs.GetBinCenter(iBin + 1) < np.pi / 2:
                # Near Side
                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth))
                    /2
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin, dPhiMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSig.SetBinError(
                    iBin, dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )

                
            else:
                # Away Side

                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSig.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )

                
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrsForRefoldedSpecies[species][i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsForRefoldedSpecies[species][i, j, k] = dPhiSig
            
        elif self.analysisType == "pp":
            self.dPhiBGcorrsForRefoldedSpecies[species][i, j] = dPhiBGcorrs
            self.dPhiSigcorrsForRefoldedSpecies[species][i, j] = dPhiSig
            
        del dPhiSig
    
    @log_function_call(description="")
    def fill_dPhi_correlation_functions_for_true_species_in_z_vertex_bins(self, i, j, k, species):

        dPhiMid, nbins_mid, binwidth, dPhiPIDerrMid = self.get_dPhi_projection_in_dEta_range_for_true_species(i,j,k,
            self.dEtaSig, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGLo, nbins_lo, binwidth, dPhiPIDerrLo = self.get_dPhi_projection_in_dEta_range_for_true_species(i,j,k,
            self.dEtaBGLo, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGHi, nbins_hi, binwidth, dPhiPIDerrHi = self.get_dPhi_projection_in_dEta_range_for_true_species(i,j,k,
            self.dEtaBGHi, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGcorrs = dPhiMid.Clone()
        dPhiSig = dPhiMid.Clone()
        dPhiBGcorrs.Reset()
        dPhiSig.Reset()

        dPhiBGPIDerr = dPhiPIDerrMid.Clone()
        dPhiSigPIDerr = dPhiPIDerrMid.Clone()
        dPhiBGPIDerr.Reset()
        dPhiSigPIDerr.Reset()

        for iBin in range(1, dPhiBGcorrs.GetNbinsX() + 1):
            if dPhiBGcorrs.GetBinCenter(iBin + 1) < np.pi / 2:
                # Near Side

                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                    )
                    /2
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin, dPhiMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSig.SetBinError(
                    iBin, dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )

                dPhiBGPIDerr.SetBinContent(
                    iBin,
                    (
                        dPhiPIDerrLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiPIDerrHi.GetBinContent(iBin)/((nbins_hi+1)*binwidth)
                    )
                    /2
                )
                dPhiBGPIDerr.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiPIDerrLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiPIDerrHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )

                dPhiSigPIDerr.SetBinContent(
                    iBin, dPhiPIDerrMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSigPIDerr.SetBinError(
                    iBin, dPhiPIDerrMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )
            else:
                # Away Side

                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSig.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiBGPIDerr.SetBinContent(
                    iBin,
                    (
                        dPhiPIDerrLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiPIDerrHi.GetBinContent(iBin)/((nbins_hi+1)*binwidth)
                        + dPhiPIDerrMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGPIDerr.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiPIDerrLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiPIDerrHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiPIDerrMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiSigPIDerr.SetBinContent(
                    iBin,
                    (
                        dPhiPIDerrLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiPIDerrHi.GetBinContent(iBin)/((nbins_hi+1)*binwidth)
                        + dPhiPIDerrMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiSigPIDerr.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiPIDerrLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiPIDerrHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiPIDerrMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )

        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrsForTrueSpeciesZV[species][i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsForTrueSpeciesZV[species][i, j, k] = dPhiSig
            self.dPhiBGPIDErrForTrueSpeciesZV[species][i, j, k] = dPhiBGPIDerr
            self.dPhiSigPIDErrForTrueSpeciesZV[species][i, j, k] = dPhiSigPIDerr
        elif self.analysisType == "pp":
            self.dPhiBGcorrsForTrueSpeciesZV[species][i, j] = dPhiBGcorrs
            self.dPhiSigcorrsForTrueSpeciesZV[species][i, j] = dPhiSig
            self.dPhiBGPIDErrForTrueSpeciesZV[species][i, j] = dPhiBGPIDerr
            self.dPhiSigPIDErrForTrueSpeciesZV[species][i, j] = dPhiSigPIDerr
        del dPhiSig
    
    @log_function_call(description="", logging_level=logging.INFO)
    def fill_dPhi_correlation_functions_for_enhanced_species(self, i, j, k, species):

        dPhiMid, nbins_mid, binwidth = self.get_dPhi_projection_in_dEta_range_for_enhanced_species(i,j,k,
            self.dEtaSig, TOFcutSpecies=species
        )

        dPhiBGLo, nbins_lo, binwidth = self.get_dPhi_projection_in_dEta_range_for_enhanced_species(i,j,k,
            self.dEtaBGLo, TOFcutSpecies=species
        )

        dPhiBGHi, nbins_hi, binwidth = self.get_dPhi_projection_in_dEta_range_for_enhanced_species(i,j,k,
            self.dEtaBGHi, TOFcutSpecies=species
        )

        dPhiBGcorrs = dPhiMid.Clone()
        dPhiSig = dPhiMid.Clone()
        dPhiBGcorrs.Reset()
        dPhiSig.Reset()


        for iBin in range(1, dPhiBGcorrs.GetNbinsX() + 1):
            if dPhiBGcorrs.GetBinCenter(iBin + 1) < np.pi / 2:
                # Near Side
                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth))
                    /2
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin, dPhiMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSig.SetBinError(
                    iBin, dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )

                

            else:
                # Away Side

                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSig.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )

                
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrsForEnhancedSpecies[species][i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsForEnhancedSpecies[species][i, j, k] = dPhiSig

        elif self.analysisType == "pp":
            self.dPhiBGcorrsForEnhancedSpecies[species][i, j] = dPhiBGcorrs
            self.dPhiSigcorrsForEnhancedSpecies[species][i, j] = dPhiSig

        del dPhiSig
    
    @log_function_call(description="")
    def fill_dPhi_correlation_functions_for_enhanced_species_in_z_vertex_bins(self, i, j, k, species):

        dPhiMid, nbins_mid, binwidth = self.get_dPhi_projection_in_dEta_range_for_enhanced_species(i,j,k,
            self.dEtaSig, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGLo, nbins_lo, binwidth = self.get_dPhi_projection_in_dEta_range_for_enhanced_species(i,j,k,
            self.dEtaBGLo, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGHi, nbins_hi, binwidth = self.get_dPhi_projection_in_dEta_range_for_enhanced_species(i,j,k,
            self.dEtaBGHi, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGcorrs = dPhiMid.Clone()
        dPhiSig = dPhiMid.Clone()
        dPhiBGcorrs.Reset()
        dPhiSig.Reset()


        for iBin in range(1, dPhiBGcorrs.GetNbinsX() + 1):
            if dPhiBGcorrs.GetBinCenter(iBin + 1) < np.pi / 2:
                # Near Side

                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                    )
                    /2
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                        )
                        /4
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin, dPhiMid.GetBinContent(iBin)/((nbins_mid+1)*binwidth)
                )
                dPhiSig.SetBinError(
                    iBin, dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)
                )

                
            else:
                # Away Side

                dPhiBGcorrs.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin) /((nbins_hi+1)*binwidth)
                        + dPhiMid.GetBinContent(iBin) /((nbins_mid+1)*binwidth)
                    )
                    /3
                )
                dPhiBGcorrs.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth))**2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth))**2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth))**2
                        )
                        /9
                    )
                    ** 0.5,
                )

                dPhiSig.SetBinContent(
                    iBin,
                    (
                        dPhiBGLo.GetBinContent(iBin)/((nbins_lo+1)*binwidth)
                        + dPhiBGHi.GetBinContent(iBin)/((nbins_hi +1)*binwidth)
                        + dPhiMid.GetBinContent(iBin)/((nbins_mid +1)*binwidth)
                    )
                    /3
                )
                dPhiSig.SetBinError(
                    iBin,
                    (
                        (
                            (dPhiBGHi.GetBinError(iBin)/((nbins_hi+1)*binwidth)) ** 2
                            + (dPhiBGLo.GetBinError(iBin)/((nbins_lo+1)*binwidth)) ** 2
                            + (dPhiMid.GetBinError(iBin)/((nbins_mid+1)*binwidth)) ** 2
                        )
                        /9
                    )
                    ** 0.5,
                )

                

        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrsForEnhancedSpeciesZV[species][i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsForEnhancedSpeciesZV[species][i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiBGcorrsForEnhancedSpeciesZV[species][i, j] = dPhiBGcorrs
            self.dPhiSigcorrsForEnhancedSpeciesZV[species][i, j] = dPhiSig
        del dPhiSig



    @log_function_call(description="", logging_level=logging.INFO)
    def fill_BG_subtracted_AccCorrected_SE_correlation_functions(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            NSCorr = self.get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(i, j, k)  # type: ignore
            ASCorr = self.get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(i, j, k)  # type: ignore
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, k] = NSCorr
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, k] = ASCorr
        elif self.analysisType == "pp":
            NSCorr, NSminVal = self.get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(i, j, k)  # type: ignore
            ASCorr, ASminVal = self.get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(i, j, k)  # type: ignore
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j] = NSCorr
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j] = ASCorr
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals[i, j] = NSminVal
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals[i, j] = ASminVal
        del NSCorr, ASCorr
    
    @log_function_call(description="")
    def fill_BG_subtracted_AccCorrected_SE_correlation_functions_in_z_vertex_bins(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            NSCorr = self.get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(i, j, k, in_z_vertex_bins=True)  # type: ignore
            ASCorr = self.get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(i, j, k, in_z_vertex_bins=True)  # type: ignore
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, k] = NSCorr
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, k] = ASCorr
        elif self.analysisType == "pp":
            NSCorr, NSminVal = self.get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(i, j, k, in_z_vertex_bins=True)  # type: ignore
            ASCorr, ASminVal = self.get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(i, j, k, in_z_vertex_bins=True)  # type: ignore
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j] = NSCorr
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j] = ASCorr
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminValsZV[i, j] = NSminVal
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminValsZV[i, j] = ASminVal
        del NSCorr, ASCorr

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(
        self, i, j, k
    ):

        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(i, j, k)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(i, j, k)  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j] = Corr  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[i, j] = minVal  # type: ignore
        del Corr
    
    @log_function_call(description="")
    def fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_in_z_vertex_bins(
        self, i, j, k
    ):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(i, j, k, in_z_vertex_bins=True)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(i, j, k, in_z_vertex_bins=True)  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j] = Corr  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminValsZV[i, j] = minVal  # type: ignore
        del Corr

    @log_function_call(description="")
    def fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
        self, i, j, k, species
    ):

        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_true_species(
                i, j, k, species
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs[
                species
            ][i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_true_species(i, j, k, species)  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrsminVals[species][i, j] = minVal  # type: ignore
        del Corr

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_normalized_background_subtracted_dPhi_for_true_species(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_background_subtracted_dPhi_for_true_species(
                i, j, k, species
            )
            self.NormalizedBGSubtracteddPhiForTrueSpecies[species][i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_background_subtracted_dPhi_for_true_species(i, j, k, species)  # type: ignore
            self.NormalizedBGSubtracteddPhiForTrueSpecies[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtracteddPhiForTrueSpeciesminVals[species][i, j] = minVal  # type: ignore
        del Corr
   
    @log_function_call(description="", logging_level=logging.INFO)
    def fill_normalized_background_subtracted_dPhi_for_enhanced_species(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_background_subtracted_dPhi_for_enhanced_species(
                i, j, k, species
            )
            self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][i,j,k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_background_subtracted_dPhi_for_enhanced_species(i, j, k, species)  # type: ignore
            self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminVals[species][i, j] = minVal  # type: ignore
        del Corr
    
    @log_function_call(description="", logging_level=logging.INFO)
    def fill_normalized_background_subtracted_dPhi_for_refolded_species(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_background_subtracted_dPhi_for_refolded_species(
                i, j, k, species
            )
            self.NormalizedBGSubtracteddPhiForRefoldedSpecies[species][i,j,k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_background_subtracted_dPhi_for_refolded_species(i, j, k, species)  # type: ignore
            self.NormalizedBGSubtracteddPhiForRefoldedSpecies[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtracteddPhiForRefoldedSpeciesminVals[species][i, j] = minVal  # type: ignore
        del Corr
    
    @log_function_call(description="")
    def fill_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_background_subtracted_dPhi_for_true_species(
                i, j, k, species, in_z_vertex_bins=True
            )
            self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_background_subtracted_dPhi_for_true_species(i, j, k, species, in_z_vertex_bins=True)  # type: ignore
            self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtracteddPhiForTrueSpeciesminValsZV[species][i, j] = minVal  # type: ignore
        del Corr
    
    @log_function_call(description="")
    def fill_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_background_subtracted_dPhi_for_enhanced_species(
                i, j, k, species, in_z_vertex_bins=True
            )
            self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_background_subtracted_dPhi_for_enhanced_species(i, j, k, species, in_z_vertex_bins=True)  # type: ignore
            self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminValsZV[species][i, j] = minVal  # type: ignore
        del Corr

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_yield_for_true_species(self, i, j, species):
        yield_, yield_err_ = self.get_yield_for_true_species(i, j, 3, species)
        self.YieldsTrue[species][i, j] = yield_
        self.YieldErrsTrue[species][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_yield_for_true_species(
            i, j, 3, species, region="NS"
        )
        self.YieldsTrueNS[species][i, j] = yield_NS
        self.YieldErrsTrueNS[species][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_yield_for_true_species(
            i, j, 3, species, region="AS"
        )
        self.YieldsTrueAS[species][i, j] = yield_AS
        self.YieldErrsTrueAS[species][i, j] = yield_err_AS
    
    @log_function_call(description="", logging_level=logging.INFO)
    def fill_yield_for_enhanced_species(self, i, j, species):
        yield_, yield_err_ = self.get_yield_for_enhanced_species(i, j, 3, species)
        self.YieldsEnhanced[species][i, j] = yield_
        self.YieldErrsEnhanced[species][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_yield_for_enhanced_species(
            i, j, 3, species, region="NS"
        )
        self.YieldsEnhancedNS[species][i, j] = yield_NS
        self.YieldErrsEnhancedNS[species][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_yield_for_enhanced_species(
            i, j, 3, species, region="AS"
        )
        self.YieldsEnhancedAS[species][i, j] = yield_AS
        self.YieldErrsEnhancedAS[species][i, j] = yield_err_AS
    
    @log_function_call(description="")
    def fill_yield_for_true_species_in_z_vertex_bins(self, i, j, species):
        yield_, yield_err_ = self.get_yield_for_true_species(i, j, 3, species, in_z_vertex_bins=True)
        self.YieldsTrueZV[species][i, j] = yield_
        self.YieldErrsTrueZV[species][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_yield_for_true_species(
            i, j, 3, species, region="NS", in_z_vertex_bins=True
        )
        self.YieldsTrueNSZV[species][i, j] = yield_NS
        self.YieldErrsTrueNSZV[species][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_yield_for_true_species(
            i, j, 3, species, region="AS", in_z_vertex_bins=True
        )
        self.YieldsTrueASZV[species][i, j] = yield_AS
        self.YieldErrsTrueASZV[species][i, j] = yield_err_AS
    
    @log_function_call(description="")
    def fill_yield_for_enhanced_species_in_z_vertex_bins(self, i, j, species):
        yield_, yield_err_ = self.get_yield_for_true_species(i, j, 3, species, in_z_vertex_bins=True)
        self.YieldsEnhancedZV[species][i, j] = yield_
        self.YieldErrsEnhancedZV[species][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_yield_for_true_species(
            i, j, 3, species, region="NS", in_z_vertex_bins=True
        )
        self.YieldsEnhancedNSZV[species][i, j] = yield_NS
        self.YieldErrsEnhancedNSZV[species][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_yield_for_true_species(
            i, j, 3, species, region="AS", in_z_vertex_bins=True
        )
        self.YieldsEnhancedASZV[species][i, j] = yield_AS
        self.YieldErrsEnhancedASZV[species][i, j] = yield_err_AS

    @log_function_call(description="", logging_level=logging.INFO)
    def fill_inclusive_yield(self, i, j,):
        yield_, yield_err_ = self.get_inclusive_yield(i, j, 3)
        self.YieldsTrue["inclusive"][i, j] = yield_
        self.YieldErrsTrue["inclusive"][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_inclusive_yield(
            i, j, 3, region="NS"
        )
        self.YieldsTrueNS["inclusive"][i, j] = yield_NS
        self.YieldErrsTrueNS["inclusive"][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_inclusive_yield(
            i, j, 3, region="AS"
        )
        self.YieldsTrueAS["inclusive"][i, j] = yield_AS
        self.YieldErrsTrueAS["inclusive"][i, j] = yield_err_AS
    
    @log_function_call(description="")
    def fill_inclusive_yield_in_z_vertex_bins(self, i, j):
        yield_, yield_err_ = self.get_inclusive_yield(i, j, 3, in_z_vertex_bins=True)
        self.YieldsTrueZV['inclusive'][i, j] = yield_
        self.YieldErrsTrueZV['inclusive'][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_inclusive_yield(
            i, j, 3, region="NS", in_z_vertex_bins=True
        )
        self.YieldsTrueNSZV['inclusive'][i, j] = yield_NS
        self.YieldErrsTrueNSZV['inclusive'][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_inclusive_yield(
            i, j, 3, region="AS", in_z_vertex_bins=True
        )
        self.YieldsTrueASZV['inclusive'][i, j] = yield_AS
        self.YieldErrsTrueASZV['inclusive'][i, j] = yield_err_AS

    @log_function_call(description="")
    def set_pT_epAngle_bin(self, i, j, k):

        for sparse_ind in range(len(self.JH)):
            # set the pT and event plane angle ranges
            self.JH[sparse_ind].GetAxis(1).SetRangeUser(
                self.pTtrigBinEdges[i], self.pTtrigBinEdges[i + 1]
            )
            self.MixedEvent[sparse_ind].GetAxis(1).SetRangeUser(
                self.pTtrigBinEdges[i], self.pTtrigBinEdges[i + 1]
            )
            self.Trigger[sparse_ind].GetAxis(1).SetRangeUser(
                self.pTtrigBinEdges[i], self.pTtrigBinEdges[i + 1]
            )

            self.JH[sparse_ind].GetAxis(2).SetRangeUser(
                self.pTassocBinEdges[j], self.pTassocBinEdges[j + 1]
            )
            self.MixedEvent[sparse_ind].GetAxis(2).SetRangeUser(
                self.pTassocBinEdges[j], self.pTassocBinEdges[j + 1]
            )
            if self.analysisType in ["central", "semicentral"]:
                # set the centrality to be 30-50%
                if self.analysisType == "semicentral":
                    self.JH[sparse_ind].GetAxis(0).SetRangeUser(30,50)
                    self.MixedEvent[sparse_ind].GetAxis(0).SetRangeUser(30,50)
                    self.Trigger[sparse_ind].GetAxis(0).SetRangeUser(30,50)
                else: 
                    self.JH[sparse_ind].GetAxis(0).SetRangeUser(0,10)
                    self.MixedEvent[sparse_ind].GetAxis(0).SetRangeUser(0,10)
                    self.Trigger[sparse_ind].GetAxis(0).SetRangeUser(0,10)
                if k == 3:
                    self.JH[sparse_ind].GetAxis(5).SetRangeUser(
                        self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[3]
                    )
                    self.MixedEvent[sparse_ind].GetAxis(5).SetRangeUser(
                        self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[3]
                    )
                    self.Trigger[sparse_ind].GetAxis(2).SetRangeUser(
                        self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[3]
                    )
                else:
                    self.JH[sparse_ind].GetAxis(5).SetRangeUser(
                        self.eventPlaneAngleBinEdges[k],
                        self.eventPlaneAngleBinEdges[k + 1],
                    )
                    # self.MixedEvent.GetAxis(5).SetRangeUser(self.eventPlaneAngleBinEdges[k], self.eventPlaneAngleBinEdges[k+1])
                    self.Trigger[sparse_ind].GetAxis(2).SetRangeUser(
                        self.eventPlaneAngleBinEdges[k],
                        self.eventPlaneAngleBinEdges[k + 1],
                    )
        self.set_has_changed()

    @log_function_call(description="")
    def set_pT_assoc_range(self, j_low, j_hi):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(2).SetRangeUser(
                self.pTassocBinEdges[j_low], self.pTassocBinEdges[j_hi]
            )
            self.MixedEvent[sparse_ind].GetAxis(2).SetRangeUser(
                self.pTassocBinEdges[j_low], self.pTassocBinEdges[j_hi]
            )
        self.set_has_changed()

    @log_function_call(description="")
    def set_pT_trig_range(self, i_low, i_hi):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(1).SetRangeUser(
                self.pTtrigBinEdges[i_low], self.pTtrigBinEdges[i_hi]
            )
            self.MixedEvent[sparse_ind].GetAxis(1).SetRangeUser(
                self.pTtrigBinEdges[i_low], self.pTtrigBinEdges[i_hi]
            )
            self.Trigger[sparse_ind].GetAxis(1).SetRangeUser(
                self.pTtrigBinEdges[i_low], self.pTtrigBinEdges[i_hi]
            )
        self.set_has_changed()

    @log_function_call(description="")
    def set_z_vertex_bin(self, bin_no):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(5).SetRangeUser(self.z_vertex_bins[bin_no], self.z_vertex_bins[bin_no+1])
            self.MixedEvent[sparse_ind].GetAxis(5).SetRangeUser(self.z_vertex_bins[bin_no], self.z_vertex_bins[bin_no+1])
            self.Trigger[sparse_ind].GetAxis(2).SetRangeUser(self.z_vertex_bins[bin_no], self.z_vertex_bins[bin_no+1])
        self.set_has_changed()

    @log_function_call(description="")
    def reset_z_vertex_bin(self):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(5).SetRange(0, -1)
            self.MixedEvent[sparse_ind].GetAxis(5).SetRange(0, -1)
            self.Trigger[sparse_ind].GetAxis(2).SetRange(0, -1)
        self.set_has_changed()

    @log_function_call(description="")
    def set_PID_selection(self, species):
        for sparse_ind in range(len(self.JH)):

            if species=="pion":
                self.JH[sparse_ind].GetAxis(self.pionID).SetRangeUser(
                    -2 , 2
                ) 
                self.JH[sparse_ind].GetAxis(self.kaonID).SetRange(0,self.JH[sparse_ind].GetAxis(self.kaonID).FindBin(-2 if self.current_assoc_bin<2 else -1))
                self.JH[sparse_ind].GetAxis(self.protonID).SetRange(0,self.JH[sparse_ind].GetAxis(self.protonID).FindBin(-2))

            if species=="kaon":
                self.JH[sparse_ind].GetAxis(self.kaonID).SetRangeUser(
                    -2 if self.current_assoc_bin<2 else -1 , 2
                )  
                self.JH[sparse_ind].GetAxis(self.protonID).SetRange(0,self.JH[sparse_ind].GetAxis(self.protonID).FindBin(-2))

            if species=="proton":
                self.JH[sparse_ind].GetAxis(self.protonID).SetRangeUser(
                    -2 , 2
                )

            if species=="other_p+":
                self.JH[sparse_ind].GetAxis(self.protonID).SetRange(
                    self.JH[sparse_ind].GetAxis(self.protonID).FindBin(2), self.JH[sparse_ind].GetAxis(self.protonID).GetNbins()+1
                )  

            if species=="other_p-_k+":
                self.JH[sparse_ind].GetAxis(self.protonID).SetRange(
                    0, self.JH[sparse_ind].GetAxis(self.protonID).FindBin(-2)
                )  
                # k
                self.JH[sparse_ind].GetAxis(self.kaonID).SetRange(
                     self.JH[sparse_ind].GetAxis(self.kaonID).FindBin(2), self.JH[sparse_ind].GetAxis(self.kaonID).GetNbins()+1
                )  

            if species=="other_pi-_p-_k-":
                self.JH[sparse_ind].GetAxis(self.pionID).SetRange(
                    0, self.JH[sparse_ind].GetAxis(self.pionID).FindBin(-2)
                )  
                # p
                self.JH[sparse_ind].GetAxis(self.protonID).SetRange(
                     0,self.JH[sparse_ind].GetAxis(self.protonID).FindBin(-2)
                )  
                # k
                self.JH[sparse_ind].GetAxis(self.kaonID).SetRange(
                     0, self.JH[sparse_ind].GetAxis(self.kaonID).FindBin(-2)
                )  

            if species=="other_pi+_p-_k-":
                self.JH[sparse_ind].GetAxis(self.pionID).SetRange(
                     self.JH[sparse_ind].GetAxis(self.pionID).FindBin(2), self.JH[sparse_ind].GetAxis(self.pionID).GetNbins()+1
                ) 
                # p
                self.JH[sparse_ind].GetAxis(self.protonID).SetRange(
                     0, self.JH[sparse_ind].GetAxis(self.protonID).FindBin(-2)
                ) 
                # k
                self.JH[sparse_ind].GetAxis(self.kaonID).SetRange(
                     0, self.JH[sparse_ind].GetAxis(self.kaonID).FindBin(-2)
                ) 
            else:
                raise ValueError(f"{species} is not a recognized particle species or other* specification.")

    @log_function_call(description="")
    def reset_PID_selection(self):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(self.pionID).SetRange(
                0,0
            )  # type:ignore
            self.JH[sparse_ind].GetAxis(self.protonID).SetRange(
                0,0
            )  # type:ignore
            self.JH[sparse_ind].GetAxis(self.kaonID).SetRange(
                0,0
            )  # type:ignore

    @log_function_call(description="")
    def set_has_changed(self):
        self.get_SE_correlation_function_has_changed = True
        self.get_SE_correlation_function_w_Pion_has_changed = init_bool_dict()
        self.get_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_error_has_changed = True
        self.get_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed = True
        self.get_normalized_acceptance_corrected_correlation_function_has_changed = True
        self.ME_norm_sliding_window_has_changed = True
        self.resetUnfoldingResults = True

    def __repr__(self) -> str:
        string_rep = f"""JetHadron object for {self.analysisType} events\n
                                 with {self.nTrigPtBins} trigger pT bins, 
                                {self.nAssocPtBins} associated pT bins, 
                                and {self.nEventPlaneAngleBins} event plane angle bins\n
                                """
        return string_rep

    def get_sparses(self, f):
        """
        Returns tuple of (fhnJH, fhnMixedEvent, fhnTrigger) for central events
        """
        if self.analysisType in ["central", "semicentral"]:
            centralityString = (
                "Central" if self.analysisType == "central" else "SemiCentral"
            )
            anaList = f.Get(
                f"AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2{centralityString}q"
            )
        else:
            anaList = f.Get("AliAnalysisTaskJetH_tracks_caloClusters_biased")
        fhnJH = anaList.FindObject("fhnJH")
        fhnMixedEvent = anaList.FindObject("fhnMixedEvents")
        fhnTrigger = anaList.FindObject("fhnTrigger")
        # turn on errors with sumw2

        return fhnJH, fhnMixedEvent, fhnTrigger

    def get_event_plane_angle_hist(self, f):
        if self.analysisType in ["central", "semicentral"]:
            centralityString = (
                "Central" if self.analysisType == "central" else "SemiCentral"
            )
            anaList = f.Get(
                f"AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2{centralityString}q"
            )
        else:
            anaList = f.Get("AliAnalysisTaskJetH_tracks_caloClusters_biased")
        fHistEventPlane = TH1F(anaList.FindObject("fHistEventPlane"))
        return fHistEventPlane.Clone()

    def assert_sparses_filled(self):
        for sparse_ind in range(len(self.JH)):
            assert (
                self.JH[sparse_ind].GetEntries() != 0
            ), f"⚠️⚠️ No entries for {self.analysisType} JH sparse at index {sparse_ind}. ⚠️⚠️"
            assert (
                self.MixedEvent[sparse_ind].GetEntries() != 0
            ), f"⚠️⚠️ No entries for {self.analysisType} Mixed Event sparse at index {sparse_ind}. ⚠️⚠️"
            assert (
                self.Trigger[sparse_ind].GetEntries() != 0
            ), f"⚠️⚠️ No entries for {self.analysisType} Trigger sparse at index {sparse_ind}. ⚠️⚠️"
