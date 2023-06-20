from collections import defaultdict
from itertools import product
import pickle
import subprocess
from typing import Optional
import numpy as np
import requests
import ROOT
import matplotlib.pyplot as plt
from functools import partial


from time import time

import warnings

warnings.filterwarnings("ignore")

import _JetHadronPlot, _JetHadronFit, _JetHadronAnalysis

# turn latex on in matplotlib
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# increase the font size for christine
plt.rcParams.update({"font.size": 16})


def print_function_name_with_description_on_call(description):
    """
    Prints the name of the function and a description of what it does
    """

    def function_wrapper(function):
        def method_wrapper(self, *args, **kwargs):
            print(f"{function.__name__} in {self.__class__.__name__}:\n\t{description}")
            return function(self, *args, **kwargs)

        return method_wrapper

    return function_wrapper

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
        ROOT.TH1.AddDirectory(False)
        assert analysisType in ["central", "semicentral", "pp"]
        self.analysisType = analysisType
        self.base_save_path = (
            f"/home/steffanic/Projects/Thesis/backend_output/{analysisType}/"
        )
        # let's turn the sparses into lists of sparses to use all files
        self.JH, self.MixedEvent, self.Trigger = [], [], []
        self.EventPlaneAngleHist = []
        first = True
        for filename in rootFileNames:
            print(f"Loading file {filename}")
            file = ROOT.TFile(filename)
            fileJH, fileME, fileT = self.get_sparses(file)
            self.JH.append(fileJH)
            self.MixedEvent.append(fileME)
            self.Trigger.append(fileT)
            if self.analysisType in ["central", "semicentral"]:
                self.EventPlaneAngleHist.append(self.get_event_plane_angle_hist(file))

        print("Finished loading files")
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
            60,
        ]  # subject to change based on statistics
        self.pTtrigBinCenters = [
            # 15,
            30,
            50,
        ]  # subject to change based on statistics
        self.pTtrigBinWidths = [
            # 10,
            20,
            20,
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

        self.z_vertex_bins_PbPb= [-10,-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10]
        self.z_vertex_bins_pp= [-2, -1, 0, 1, 2]
        if self.analysisType == 'pp':
            self.z_vertex_bins = self.z_vertex_bins_pp
        else:
            self.z_vertex_bins = self.z_vertex_bins_PbPb

        # 15 pt assoc bins * 7 pt trig bins * 4 event plane bins = 420 bins

        # define signal and background regions
        self.dEtaBGHi = [0.8, 1.20]
        self.dEtaBGLo = [-1.20, -0.8]
        self.dEtaSig = [-0.6, 0.6]
        self.dEtaSigAS = [-1.20, 1.20]
        self.dPhiSigNS = [-np.pi / 2, np.pi / 2]
        self.dPhiSigAS = [np.pi / 2, 3 * np.pi / 2]

        self.get_SE_correlation_function_has_changed = True
        self.get_SE_correlation_function_w_Pion_has_changed = self.init_bool_dict()
        self.get_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_error_has_changed = True
        self.get_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed = True
        self.get_normalized_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_correlation_function_w_pionTPCnSigma_has_changed = (
            self.init_bool_dict()
        )
        self.ME_norm_sliding_window_has_changed = True

        self.get_SE_correlation_function_result = None
        self.get_SE_correlation_function_w_Pion_result = self.init_none_dict()
        self.get_ME_correlation_function_result = None
        self.get_normalized_ME_correlation_function_result = None
        self.get_normalized_ME_correlation_error = None
        self.get_acceptance_corrected_correlation_function_result = None
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_result = None
        self.get_normalized_acceptance_corrected_correlation_function_result = None
        self.get_acceptance_corrected_correlation_function_w_pionTPCnSigma_result = (
            self.init_none_dict()
        )
        self.ME_norm_sliding_window_result = None

        if self.analysisType in ["central", "semicentral"]:
            # Define all the arrays that will hold various objects for each bin
            self.N_trigs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.eventPlaneAngleBinEdges)),
                dtype=int,
            )
            self.N_assoc =  self.init_pTtrig_pTassoc_eventPlane_array(int)
            self.N_assoc_for_species = self.init_pTtrig_pTassoc_eventPlane_dict(int)
            self.SEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )  # Event plane angle has 4 bins, in-, mid-, out, and inclusive
            self.NormMEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH2F)
            self.AccCorrectedSEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )
            self.AccCorrectedSEcorrsZV = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )
            self.NormAccCorrectedSEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )
            self.NormAccCorrectedSEcorrsZV = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )
            self.ME_norm_systematics = self.init_pTtrig_pTassoc_eventPlane_array(float)
            self.dPhiSigcorrs = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.dPhiSigcorrsZV = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.dPhiBGcorrsForSpecies = self.init_pTtrig_pTassoc_eventPlane_dict(
                ROOT.TH1F
            )
            self.dPhiSigcorrsForSpecies = self.init_pTtrig_pTassoc_eventPlane_dict(
                ROOT.TH1F
            )
            self.dPhiBGcorrsForSpeciesZV = self.init_pTtrig_pTassoc_eventPlane_dict(
                ROOT.TH1F
            )
            self.dPhiSigcorrsForSpeciesZV = self.init_pTtrig_pTassoc_eventPlane_dict(
                ROOT.TH1F
            )
            self.dPhiBGcorrs = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.dPhiBGcorrsZV = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.dPhiSigdpionTPCnSigmacorrs = self.init_pTtrig_pTassoc_eventPlane_dict(
                ROOT.TH2F
            )
            self.dEtacorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.RPFObjs = self.init_pTtrig_pTassoc_array(object)
            self.RPFObjsZV = self.init_pTtrig_pTassoc_array(object)
            self.PionTPCNSigmaFitObjs = self.init_pTtrig_pTassoc_eventPlane_array(object)
            self.RPFObjsForSpecies = self.init_pTtrig_pTassoc_dict(object)
            self.RPFObjsForSpeciesZV = self.init_pTtrig_pTassoc_dict(object)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs = (
                self.init_pTtrig_pTassoc_eventPlane_dict(ROOT.TH2F)
            )
            self.NormalizedBGSubtracteddPhiForSpecies = (
                self.init_pTtrig_pTassoc_eventPlane_dict(ROOT.TH1F)
            )
            self.NormalizedBGSubtracteddPhiForSpeciesZV = (
                self.init_pTtrig_pTassoc_eventPlane_dict(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            )
            self.pionTPCsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPionNSsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPionASsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_pionTOFcut = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.pionTPCnSigma_protonTOFcut = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.pionTPCnSigma_kaonTOFcut = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)


            self.Yields = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsAS = self.init_pTtrig_pTassoc_dict(float)

            self.YieldErrs = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsAS = self.init_pTtrig_pTassoc_dict(float)
            
            self.YieldsZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsNSZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsASZV = self.init_pTtrig_pTassoc_dict(float)

            self.YieldErrsZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsNSZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsASZV = self.init_pTtrig_pTassoc_dict(float)

        else:
            self.N_trigs = np.zeros(
                (len(self.pTtrigBinEdges) - 1),
                dtype=object,
            )
            self.N_assoc = self.init_pTtrig_pTassoc_array(int)
            self.N_assoc_for_species = self.init_pTtrig_pTassoc_dict(int)
            self.SEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.NormMEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.AccCorrectedSEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.AccCorrectedSEcorrsZV = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.NormAccCorrectedSEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.NormAccCorrectedSEcorrsZV = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.ME_norm_systematics = self.init_pTtrig_pTassoc_array(float)
            self.dPhiSigcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPhiSigcorrsZV = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPhiBGcorrsForSpecies = self.init_pTtrig_pTassoc_dict(ROOT.TH1F)
            self.dPhiSigcorrsForSpecies = self.init_pTtrig_pTassoc_dict(ROOT.TH1F)
            self.dPhiBGcorrsForSpeciesZV = self.init_pTtrig_pTassoc_dict(ROOT.TH1F)
            self.dPhiSigcorrsForSpeciesZV = self.init_pTtrig_pTassoc_dict(ROOT.TH1F)
            self.dPhiBGcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPhiBGcorrsZV = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPhiSigdpionTPCnSigmacorrs = self.init_pTtrig_pTassoc_dict(ROOT.TH2F)
            self.dEtacorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.RPFObjs = self.init_pTtrig_pTassoc_array(object)
            self.RPFObjsZV = self.init_pTtrig_pTassoc_array(object)
            self.PionTPCNSigmaFitObjs = self.init_pTtrig_pTassoc_array(object)
            self.RPFObjsForSpecies = self.init_pTtrig_pTassoc_dict(object)
            self.RPFObjsForSpeciesZV = self.init_pTtrig_pTassoc_dict(object)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs = (
                self.init_pTtrig_pTassoc_dict(ROOT.TH2F)
            )
            self.NormalizedBGSubtracteddPhiForSpecies = self.init_pTtrig_pTassoc_dict(
                ROOT.TH1F
            )
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals = (
                self.init_pTtrig_pTassoc_array(float)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals = (
                self.init_pTtrig_pTassoc_array(float)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals = (
                self.init_pTtrig_pTassoc_array(float)
            )
            self.NormalizedBGSubtracteddPhiForSpeciesZV = self.init_pTtrig_pTassoc_dict(
                ROOT.TH1F
            )
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminValsZV = (
                self.init_pTtrig_pTassoc_array(float)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminValsZV = (
                self.init_pTtrig_pTassoc_array(float)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminValsZV = (
                self.init_pTtrig_pTassoc_array(float)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrsminVals = self.init_pTtrig_pTassoc_dict(
                float
            )
            self.NormalizedBGSubtracteddPhiForSpeciesminVals = (
                self.init_pTtrig_pTassoc_dict(float)
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            )
            self.NormalizedBGSubtracteddPhiForSpeciesminValsZV = (
                self.init_pTtrig_pTassoc_dict(float)
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrsZV = (
                self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrsZV = (
                self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            )
            self.pionTPCsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPionNSsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPionASsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_pionTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_protonTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_kaonTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)


            self.Yields = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsAS = self.init_pTtrig_pTassoc_dict(float)

            self.YieldErrs = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsAS = self.init_pTtrig_pTassoc_dict(float)
            
            self.YieldsZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsNSZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsASZV = self.init_pTtrig_pTassoc_dict(float)

            self.YieldErrsZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsNSZV = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsASZV = self.init_pTtrig_pTassoc_dict(float)

        if fill_on_init:
            [
                [
                    self.fill_hist_arrays(i, j)
                    for j in range(len(self.pTassocBinEdges) - 1)
                ]
                for i in range(len(self.pTtrigBinEdges) - 1)
            ]
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

    def return_none(self):
        return None

    def return_true(self):
        return True

    def init_none_dict(self):
        return defaultdict(self.return_none)

    def init_bool_dict(self):
        return defaultdict(self.return_true)

    def init_pTtrig_pTassoc_dict(self, dtype):
        return defaultdict(partial(self.init_pTtrig_pTassoc_array, dtype=dtype))

    def init_pTtrig_pTassoc_eventPlane_dict(self, dtype):
        return defaultdict(
            partial(self.init_pTtrig_pTassoc_eventPlane_array, dtype=dtype)
        )

    def init_pTtrig_pTassoc_eventPlane_array(self, dtype=object):
        return np.zeros(
            (
                len(self.pTtrigBinEdges) - 1,
                len(self.pTassocBinEdges) - 1,
                len(self.eventPlaneAngleBinEdges),
            ),
            dtype=dtype,
        )

    def init_pTtrig_pTassoc_array(self, dtype=object):
        return np.zeros(
            (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1), dtype=dtype
        )

    @print_function_name_with_description_on_call(description="")
    def fill_hist_arrays(self, i, j, hists_to_fill: "Optional[dict[str, bool]]" = None):
        if self.analysisType in ["central", "semicentral"]:
            self.set_pT_epAngle_bin(i, j, 3)
            self.fill_N_assoc(i, j,3)
            
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
                    self.fill_N_assoc(i, j, k)
                for species in ['pion', 'proton', 'kaon']:
                    self.fill_N_assoc_for_species(i, j, k, species)
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

                pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma("pion")
                self.pionTPCnSigma_pionTOFcut[i, j, k] = pionTPCnSigma_pionTOFcut
                del pionTPCnSigma_pionTOFcut

                pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma("kaon")
                self.pionTPCnSigma_kaonTOFcut[i, j, k] = pionTPCnSigma_kaonTOFcut
                del pionTPCnSigma_kaonTOFcut

                pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma("proton")
                self.pionTPCnSigma_protonTOFcut[i, j, k] = pionTPCnSigma_protonTOFcut
                del pionTPCnSigma_protonTOFcut

                self.fit_PionTPCNSigma(i, j, k)
                self.plot_PionTPCNSigmaFit(i,j, k)

                for species in ["pion", "proton", "kaon"]:
                    self.fill_dPhi_correlation_functions_for_species(i, j, k, species)
                    self.fill_dPhi_correlation_functions_for_species_in_z_vertex_bins(i, j, k, species)
            self.fill_N_trigs(i, j, 3)
            self.fit_RPF(i, j, p0=self.central_p0s[(i, j)])
            self.plot_RPF(i, j, withSignal=True)
            self.fit_RPF_in_z_vertex_bins(i, j, p0=self.central_p0s[(i, j)])
            self.plot_RPF_in_z_vertex_bins(i, j, withSignal=True)
            for species in ["pion", "proton", "kaon"]:
                self.fit_RPF_for_species(i, j, species, p0=self.central_p0s[(i, j)])
                self.fit_RPF_for_species_in_z_vertex_bins(i, j, species, p0=self.central_p0s[(i, j)])
                #self.plot_RPF_for_species(i, j, species, withSignal=True)
            if j == len(self.pTassocBinCenters):
                self.plot_optimal_parameters(i)
                self.plot_optimal_parameters_in_z_vertex_bins(i)
                for species in ["pion", "proton", "kaon"]:
                    self.plot_optimal_parameters_for_species(i, species)
                    self.plot_optimal_parameters_for_species_in_z_vertex_bins(i, species)
            jsonpayload = {
                "value1": f"RPF fitted for p_Ttrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, p_Tassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV"
            }
            url = "https://maker.ifttt.com/trigger/code_finished/with/key/bFQ9TznlbsocL7hbBL_sDyk33qkIJdDNVSIjhyJ7Mqm"
            try:
                requests.post(url, json=jsonpayload)
            except:
                print("Status notification failed, oh well")

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
                    self.fill_dPhi_dpionTPCnSigma_correlation_functions(
                        i, j, k, species
                    )
                    self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
                        i, j, k, species
                    )
                    self.fill_normalized_background_subtracted_dPhi_for_species(
                        i, j, k, species
                    )
                    self.fill_normalized_background_subtracted_dPhi_for_species_in_z_vertex_bins(
                        i, j, k, species
                    )
                    print(
                        f"Inclusive yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}: {self.get_yield_for_species(i,j,k,species)}"
                    )
                    print(
                        f"NS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}: {self.get_yield_for_species(i,j,k,species, 'NS')}"
                    )
                    print(
                        f"AS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}: {self.get_yield_for_species(i,j,k,species, 'AS')}"
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
            self.fill_N_assoc(i, j, 3)
        for species in ['pion', 'proton', 'kaon']:
            self.fill_N_assoc_for_species(i, j, 3, species)

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

        pionTPCsignal = self.get_pion_TPC_signal()
        self.pionTPCsignals[i, j] = pionTPCsignal
        del pionTPCsignal

        if self.analysisType in ["central", "semicentral"]:
            pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma("pion")
            self.pionTPCnSigma_pionTOFcut[i, j,3] = pionTPCnSigma_pionTOFcut
            del pionTPCnSigma_pionTOFcut

            pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma("kaon")
            self.pionTPCnSigma_kaonTOFcut[i, j,3] = pionTPCnSigma_kaonTOFcut
            del pionTPCnSigma_kaonTOFcut

            pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma("proton")
            self.pionTPCnSigma_protonTOFcut[i, j,3] = pionTPCnSigma_protonTOFcut
            del pionTPCnSigma_protonTOFcut
        else:
            pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma("pion")
            self.pionTPCnSigma_pionTOFcut[i, j] = pionTPCnSigma_pionTOFcut
            del pionTPCnSigma_pionTOFcut

            pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma("kaon")
            self.pionTPCnSigma_kaonTOFcut[i, j] = pionTPCnSigma_kaonTOFcut
            del pionTPCnSigma_kaonTOFcut

            pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma("proton")
            self.pionTPCnSigma_protonTOFcut[i, j] = pionTPCnSigma_protonTOFcut
            del pionTPCnSigma_protonTOFcut

        self.fit_PionTPCNSigma(i, j, 3)
        self.plot_PionTPCNSigmaFit(i, j, 3)
       

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
                self.fill_dPhi_correlation_functions_for_species(i, j, 3, species)
                self.fill_dPhi_dpionTPCnSigma_correlation_functions(i, j, 3, species)
                self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
                    i, j, 3, species
                )
                self.fill_normalized_background_subtracted_dPhi_for_species(
                    i, j, 3, species
                )
                self.fill_dPhi_correlation_functions_for_species_in_z_vertex_bins(i, j, 3, species)
                self.fill_normalized_background_subtracted_dPhi_for_species_in_z_vertex_bins(
                    i, j, 3, species
                )
                self.fill_yield_for_species(i, j, species)
                self.fill_yield_for_species_in_z_vertex_bins(i, j, species)
                
                print(
                    f"Inclusive yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle inclusive: {self.get_yield_for_species(i,j,3,species)}"
                )
                print(
                    f"NS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle inclusive: {self.get_yield_for_species(i,j,3,species, 'NS')}"
                )
                print(
                    f"AS yield for {species} in p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, event plane angle inclusive: {self.get_yield_for_species(i,j,3,species, 'AS')}"
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

    @print_function_name_with_description_on_call(description="")
    def fill_N_trigs(self, i, j, k):
        Ntrigs = self.get_N_trig()
        if self.analysisType in ["central", "semicentral"]:
            self.N_trigs[i, k] = Ntrigs
        elif self.analysisType == "pp":
            self.N_trigs[i] = Ntrigs
        del Ntrigs

    @print_function_name_with_description_on_call(description="")
    def fill_N_assoc(self, i, j, k):
        Nassoc = self.get_N_assoc()
        if self.analysisType in ["central", "semicentral"]:
            self.N_assoc[i, j, k] = Nassoc
        elif self.analysisType == "pp":
            self.N_assoc[i, j] = Nassoc
        del Nassoc

    @print_function_name_with_description_on_call(description="")
    def fill_N_assoc_for_species(self, i, j, k, species):
        Nassoc = self.get_N_assoc_for_species(species)
        if self.analysisType in ["central", "semicentral"]:
            self.N_assoc_for_species[species][i, j, k] = Nassoc
        elif self.analysisType == "pp":
            self.N_assoc_for_species[species][i, j] = Nassoc
        del Nassoc

    @print_function_name_with_description_on_call(description="")
    def fill_SE_correlation_function(self, i, j, k):
        SEcorr = self.get_SE_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.SEcorrs[i, j, k] = SEcorr
        elif self.analysisType == "pp":
            self.SEcorrs[i, j] = SEcorr
        del SEcorr

    @print_function_name_with_description_on_call(description="")
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

    @print_function_name_with_description_on_call(description="")
    def fill_AccCorrected_SE_correlation_function(self, i, j, k):
        AccCorrectedSEcorr = self.get_acceptance_corrected_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.AccCorrectedSEcorrs[i, j, k] = AccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.AccCorrectedSEcorrs[i, j] = AccCorrectedSEcorr
        del AccCorrectedSEcorr

    @print_function_name_with_description_on_call(description="")
    def fill_AccCorrected_SE_correlation_function_in_z_vertex_bins(self, i, j, k):
        AccCorrectedSEcorr = self.get_acceptance_corrected_correlation_function(in_z_vertex_bins=True)
        if self.analysisType in ["central", "semicentral"]:
            self.AccCorrectedSEcorrsZV[i, j, k] = AccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.AccCorrectedSEcorrsZV[i, j] = AccCorrectedSEcorr
        del AccCorrectedSEcorr

    @print_function_name_with_description_on_call(description="")
    def fill_NormAccCorrected_SE_correlation_function(self, i, j, k):
        NormAccCorrectedSEcorr = (
            self.get_normalized_acceptance_corrected_correlation_function()
        )
        if self.analysisType in ["central", "semicentral"]:
            self.NormAccCorrectedSEcorrs[i, j, k] = NormAccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.NormAccCorrectedSEcorrs[i, j] = NormAccCorrectedSEcorr
        del NormAccCorrectedSEcorr

    @print_function_name_with_description_on_call(description="")
    def fill_NormAccCorrected_SE_correlation_function_in_z_vertex_bins(self, i, j, k):
        NormAccCorrectedSEcorr = (
            self.get_normalized_acceptance_corrected_correlation_function(in_z_vertex_bins=True)
        )
        if self.analysisType in ["central", "semicentral"]:
            self.NormAccCorrectedSEcorrsZV[i, j, k] = NormAccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.NormAccCorrectedSEcorrsZV[i, j] = NormAccCorrectedSEcorr
        del NormAccCorrectedSEcorr

    @print_function_name_with_description_on_call(description="")
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

        print(f"Max of dPhiBGHi is {dPhiBGHi.GetMaximum()}")
        print(f"Max of dPhiBGLo is {dPhiBGLo.GetMaximum()}")
        print(f"Max of dPhiMid is {dPhiMid.GetMaximum()}")
        print(f"Number of bins in dPhiBGHi is {nbins_hi}")
        print(f"Number of bins in dPhiBGLo is {nbins_lo}")
        print(f"Number of bins in dPhiMid is {nbins_mid}")
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
    
    @print_function_name_with_description_on_call(description="")
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

        print(f"Max of dPhiBGHi is {dPhiBGHi.GetMaximum()}")
        print(f"Max of dPhiBGLo is {dPhiBGLo.GetMaximum()}")
        print(f"Max of dPhiMid is {dPhiMid.GetMaximum()}")
        print(f"Number of bins in dPhiBGHi is {nbins_hi}")
        print(f"Number of bins in dPhiBGLo is {nbins_lo}")
        print(f"Number of bins in dPhiMid is {nbins_mid}")
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

    @print_function_name_with_description_on_call(description="")
    def fill_dPhi_correlation_functions_for_species(self, i, j, k, species):

        dPhiMid, nbins_mid, binwidth = self.get_dPhi_projection_in_dEta_dpionTPCnSigma_range(i,j,k,
            self.dEtaSig, TOFcutSpecies=species
        )

        dPhiBGLo, nbins_lo, binwidth = self.get_dPhi_projection_in_dEta_dpionTPCnSigma_range(i,j,k,
            self.dEtaBGLo, TOFcutSpecies=species
        )

        dPhiBGHi, nbins_hi, binwidth = self.get_dPhi_projection_in_dEta_dpionTPCnSigma_range(i,j,k,
            self.dEtaBGHi, TOFcutSpecies=species
        )

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
            self.dPhiBGcorrsForSpecies[species][i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsForSpecies[species][i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiBGcorrsForSpecies[species][i, j] = dPhiBGcorrs
            self.dPhiSigcorrsForSpecies[species][i, j] = dPhiSig
        del dPhiSig
    
    @print_function_name_with_description_on_call(description="")
    def fill_dPhi_correlation_functions_for_species_in_z_vertex_bins(self, i, j, k, species):

        dPhiMid, nbins_mid, binwidth = self.get_dPhi_projection_in_dEta_dpionTPCnSigma_range(i,j,k,
            self.dEtaSig, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGLo, nbins_lo, binwidth = self.get_dPhi_projection_in_dEta_dpionTPCnSigma_range(i,j,k,
            self.dEtaBGLo, TOFcutSpecies=species, in_z_vertex_bins=True
        )

        dPhiBGHi, nbins_hi, binwidth = self.get_dPhi_projection_in_dEta_dpionTPCnSigma_range(i,j,k,
            self.dEtaBGHi, TOFcutSpecies=species, in_z_vertex_bins=True
        )

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
            self.dPhiBGcorrsForSpeciesZV[species][i, j, k] = dPhiBGcorrs
            self.dPhiSigcorrsForSpeciesZV[species][i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiBGcorrsForSpeciesZV[species][i, j] = dPhiBGcorrs
            self.dPhiSigcorrsForSpeciesZV[species][i, j] = dPhiSig
        del dPhiSig

    @print_function_name_with_description_on_call(description="")
    def fill_dPhi_dpionTPCnSigma_correlation_functions(self, i, j, k, TOFcutSpecies):

        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiSig = self.get_dPhi_dpionTPCnSigma_projection_in_dEta_range(
            self.dEtaSig, TOFcutSpecies
        ).Clone()
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiSigdpionTPCnSigmacorrs[TOFcutSpecies][i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiSigdpionTPCnSigmacorrs[TOFcutSpecies][i, j] = dPhiSig
        del dPhiSig

    @print_function_name_with_description_on_call(description="")
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
    
    @print_function_name_with_description_on_call(description="")
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

    @print_function_name_with_description_on_call(description="")
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
    
    @print_function_name_with_description_on_call(description="")
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

    @print_function_name_with_description_on_call(description="")
    def fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
        self, i, j, k, species
    ):

        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_species(
                i, j, k, species
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs[
                species
            ][i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_species(i, j, k, species)  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrsminVals[species][i, j] = minVal  # type: ignore
        del Corr

    @print_function_name_with_description_on_call(description="")
    def fill_normalized_background_subtracted_dPhi_for_species(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_background_subtracted_dPhi_for_species(
                i, j, k, species
            )
            self.NormalizedBGSubtracteddPhiForSpecies[species][i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_background_subtracted_dPhi_for_species(i, j, k, species)  # type: ignore
            self.NormalizedBGSubtracteddPhiForSpecies[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtracteddPhiForSpeciesminVals[species][i, j] = minVal  # type: ignore
        del Corr
    
    @print_function_name_with_description_on_call(description="")
    def fill_normalized_background_subtracted_dPhi_for_species_in_z_vertex_bins(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_background_subtracted_dPhi_for_species(
                i, j, k, species, in_z_vertex_bins=True
            )
            self.NormalizedBGSubtracteddPhiForSpeciesZV[species][i, j, k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_background_subtracted_dPhi_for_species(i, j, k, species, in_z_vertex_bins=True)  # type: ignore
            self.NormalizedBGSubtracteddPhiForSpeciesZV[species][i, j] = Corr  # type: ignore
            self.NormalizedBGSubtracteddPhiForSpeciesminValsZV[species][i, j] = minVal  # type: ignore
        del Corr

    @print_function_name_with_description_on_call(description="")
    def fill_yield_for_species(self, i, j, species):
        yield_, yield_err_ = self.get_yield_for_species(i, j, 3, species)
        self.Yields[species][i, j] = yield_
        self.YieldErrs[species][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_yield_for_species(
            i, j, 3, species, region="NS"
        )
        self.YieldsNS[species][i, j] = yield_NS
        self.YieldErrsNS[species][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_yield_for_species(
            i, j, 3, species, region="AS"
        )
        self.YieldsAS[species][i, j] = yield_AS
        self.YieldErrsAS[species][i, j] = yield_err_AS
    
    @print_function_name_with_description_on_call(description="")
    def fill_yield_for_species_in_z_vertex_bins(self, i, j, species):
        yield_, yield_err_ = self.get_yield_for_species(i, j, 3, species, in_z_vertex_bins=True)
        self.YieldsZV[species][i, j] = yield_
        self.YieldErrsZV[species][i, j] = yield_err_
        yield_NS, yield_err_NS = self.get_yield_for_species(
            i, j, 3, species, region="NS", in_z_vertex_bins=True
        )
        self.YieldsNSZV[species][i, j] = yield_NS
        self.YieldErrsNSZV[species][i, j] = yield_err_NS
        yield_AS, yield_err_AS = self.get_yield_for_species(
            i, j, 3, species, region="AS", in_z_vertex_bins=True
        )
        self.YieldsASZV[species][i, j] = yield_AS
        self.YieldErrsASZV[species][i, j] = yield_err_AS

    @print_function_name_with_description_on_call(description="")
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

    @print_function_name_with_description_on_call(description="")
    def set_pT_assoc_range(self, j_low, j_hi):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(2).SetRangeUser(
                self.pTassocBinEdges[j_low], self.pTassocBinEdges[j_hi]
            )
            self.MixedEvent[sparse_ind].GetAxis(2).SetRangeUser(
                self.pTassocBinEdges[j_low], self.pTassocBinEdges[j_hi]
            )
        self.set_has_changed()

    @print_function_name_with_description_on_call(description="")
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

    @print_function_name_with_description_on_call(description="")
    def set_z_vertex_bin(self, bin_no):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(5).SetRangeUser(self.z_vertex_bins[bin_no], self.z_vertex_bins[bin_no+1])
            self.MixedEvent[sparse_ind].GetAxis(5).SetRangeUser(self.z_vertex_bins[bin_no], self.z_vertex_bins[bin_no+1])
            self.Trigger[sparse_ind].GetAxis(2).SetRangeUser(self.z_vertex_bins[bin_no], self.z_vertex_bins[bin_no+1])
        self.set_has_changed()

    @print_function_name_with_description_on_call(description="")
    def reset_z_vertex_bin(self):
        for sparse_ind in range(len(self.JH)):
            self.JH[sparse_ind].GetAxis(5).SetRange(0, -1)
            self.MixedEvent[sparse_ind].GetAxis(5).SetRange(0, -1)
            self.Trigger[sparse_ind].GetAxis(2).SetRange(0, -1)
        self.set_has_changed()

    @print_function_name_with_description_on_call(description="")
    def set_has_changed(self):
        self.get_SE_correlation_function_has_changed = True
        self.get_SE_correlation_function_w_Pion_has_changed = self.init_bool_dict()
        self.get_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_error_has_changed = True
        self.get_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed = True
        self.get_normalized_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_correlation_function_w_pionTPCnSigma_has_changed = (
            self.init_bool_dict()
        )
        self.ME_norm_sliding_window_has_changed = True

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
        fHistEventPlane = ROOT.TH1F(anaList.FindObject("fHistEventPlane"))
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
