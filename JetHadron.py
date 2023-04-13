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

    def __init__(self, rootFileNames: list, analysisType, fill_on_init=True):
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
        self.central_p0s = {(i,j):[1, 0.02, 0.005, 0.02, 0.05, 0.03,] for i in range(len(self.pTtrigBinCenters)) for j in range(len(self.pTassocBinCenters))}
        self.central_p0s[(0,0)] = [1000042.8, 0.0473, -.000306, 0.02, 0.0513, 0.03,] # pTtrig 20-40, pTassoc 1.0-1.5   
        self.central_p0s[(0,1)] = [40000.19, 0.0402, -0.0058, 0.02, 0.0906, 0.03,] # pTtrig 20-40, pTassoc 1.5-2.0
        self.central_p0s[(0,2)] = [4006.86, 0.0414, 0.0015, 0.02, 0.1034, 0.03,] # pTtrig 20-40, pTassoc 2.0-3.0
        self.central_p0s[(0,3)] = [56.84, 0.0636, -0.00766, 0.02, 0.1237, 0.03,] # pTtrig 20-40, pTassoc 3.0-4.0
        self.central_p0s[(0,4)] = [8.992, 0.1721, -0.0987, 0.02, 0.233, 0.03,] # pTtrig 20-40, pTassoc 4.0-5.0
        self.central_p0s[(0,5)] = [2.318, -0.0508, -0.143, 0.02, 0.0876, 0.03,] # pTtrig 20-40, pTassoc  5.0-6.0
        self.central_p0s[(0,6)] = [2.076, -0.0886, 0.06929, 0.02, 0.0692, 0.03,] # pTtrig 20-40, pTassoc 6.0-10.0
        self.central_p0s[(1,0)] = [1, 0.02, 0.005, 0.02, 0.05, 0.03,] # pTtrig 40-60, pTassoc 1.0-1.5
        self.central_p0s[(1,1)] = [1, 0.02, 0.005, 0.02, 0.05, 0.03,] # pTtrig 40-60, pTassoc 1.5-2.0
        self.central_p0s[(1,2)] = [1, 0.02, 0.005, 0.02, 0.05, 0.03,] # pTtrig 40-60, pTassoc 2.0-3.0
        self.central_p0s[(1,3)] = [1, 0.02, 0.005, 0.02, 0.05, 0.03,] # pTtrig 40-60, pTassoc 3.0-4.0
        self.central_p0s[(1,4)] = [1, 0.02, 0.005, 0.02, 0.05, 0.03,] # pTtrig 40-60, pTassoc    4.0-5.0
        self.central_p0s[(1,5)] = [1, 0.02, 0.005, 0.02, 0.05, 0.03,] # pTtrig 40-60, pTassoc 5.0-6.0
        self.central_p0s[(1,6)] = [1, 0.02, 0.005, 0.02, 0.05, 0.03,] # pTtrig 40-60, pTassoc 6.0-10.0
        # define event plane bins
        self.eventPlaneAngleBinEdges = [0, np.pi / 6, np.pi / 3, np.pi / 2]

        # 15 pt assoc bins * 7 pt trig bins * 4 event plane bins = 420 bins
        
        # define signal and background regions
        self.dEtaBGHi = [0.8, 1.21]
        self.dEtaBGLo = [-1.21, -0.8]
        self.dEtaSig = [-0.6, 0.6]
        self.dEtaSigAS = [-1.21, 1.21]
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
            self.SEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )  # Event plane angle has 4 bins, in-, mid-, out, and inclusive
            self.NormMEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH2F)
            self.AccCorrectedSEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )
            self.NormAccCorrectedSEcorrs = self.init_pTtrig_pTassoc_eventPlane_array(
                ROOT.TH2F
            )
            self.ME_norm_systematics = self.init_pTtrig_pTassoc_eventPlane_array(float)
            self.dPhiSigcorrs = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.dPhiSigcorrsForSpecies = self.init_pTtrig_pTassoc_eventPlane_dict(
                ROOT.TH1F
            )
            self.dPhiBGcorrs = self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            self.dPhiSigdpionTPCnSigmacorrs = self.init_pTtrig_pTassoc_eventPlane_dict(
                ROOT.TH2F
            )
            self.dEtacorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.RPFObjs = self.init_pTtrig_pTassoc_array(object)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = (
                self.init_pTtrig_pTassoc_eventPlane_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs = (
                self.init_pTtrig_pTassoc_eventPlane_dict(ROOT.TH2F)
            )
            self.NormalizedBGSubtracteddPhiForSpecies = (
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
            self.pionTPCnSigma_pionTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_protonTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_kaonTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_electronTOFcut = self.init_pTtrig_pTassoc_array(
                ROOT.TH1F
            )

            self.Yields = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsAS = self.init_pTtrig_pTassoc_dict(float)

            self.YieldErrs = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsAS = self.init_pTtrig_pTassoc_dict(float)

        else:
            self.N_trigs = np.zeros(
                (len(self.pTtrigBinEdges) - 1),
                dtype=object,
            )
            self.SEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.NormMEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.AccCorrectedSEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.NormAccCorrectedSEcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH2F)
            self.ME_norm_systematics = self.init_pTtrig_pTassoc_array(float)
            self.dPhiSigcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPhiSigcorrsForSpecies = self.init_pTtrig_pTassoc_dict(ROOT.TH1F)
            self.dPhiBGcorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPhiSigdpionTPCnSigmacorrs = self.init_pTtrig_pTassoc_dict(ROOT.TH2F)
            self.dEtacorrs = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.RPFObjs = self.init_pTtrig_pTassoc_array(object)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = (
                self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = (
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
            self.pionTPCsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPionNSsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.dPionASsignals = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_pionTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_protonTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_kaonTOFcut = self.init_pTtrig_pTassoc_array(ROOT.TH1F)
            self.pionTPCnSigma_electronTOFcut = self.init_pTtrig_pTassoc_array(
                ROOT.TH1F
            )

            self.Yields = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldsAS = self.init_pTtrig_pTassoc_dict(float)

            self.YieldErrs = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsNS = self.init_pTtrig_pTassoc_dict(float)
            self.YieldErrsAS = self.init_pTtrig_pTassoc_dict(float)

        

        if fill_on_init:
            [
                [
                    self.fill_hist_arrays(i, j)
                    for j in range(len(self.pTassocBinEdges) - 1)
                ]
                for i in range(len(self.pTtrigBinEdges) - 1)
            ]
        # pickle the object
        pickle_filename = "jhAnapp.pickle" if self.analysisType == "pp" else "jhAnaCentral.pickle" if self.analysisType == "central" else "jhAnaSemiCentral.pickle"
        pickle.dump(self, open(pickle_filename, "wb"))
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
                    self.plot_acceptance_corrected_SE_correlation_function(i,j,k)

                # Get the number of triggers to normalize
                if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
                    self.fill_NormAccCorrected_SE_correlation_function(i, j, k)
                    self.plot_normalized_acceptance_corrected_correlation_function(i,j,k)

                if hists_to_fill is None or hists_to_fill.get("dPhi"):
                    self.fill_dPhi_correlation_functions(i, j, k)

            self.fit_RPF(i, j, p0=self.central_p0s[(i,j)])
            self.plot_RPF(i,j, withSignal=True)
            jsonpayload ={"value1":f"RPF fitted for p_Ttrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, p_Tassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV"}
            url = 'https://maker.ifttt.com/trigger/code_finished/with/key/bFQ9TznlbsocL7hbBL_sDyk33qkIJdDNVSIjhyJ7Mqm'
            requests.post(url, json=jsonpayload)

            # get the background subtracted correlation function
            for k in range(len(self.eventPlaneAngleBinEdges) - 1):
                self.set_pT_epAngle_bin(i, j, k)
                self.assert_sparses_filled()
                self.fill_BG_subtracted_AccCorrected_SE_correlation_functions(i, j, k)
                self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(
                    i, j, k
                )
                for species in ["pion", "proton", "kaon", "electron"]:
                    self.fill_dPhi_correlation_functions_for_species(i, j, k, species)
                    self.fill_dPhi_dpionTPCnSigma_correlation_functions(
                        i, j, k, species
                    )
                    self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
                        i, j, k, species
                    )
                    self.fill_normalized_background_subtracted_dPhi_for_species(
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
            self.epString = (
                        "inclusive"
                    )
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

        # get the SE correlation function
        if hists_to_fill is None or hists_to_fill.get("SE"):
            self.fill_SE_correlation_function(i, j, 3)
            self.plot_SE_correlation_function(i,j,3)

        # get the ME correlation function
        if hists_to_fill is None or hists_to_fill.get("ME"):
            self.fill_ME_correlation_function(i, j, 3)
            self.plot_ME_correlation_function(i,j,3)

        # get the acceptance corrected SE correlation function
        if hists_to_fill is None or hists_to_fill.get("AccCorrectedSE"):
            self.fill_AccCorrected_SE_correlation_function(i, j, 3)
            self.plot_acceptance_corrected_SE_correlation_function(i,j,3)

        # Get the number of triggers to normalize
        if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
            self.fill_NormAccCorrected_SE_correlation_function(i, j, 3)
            self.plot_normalized_acceptance_corrected_correlation_function(i,j,3)

        if hists_to_fill is None or hists_to_fill.get("dPhi"):
            self.fill_dPhi_correlation_functions(i, j, 3)
        dEta = self.get_dEta_projection_NS()
        self.dEtacorrs[i, j] = dEta
        del dEta

        pionTPCsignal = self.get_pion_TPC_signal()
        self.pionTPCsignals[i, j] = pionTPCsignal
        del pionTPCsignal

        pionTPCnSigma_pionTOFcut = self.get_pion_TPC_nSigma("pion")
        self.pionTPCnSigma_pionTOFcut[i, j] = pionTPCnSigma_pionTOFcut
        del pionTPCnSigma_pionTOFcut

        pionTPCnSigma_kaonTOFcut = self.get_pion_TPC_nSigma("kaon")
        self.pionTPCnSigma_kaonTOFcut[i, j] = pionTPCnSigma_kaonTOFcut
        del pionTPCnSigma_kaonTOFcut

        pionTPCnSigma_protonTOFcut = self.get_pion_TPC_nSigma("proton")
        self.pionTPCnSigma_protonTOFcut[i, j] = pionTPCnSigma_protonTOFcut
        del pionTPCnSigma_protonTOFcut

        pionTPCnSigma_electronTOFcut = self.get_pion_TPC_nSigma("electron")
        self.pionTPCnSigma_electronTOFcut[i, j] = pionTPCnSigma_electronTOFcut
        del pionTPCnSigma_electronTOFcut

        # get the background subtracted correlation functions
        if hists_to_fill is None or hists_to_fill.get("BGSubtractedSE"):
            self.fill_BG_subtracted_AccCorrected_SE_correlation_functions(i, j, 3)
            self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(
                i, j, 3
            )
            for species in ["pion", "proton", "kaon", "electron"]:
                self.fill_dPhi_correlation_functions_for_species(i, j, 3, species)
                self.fill_dPhi_dpionTPCnSigma_correlation_functions(i, j, 3, species)
                self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions_w_pionTPCnSigma(
                    i, j, 3, species
                )
                self.fill_normalized_background_subtracted_dPhi_for_species(
                    i, j, 3, species
                )
                self.fill_yield_for_species(i, j, species)
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
        p=subprocess.Popen(['bash', '-c', f'. ~/.notifyme; notifyme "Finished with p_T^assoc bin {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, p_T^trig bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV";'])
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
    def fill_SE_correlation_function(self, i, j, k):
        SEcorr = self.get_SE_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.SEcorrs[i, j, k] = SEcorr
        elif self.analysisType == "pp":
            self.SEcorrs[i, j] = SEcorr
        del SEcorr

    @print_function_name_with_description_on_call(description="")
    def fill_ME_correlation_function(self, i, j, k):
        if self.analysisType == "pp" and j >= 2:
            self.set_pT_assoc_range(2, 7)
            self.set_pT_trig_range(0,2)
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
    def fill_dPhi_correlation_functions(self, i, j, k):
        dPhiBGHi = self.get_dPhi_projection_in_dEta_range(
            self.dEtaBGHi, scaleUp=True
        ).Clone()
        # print(f"dPhiBGHi is at {hex(id(dPhiBGHi))}")
        dPhiBGLo = self.get_dPhi_projection_in_dEta_range(
            self.dEtaBGLo, scaleUp=True
        ).Clone()
        # print(f"dPhiBGLo is at {hex(id(dPhiBGLo))}")
        dPhiBG = dPhiBGHi
        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiBG.Add(dPhiBGLo, dPhiBGHi)
        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiBG.Scale(0.5)
        # print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiSig = self.get_dPhi_projection_in_dEta_range(
            self.dEtaSig,
        ).Clone()
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrs[i, j, k] = dPhiBG
            self.dPhiSigcorrs[i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiBGcorrs[i, j] = dPhiBG
            self.dPhiSigcorrs[i, j] = dPhiSig
        del dPhiBGHi, dPhiBGLo, dPhiBG, dPhiSig

    @print_function_name_with_description_on_call(description="")
    def fill_dPhi_correlation_functions_for_species(self, i, j, k, species):

        dPhiSig = self.get_dPhi_projection_in_dEta_dpionTPCnSigma_range(
            self.dEtaSig, TOFcutSpecies=species
        ).Clone()
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiSigcorrsForSpecies[species][i, j, k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiSigcorrsForSpecies[species][i, j] = dPhiSig
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
