from collections import defaultdict
from itertools import product
from typing import Optional
import numpy as np

import matplotlib.pyplot as plt


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

    def __init__(self, rootFiles: list, analysisType, fill_on_init=True):
        """
        Initializes the location to save plots and grabs the main components of the rootFile
        """
        assert analysisType in ["central", "semicentral", "pp"]
        self.analysisType = analysisType
        self.base_save_path = (
            f"/home/steffanic/Projects/Thesis/backend_output/{analysisType}/"
        )
        # let's turn the sparses into lists of sparses to use all files
        self.JH, self.MixedEvent, self.Trigger = [], [], []
        self.EventPlaneAngleHist = None
        for file in rootFiles:
            fileJH, fileME, fileT = self.get_sparses(file)
            self.JH.append(fileJH)
            self.MixedEvent.append(fileME)
            self.Trigger.append(fileT)
            if self.EventPlaneAngleHist is None:
                self.EventPlaneAngleHist = self.get_event_plane_angle_hist(file)
            else:
                self.EventPlaneAngleHist.Add(self.get_event_plane_angle_hist(file))

        self.assert_sparses_filled()

        # define pT bins
        # see http://cds.cern.ch/record/1381321/plots#1 for motivation
        self.pTassocBinEdges = [
            0.15,
            0.5,
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
            0.325,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            4.5,
            5.5,
            8.0,
        ]  # subject to change based on statistics
        self.pTassocBinWidths = [
            0.35,
            0.5,
            0.5,
            0.5,
            1,
            1,
            1,
            1,
            4,
        ]  # subject to change based on statistics
        self.pTtrigBinEdges = [10, 20, 40, 100]  # subject to change based on statistics

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
                dtype=object,
            )
            self.SEcorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )  # Event plane angle has 4 bins, in-, mid-, out, and inclusive
            self.NormMEcorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.AccCorrectedSEcorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.NormAccCorrectedSEcorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.ME_norm_systematics = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.dPhiSigcorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )  # dPhiSigcorrs is the same as AccCorrectedSEcorrs, but with the signal region only
            self.dPhiSigcorrsForSpecies = (
                self.init_pTtrig_pTassoc_eventPlane_dict()
            )  # dPhiSigcorrs is the same as AccCorrectedSEcorrs, but with the signal region only
            self.dPhiBGcorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.dPhiSigdpionTPCnSigmacorrs = (
                self.init_pTtrig_pTassoc_eventPlane_dict()
            )  # dPhiSigcorrs is the same as AccCorrectedSEcorrs, but with the signal region only
            self.dEtacorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.RPFObjs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = np.zeros(
                (
                    len(self.pTtrigBinEdges) - 1,
                    len(self.pTassocBinEdges) - 1,
                    len(self.eventPlaneAngleBinEdges),
                ),
                dtype=object,
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs = (
                self.init_pTtrig_pTassoc_eventPlane_dict()
            )
            self.NormalizedBGSubtracteddPhiForSpecies = (
                self.init_pTtrig_pTassoc_eventPlane_dict()
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCsignals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.dPionNSsignals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.dPionASsignals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_pionTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_protonTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_kaonTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_electronTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )

            self.Yields = self.init_pTtrig_pTassoc_dict()
            self.YieldsNS = self.init_pTtrig_pTassoc_dict()
            self.YieldsAS = self.init_pTtrig_pTassoc_dict()

            self.YieldErrs = self.init_pTtrig_pTassoc_dict()
            self.YieldErrsNS = self.init_pTtrig_pTassoc_dict()
            self.YieldErrsAS = self.init_pTtrig_pTassoc_dict()

        else:
            self.N_trigs = np.zeros(
                (len(self.pTtrigBinEdges) - 1),
                dtype=object,
            )
            self.SEcorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.NormMEcorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.AccCorrectedSEcorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.NormAccCorrectedSEcorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.ME_norm_systematics = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.dPhiSigcorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.dPhiSigcorrsForSpecies = self.init_pTtrig_pTassoc_dict()
            self.dPhiBGcorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.dPhiSigdpionTPCnSigmacorrs = self.init_pTtrig_pTassoc_dict()
            self.dEtacorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.RPFObjs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs = (
                self.init_pTtrig_pTassoc_dict()
            )
            self.NormalizedBGSubtracteddPhiForSpecies = self.init_pTtrig_pTassoc_dict()
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrsminVals = (
                self.init_pTtrig_pTassoc_dict()
            )
            self.NormalizedBGSubtracteddPhiForSpeciesminVals = (
                self.init_pTtrig_pTassoc_dict()
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrs = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCsignals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.dPionNSsignals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.dPionASsignals = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_pionTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_protonTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_kaonTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )
            self.pionTPCnSigma_electronTOFcut = np.zeros(
                (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1),
                dtype=object,
            )

            self.Yields = self.init_pTtrig_pTassoc_dict()
            self.YieldsNS = self.init_pTtrig_pTassoc_dict()
            self.YieldsAS = self.init_pTtrig_pTassoc_dict()

            self.YieldErrs = self.init_pTtrig_pTassoc_dict()
            self.YieldErrsNS = self.init_pTtrig_pTassoc_dict()
            self.YieldErrsAS = self.init_pTtrig_pTassoc_dict()

        if fill_on_init:
            [
                [
                    self.fill_hist_arrays(i, j)
                    for j in range(len(self.pTassocBinEdges) - 1)
                ]
                for i in range(len(self.pTtrigBinEdges) - 1)
            ]

    def return_none(self):
        return None

    def return_true(self):
        return True

    def return_zeros_for_pTtrig_pTassoc(self):
        return np.zeros(
            (len(self.pTtrigBinEdges) - 1, len(self.pTassocBinEdges) - 1), dtype=object
        )

    def return_zeros_for_pTtrig_pTassoc_eventPlane(self):
        return np.zeros(
            (
                len(self.pTtrigBinEdges) - 1,
                len(self.pTassocBinEdges) - 1,
                len(self.eventPlaneAngleBinEdges),
            ),
            dtype=object,
        )

    def init_none_dict(self):
        return defaultdict(self.return_none)

    def init_bool_dict(self):
        return defaultdict(self.return_true)

    def init_pTtrig_pTassoc_dict(self):
        return defaultdict(self.return_zeros_for_pTtrig_pTassoc)

    def init_pTtrig_pTassoc_eventPlane_dict(self):
        return defaultdict(self.return_zeros_for_pTtrig_pTassoc_eventPlane)

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

                # fill the N_trigs array 
                if hists_to_fill is None or hists_to_fill.get("Ntrigs"):
                    self.fill_Ntrigs(i, j, k)
                # get the SE correlation function
                if hists_to_fill is None or hists_to_fill.get("SE"):
                    self.fill_SE_correlation_function(i, j, k)

                # get the ME correlation function
                if hists_to_fill is None or hists_to_fill.get("ME"):
                    self.fill_ME_correlation_function(i, j, k)

                # get the acceptance corrected SE correlation function
                if hists_to_fill is None or hists_to_fill.get("AccCorrectedSE"):
                    self.fill_AccCorrected_SE_correlation_function(i, j, k)

                # Get the number of triggers to normalize
                if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
                    self.fill_NormAccCorrected_SE_correlation_function(i, j, k)

                if hists_to_fill is None or hists_to_fill.get("dPhi"):
                    self.fill_dPhi_correlation_functions(i, j, k)

            self.fit_RPF(i, j)

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

        # get the SE correlation function
        if hists_to_fill is None or hists_to_fill.get("SE"):
            self.fill_SE_correlation_function(i, j, 3)

        # get the ME correlation function
        if hists_to_fill is None or hists_to_fill.get("ME"):
            self.fill_ME_correlation_function(i, j, 3)

        # get the acceptance corrected SE correlation function
        if hists_to_fill is None or hists_to_fill.get("AccCorrectedSE"):
            self.fill_AccCorrected_SE_correlation_function(i, j, 3)

        # Get the number of triggers to normalize
        if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
            self.fill_NormAccCorrected_SE_correlation_function(i, j, 3)

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

    @print_function_name_with_description_on_call(description="")
    def fill_N_trigs(self, i,j,k):
        Ntrigs = self.get_N_trig()
        if self.analysisType in ["central", "semicentral"]:
            self.N_trigs[i, j, k] = Ntrigs
        elif self.analysisType == "pp":
            self.N_trigs[i, j] = Ntrigs
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
            scaleForInclusive = 1 if k != 3 else 3
            dPhiSig.Scale(1 / scaleForInclusive)
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
            scaleForInclusive = 1 if k != 3 else 3
            dPhiSig.Scale(1 / scaleForInclusive)
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
            scaleForInclusive = 1 if k != 3 else 3
            dPhiSig.Scale(1 / scaleForInclusive)
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
        return f"JetHadron object for {self.analysisType} events"

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
        fHistEventPlane = anaList.FindObject("fHistEventPlane")
        return fHistEventPlane

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
