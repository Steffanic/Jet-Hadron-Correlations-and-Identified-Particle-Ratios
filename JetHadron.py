from itertools import product
from typing import Optional
import numpy as np

import matplotlib.pyplot as plt




from time import time

import warnings
warnings.filterwarnings("ignore")

import _JetHadronPlot, _JetHadronFit, _JetHadronAnalysis

# turn latex on in matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# increase the font size for christine 
plt.rcParams.update({'font.size': 16})



class JetHadron(_JetHadronPlot.PlotMixin, _JetHadronFit.FitMixin, _JetHadronAnalysis.AnalysisMixin):
    '''
    Class to track and contain all the steps along the way of my analysis for one centrality class
    '''
    def __init__(self, rootFile, analysisType, fill_on_init=True):
        '''
        Initializes the location to save plots and grabs the main components of the rootFile 
        '''
        assert analysisType in ["central", "semicentral", "pp"]
        self.analysisType = analysisType
        self.base_save_path = f"/home/steffanic/Projects/Thesis/backend_output/{analysisType}/"

        self.JH, self.MixedEvent, self.Trigger = self.get_sparses(rootFile)

        self.assert_sparses_filled()

        # define pT bins 
        # see http://cds.cern.ch/record/1381321/plots#1 for motivation
        self.pTassocBinEdges = [0.15, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0] # start at 500 MeV, feels like the average background level
        self.pTtrigBinEdges = [10, 20, 40, 100] # subject to change based on statistics

        # define event plane bins 
        self.eventPlaneAngleBinEdges = [0, np.pi/6, np.pi/3, np.pi/2]

        # 15 pt assoc bins * 7 pt trig bins * 4 event plane bins = 420 bins

        # define signal and background regions
        self.dEtaBGHi = [0.8, 1.21]
        self.dEtaBGLo = [-1.21, -0.8]
        self.dEtaSig = [-0.6, 0.6]
        self.dEtaSigAS = [-1.21, 1.21]

        self.get_SE_correlation_function_has_changed = True
        self.get_SE_correlation_function_w_Pion_has_changed = True
        self.get_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_error_has_changed = True
        self.get_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed = True
        self.get_normalized_acceptance_corrected_correlation_function_has_changed = True
        self.ME_norm_sliding_window_has_changed = True

        self.get_SE_correlation_function_result = None
        self.get_SE_correlation_function_w_Pion_result = None
        self.get_ME_correlation_function_result = None
        self.get_normalized_ME_correlation_function_result = None
        self.get_normalized_ME_correlation_error = None
        self.get_acceptance_corrected_correlation_function_result = None
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_result = None
        self.get_normalized_acceptance_corrected_correlation_function_result = None
        self.ME_norm_sliding_window_result = None

        if self.analysisType in ["central", "semicentral"]:
            # get the correlation functions 
            # an array of TH2s, one for each pTassoc, pTtrig, event plane angle bin
            self.SEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object) # Event plane angle has 4 bins, in-, mid-, out, and inclusive
            self.NormMEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.AccCorrectedSEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.NormAccCorrectedSEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.ME_norm_systematics = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.dPhiSigcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object) # dPhiSigcorrs is the same as AccCorrectedSEcorrs, but with the signal region only
            self.dPhiBGcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.dEtacorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.RPFObjs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, len(self.eventPlaneAngleBinEdges)), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.pionTPCsignals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.dPionNSsignals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.dPionASsignals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
        else:
            self.SEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.NormMEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.AccCorrectedSEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.NormAccCorrectedSEcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.ME_norm_systematics = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.dPhiSigcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.dPhiBGcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.dEtacorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.RPFObjs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrs = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.pionTPCsignals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.dPionNSsignals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)
            self.dPionASsignals = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1), dtype=object)


        if fill_on_init:
            [[self.fill_hist_arrays(i,j) for j in range(len(self.pTassocBinEdges)-1)] for i in range(len(self.pTtrigBinEdges)-1)]
            
    def fill_hist_arrays(self, i, j, hists_to_fill: "Optional[dict[str, bool]]" = None):
        if self.analysisType in ["central", "semicentral"]:
            for k in range(len(self.eventPlaneAngleBinEdges)-1):
                print(f"Getting correlation function for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV, event plane angle {self.eventPlaneAngleBinEdges[k]}-{self.eventPlaneAngleBinEdges[k+1]}")
                # set the ranges in the sparses 
                self.set_pT_epAngle_bin(i,j,k)
                self.assert_sparses_filled()
                
                # get the SE correlation function
                if hists_to_fill is None or hists_to_fill.get("SE"):
                    self.fill_SE_correlation_function(i,j,k)

                # get the ME correlation function
                if hists_to_fill is None or hists_to_fill.get("ME"):
                    self.fill_ME_correlation_function(i,j,k)

                # get the acceptance corrected SE correlation function
                if hists_to_fill is None or hists_to_fill.get("AccCorrectedSE"):
                    self.fill_AccCorrected_SE_correlation_function(i,j,k)

                # Get the number of triggers to normalize 
                if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
                    self.fill_NormAccCorrected_SE_correlation_function(i,j,k)

                if hists_to_fill is None or hists_to_fill.get("dPhi"):
                    self.fill_dPhi_correlation_functions(i,j, k)

            self.fit_RPF(i,j)
        
            # get the background subtracted correlation function
            for k in range(len(self.eventPlaneAngleBinEdges)-1):
                self.set_pT_epAngle_bin(i,j,k)
                self.assert_sparses_filled()
                self.fill_BG_subtracted_AccCorrected_SE_correlation_functions(i,j,k)
                self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(i,j,k)
        
        elif self.analysisType == "pp":
            print(f"Getting correlation function for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV")
            # set the ranges in the sparses
            self.set_pT_epAngle_bin(i,j,3)
            self.assert_sparses_filled()

        if self.analysisType in ["central", "semicentral"]:
            # then grab the inclusive event plane angle bin
            self.JH.GetAxis(5).SetRangeUser(self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[-1])
            self.MixedEvent.GetAxis(5).SetRangeUser(self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[-1])
            self.Trigger.GetAxis(2).SetRangeUser(self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[-1])
        self.set_has_changed()
        self.assert_sparses_filled()

        # get the SE correlation function
        if hists_to_fill is None or hists_to_fill.get("SE"):
            self.fill_SE_correlation_function(i,j,3)

        # get the ME correlation function
        if hists_to_fill is None or hists_to_fill.get("ME"):
            self.fill_ME_correlation_function(i,j,3)

        # get the acceptance corrected SE correlation function
        if hists_to_fill is None or hists_to_fill.get("AccCorrectedSE"):
            self.fill_AccCorrected_SE_correlation_function(i,j,3)

        # Get the number of triggers to normalize
        if hists_to_fill is None or hists_to_fill.get("NormAccCorrectedSE"):
            self.fill_NormAccCorrected_SE_correlation_function(i,j,3)

        if hists_to_fill is None or hists_to_fill.get("dPhi"):
            self.fill_dPhi_correlation_functions(i,j, 3)

        dEta = self.get_dEta_projection_NS()
        self.dEtacorrs[i,j] = dEta
        del dEta

        pionTPCsignal = self.get_pion_TPC_signal()
        self.pionTPCsignals[i,j] = pionTPCsignal
        del pionTPCsignal

        NSdPhidPion = self.get_BG_subtracted_AccCorrectedSE_dPhi_dEta_dPion_NS(i,j).Clone()
        self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs[i,j] = NSdPhidPion
        del NSdPhidPion

        ASdPhidPion = self.get_BG_subtracted_AccCorrectedSE_dPhi_dEta_dPion_AS(i,j).Clone()
        self.BGSubtractedAccCorrectedSEdPhidPionSigAScorrs[i,j] = ASdPhidPion
        del ASdPhidPion

        NSdPion = self.get_pion_TPC_signal_for_BG_subtracted_NS_signal(i,j, normalize=True).Clone()
        self.dPionNSsignals[i,j] = NSdPion
        del NSdPion

        ASdPion = self.get_pion_TPC_signal_for_BG_subtracted_AS_signal(i,j, normalize=True).Clone()
        self.dPionASsignals[i,j] = ASdPion
        del ASdPion

        # get the background subtracted correlation functions
        if hists_to_fill is None or hists_to_fill.get("BGSubtractedSE"):
            self.fill_BG_subtracted_AccCorrected_SE_correlation_functions(i,j,3)
            self.fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(i,j,3)

        print("\a")
    
    def fill_SE_correlation_function(self, i, j, k):
        SEcorr = self.get_SE_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.SEcorrs[i,j,k] = SEcorr
        elif self.analysisType == "pp":
            self.SEcorrs[i,j] = SEcorr
        del SEcorr

    def fill_ME_correlation_function(self, i, j, k):
        NormMEcorr, ME_norm_error = self.get_normalized_ME_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.NormMEcorrs[i,j,k] = NormMEcorr
            self.ME_norm_systematics[i,j,k] = ME_norm_error
        elif self.analysisType == "pp":
            self.NormMEcorrs[i,j] = NormMEcorr
            self.ME_norm_systematics[i,j] = ME_norm_error
        del NormMEcorr
    
    def fill_AccCorrected_SE_correlation_function(self, i, j, k):
        AccCorrectedSEcorr= self.get_acceptance_corrected_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.AccCorrectedSEcorrs[i,j,k] = AccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.AccCorrectedSEcorrs[i,j] = AccCorrectedSEcorr
        del AccCorrectedSEcorr

    def fill_NormAccCorrected_SE_correlation_function(self, i, j, k):
        NormAccCorrectedSEcorr = self.get_normalized_acceptance_corrected_correlation_function()
        if self.analysisType in ["central", "semicentral"]:
            self.NormAccCorrectedSEcorrs[i,j,k] = NormAccCorrectedSEcorr
        elif self.analysisType == "pp":
            self.NormAccCorrectedSEcorrs[i,j] = NormAccCorrectedSEcorr
        del NormAccCorrectedSEcorr

    def fill_dPhi_correlation_functions(self, i, j, k):
        dPhiBGHi = self.get_dPhi_projection_in_dEta_range(self.dEtaBGHi, scaleUp=True).Clone()
        #print(f"dPhiBGHi is at {hex(id(dPhiBGHi))}")
        dPhiBGLo = self.get_dPhi_projection_in_dEta_range(self.dEtaBGLo, scaleUp=True).Clone()
        #print(f"dPhiBGLo is at {hex(id(dPhiBGLo))}")
        dPhiBG = dPhiBGHi
        #print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiBG.Add(dPhiBGLo, dPhiBGHi)
        #print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiBG.Scale(0.5)
        #print(f"dPhiBG is at {hex(id(dPhiBG))}")
        dPhiSig = self.get_dPhi_projection_in_dEta_range(self.dEtaSig, ).Clone()
        if self.analysisType in ["central", "semicentral"]:
            self.dPhiBGcorrs[i,j,k] = dPhiBG
            self.dPhiSigcorrs[i,j,k] = dPhiSig
        elif self.analysisType == "pp":
            self.dPhiBGcorrs[i,j] = dPhiBG
            self.dPhiSigcorrs[i,j] = dPhiSig
        del dPhiBGHi, dPhiBGLo, dPhiBG, dPhiSig

    def fill_BG_subtracted_AccCorrected_SE_correlation_functions(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            NSCorr = self.get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(i,j,k) # type: ignore
            ASCorr= self.get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(i,j,k) # type: ignore
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k] = NSCorr
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k] = ASCorr
        elif self.analysisType == "pp":
            NSCorr, NSminVal = self.get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(i,j,k) # type: ignore
            ASCorr, ASminVal = self.get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(i,j,k) # type: ignore
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j] = NSCorr
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j] = ASCorr
            self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals[i,j] = NSminVal
            self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals[i,j] = ASminVal
        del NSCorr, ASCorr

    def fill_normalized_BG_subtracted_AccCorrected_SE_correlation_functions(self, i, j, k):
        
        if self.analysisType in ["central", "semicentral"]:
            Corr = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(i,j,k)
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k] = Corr
        elif self.analysisType == "pp":
            Corr, minVal = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(i,j,k) # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j] = Corr # type: ignore
            self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[i,j] = minVal # type: ignore
        del Corr

    def set_pT_epAngle_bin(self, i,j,k):

        # set the pT and event plane angle ranges
        self.JH.GetAxis(1).SetRangeUser(self.pTtrigBinEdges[i], self.pTtrigBinEdges[i+1]) 
        self.MixedEvent.GetAxis(1).SetRangeUser(self.pTtrigBinEdges[i], self.pTtrigBinEdges[i+1]) 
        self.Trigger.GetAxis(1).SetRangeUser(self.pTtrigBinEdges[i], self.pTtrigBinEdges[i+1])

        self.JH.GetAxis(2).SetRangeUser(self.pTassocBinEdges[j], self.pTassocBinEdges[j+1])
        self.MixedEvent.GetAxis(2).SetRangeUser(self.pTassocBinEdges[j], self.pTassocBinEdges[j+1])
        if self.analysisType in ["central", "semicentral"]:
            if k==3:
                self.JH.GetAxis(5).SetRangeUser(self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[3])
                self.MixedEvent.GetAxis(5).SetRangeUser(self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[3])
                self.Trigger.GetAxis(2).SetRangeUser(self.eventPlaneAngleBinEdges[0], self.eventPlaneAngleBinEdges[3])
            else:
                self.JH.GetAxis(5).SetRangeUser(self.eventPlaneAngleBinEdges[k], self.eventPlaneAngleBinEdges[k+1])
                #self.MixedEvent.GetAxis(5).SetRangeUser(self.eventPlaneAngleBinEdges[k], self.eventPlaneAngleBinEdges[k+1])
                self.Trigger.GetAxis(2).SetRangeUser(self.eventPlaneAngleBinEdges[k], self.eventPlaneAngleBinEdges[k+1])

        self.set_has_changed()

    def set_has_changed(self):
        self.get_SE_correlation_function_has_changed = True
        self.get_SE_correlation_function_w_Pion_has_changed = True
        self.get_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_function_has_changed = True
        self.get_normalized_ME_correlation_error_has_changed = True
        self.get_acceptance_corrected_correlation_function_has_changed = True
        self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed = True
        self.get_normalized_acceptance_corrected_correlation_function_has_changed = True
        self.ME_norm_sliding_window_has_changed = True
    
    def __repr__(self) -> str:
        return f"JetHadron object for {self.analysisType} events"

    def get_sparses(self, f):
        '''
        Returns tuple of (fhnJH, fhnMixedEvent, fhnTrigger) for central events
        '''
        if self.analysisType in ['central', 'semicentral']:
            centralityString = "Central" if self.analysisType=="central" else "SemiCentral"
            anaList = f.Get(f'AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2{centralityString}q')
        else:
            anaList = f.Get('AliAnalysisTaskJetH_tracks_caloClusters_biased')
        fhnJH = anaList.FindObject('fhnJH')
        fhnMixedEvent = anaList.FindObject('fhnMixedEvents')
        fhnTrigger = anaList.FindObject('fhnTrigger')
        #turn on errors with sumw2
        
        return fhnJH, fhnMixedEvent, fhnTrigger

    def assert_sparses_filled(self):
        assert self.JH.GetEntries()!=0, f"⚠️⚠️ No entries for {self.analysisType} JH sparse. ⚠️⚠️"
        assert self.MixedEvent.GetEntries()!=0, f"⚠️⚠️ No entries for {self.analysisType} Mixed Event sparse. ⚠️⚠️"
        assert self.Trigger.GetEntries()!=0, f"⚠️⚠️ No entries for {self.analysisType} Trigger sparse. ⚠️⚠️"

    