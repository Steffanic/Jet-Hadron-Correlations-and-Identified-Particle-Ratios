# © Patrick John Steffanic 2023
# This file contains the class whose responsibility it is to manage the analysis and its configuration 

import numpy as np
import os
import uncertainties
from JetHadronAnalysis.Sparse import TriggerSparse, MixedEventSparse, JetHadronSparse
from JetHadronAnalysis.Types import AnalysisType, ParticleType, NormalizationMethod, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Background import BackgroundFunction
from JetHadronAnalysis.TPCPionNsigmaFit import FitTPCPionNsigma
from ROOT import TFile, TH1D # type: ignore
from enum import Enum
from math import pi

from JetHadronAnalysis.Plotting import plotArrays


    
regionDeltaPhiRangeDictionary = {
    Region.NEAR_SIDE_SIGNAL: [-pi / 2, pi / 2],
    Region.AWAY_SIDE_SIGNAL: [pi / 2, 3 * pi / 2],
    Region.BACKGROUND_ETAPOS: [-pi / 2, pi / 2],
    Region.BACKGROUND_ETANEG: [-pi / 2, pi / 2],
    Region.INCLUSIVE: [-pi / 2, 3 * pi / 2]
}

regionDeltaEtaRangeDictionary = {
    Region.NEAR_SIDE_SIGNAL: [-0.6, 0.6],
    Region.AWAY_SIDE_SIGNAL: [-1.2, 1.2],
    Region.BACKGROUND_ETAPOS: [0.8, 1.2],
    Region.BACKGROUND_ETANEG: [-1.2, -0.8],
    Region.INCLUSIVE: [-1.4, 1.4]
}

regionDeltaPhiBinCountsDictionary = {
    Region.NEAR_SIDE_SIGNAL: 36,
    Region.AWAY_SIDE_SIGNAL: 36,
    Region.BACKGROUND_ETAPOS: 36,
    Region.BACKGROUND_ETANEG: 36,
    Region.INCLUSIVE: 72
}

speciesTOFRangeDictionary = {
    ParticleType.PION: ([-2,2], [-10,-2], [-10,-2]),#  -10 includes the underflow bin
    ParticleType.KAON: ([-10,10], [-2,2], [-10,-2]),
    ParticleType.PROTON: ([-10,10], [-10,10], [-2,2]),
    ParticleType.INCLUSIVE: ([-10,10], [-10,10], [-10,10]),
}

associatedHadronMomentumBinRangeDictionary = {
    AssociatedHadronMomentumBin.PT_1_15: (1, 1.5),
    AssociatedHadronMomentumBin.PT_15_2: (1.5, 2),
    AssociatedHadronMomentumBin.PT_2_3: (2, 3),
    AssociatedHadronMomentumBin.PT_3_4: (3, 4),
    AssociatedHadronMomentumBin.PT_4_5: (4, 5),
    AssociatedHadronMomentumBin.PT_5_6: (5, 6),
    AssociatedHadronMomentumBin.PT_6_10: (6, 10),
}


class Analysis:
    '''
    This class is responsible for managing the analysis and its configuration. 
    '''

    def __init__(self, analysisType: AnalysisType, rootFileNames: list):
        self.analysisType = analysisType

        self.JetHadron = JetHadronSparse(analysisType)
        self.Trigger = TriggerSparse(analysisType)
        self.MixedEvent = MixedEventSparse(analysisType)

        self.currentRegion = Region.INCLUSIVE

        self.currentAssociatedHadronMomentumBin = AssociatedHadronMomentumBin.PT_1_15

        self.current_species=ParticleType.INCLUSIVE

        for rootFileName in rootFileNames:
            self.fillSparsesFromFile(rootFileName)


    def fillSparsesFromFile(self, rootFileName: str):
        '''
        This function opens the root file and appends the sparse data to the appropriate Sparse object then closes the file
        '''
        file = TFile(rootFileName)
        rootFileListName = f"AliAnalysisTaskJetH_tracks_caloClusters_biased" if self.analysisType == AnalysisType.PP else f"AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2{'SemiCentral' if self.analysisType == AnalysisType.SEMICENTRAL else 'Central'}q"
        rootFileList = file.Get(rootFileListName)
        self.JetHadron.addSparse(rootFileList.FindObject("fhnJH"))
        self.Trigger.addSparse(rootFileList.FindObject("fhnTrigger"))
        self.MixedEvent.addSparse(rootFileList.FindObject("fhnMixedEvents"))
        file.Close()

    def setRegion(self, region: Region):
        '''
        Sets the delta-phi and delta-eta ranges for the JetHadron sparse 
        '''
        self.currentRegion = region
        self.JetHadron.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[region])
        self.JetHadron.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[region])
        if hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()

    def setAssociatedHadronMomentumBin(self, associatedHadronMomentumBin: AssociatedHadronMomentumBin):
        '''
        Sets the associated hadron momentum bin for the JetHadron sparse
        '''
        self.currentAssociatedHadronMomentumBin = associatedHadronMomentumBin
        self.JetHadron.setAssociatedHadronMomentumRange(*associatedHadronMomentumBinRangeDictionary[associatedHadronMomentumBin])
        if hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()

    def setParticleSelectionForJetHadron(self, species: ParticleType):
        '''
        Sets the pion, kaon, and proton TOF ranges to get the particle type specified in JetHadron sparse
        '''
        self.current_species = species
        if species == ParticleType.OTHER:
            self.JetHadron.setParticleTypeIsOther(True)
        else:
            self.JetHadron.setParticleTypeIsOther(False)
            self.JetHadron.setPionTOFnSigma(*speciesTOFRangeDictionary[species][0])
            self.JetHadron.setKaonTOFnSigma(*speciesTOFRangeDictionary[species][1])
            self.JetHadron.setProtonTOFnSigma(*speciesTOFRangeDictionary[species][2])

    def getDifferentialCorrelationFunction(self, per_trigger_normalized=False):
        '''
        Returns the differential correlation function
        '''
        correlationFunction = self.JetHadron.getProjection(self.JetHadron.Axes.DELTA_PHI, self.JetHadron.Axes.DELTA_ETA)
        correlationFunction.Scale(1 / self.JetHadron.getBinWidth(self.JetHadron.Axes.DELTA_PHI) / self.JetHadron.getBinWidth(self.JetHadron.Axes.DELTA_ETA))
        if per_trigger_normalized:
            correlationFunction.Scale(1 / self.getNumberOfTriggerJets())
        return correlationFunction
    
    def getAcceptanceCorrectedDifferentialCorrelationFunction(self, differentialCorrelationFunction, acceptanceCorrection):
        '''
        Returns the acceptance corrected differential correlation function
        This is the raw differential correlation function divided by the normalized mixed event correlation function
        '''
        acceptanceCorrectedDifferentialCorrelationFunction = differentialCorrelationFunction.Clone()
        acceptanceCorrection = acceptanceCorrection.Clone()
        # Set the x and y bin ranges in the acceptance correction to match the raw correlation function
        acceptanceCorrection.GetXaxis().SetRangeUser(differentialCorrelationFunction.GetXaxis().GetXmin(), differentialCorrelationFunction.GetXaxis().GetXmax())
        acceptanceCorrection.GetYaxis().SetRangeUser(differentialCorrelationFunction.GetYaxis().GetXmin(), differentialCorrelationFunction.GetYaxis().GetXmax())
        acceptanceCorrectedDifferentialCorrelationFunction.Divide(acceptanceCorrection)
        return acceptanceCorrectedDifferentialCorrelationFunction
    
    def getAcceptanceCorrectedDifferentialAzimuthalCorrelationFunction(self, acceptanceCorrectedDifferentialCorrelationFunction):
        '''
        Returns the acceptance corrected differential azimuthal correlation function
        This is the acceptance corrected differential correlation function integrated over delta-eta
        '''
        acceptanceCorrectedDifferentialAzimuthalCorrelationFunction = acceptanceCorrectedDifferentialCorrelationFunction.ProjectionX()
        acceptanceCorrectedDifferentialAzimuthalCorrelationFunction.Scale(self.JetHadron.getBinWidth(self.JetHadron.Axes.DELTA_ETA))
        # scale by the number of bins in delta-eta over which the correlation function was integrated
        number_of_delta_eta_bins = acceptanceCorrectedDifferentialCorrelationFunction.GetYaxis().GetNbins()
        acceptanceCorrectedDifferentialAzimuthalCorrelationFunction.Scale(1 / number_of_delta_eta_bins)
        return acceptanceCorrectedDifferentialAzimuthalCorrelationFunction

    def getAcceptanceCorrectedBackgroundSubtractedDifferentialAzimuthalCorrelationFunction(self, acceptanceCorrectedDifferentialAzimuthalCorrelationFunction: TH1D, backgroundFunction:TH1D):
        '''
        Returns the acceptance corrected background subtracted differential correlation function
        This is the acceptance corrected differential correlation function minus the background function
        '''
        acceptanceCorrectedBackgroundSubtractedDifferentialAzimuthalCorrelationFunction = acceptanceCorrectedDifferentialAzimuthalCorrelationFunction.Clone()
        backgroundFunction = backgroundFunction.Clone()
        backgroundFunction.Scale(-1)
        acceptanceCorrectedBackgroundSubtractedDifferentialAzimuthalCorrelationFunction.Add(backgroundFunction)
        return acceptanceCorrectedBackgroundSubtractedDifferentialAzimuthalCorrelationFunction

    def getNormalizedDifferentialMixedEventCorrelationFunction(self, normMethod: NormalizationMethod, **kwargs):
        '''
        Returns the differential mixed event correlation function
        '''
        mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
        mixedEventCorrelationFunction.Scale(1 / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_PHI) / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_ETA))

        normalization_factor = self.computeMixedEventNormalizationFactor(mixedEventCorrelationFunction, normMethod, **kwargs)

        self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[self.currentRegion])
        self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[self.currentRegion])
        mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
        self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.INCLUSIVE])
        self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.INCLUSIVE])
        mixedEventCorrelationFunction.Scale(1 / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_PHI) / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_ETA))
        mixedEventCorrelationFunction.Scale(1 / normalization_factor)
        return mixedEventCorrelationFunction
    
    def getPIDFractions(self, makeIntermediatePlots=True):
        '''
        Prepares the projections for each enhanced species
        Converts them into arrays for fitting
        Fits and Extracts the optimal fit params
        Computes the PID fractions
        Returns the PID fractions
        '''
        # get the projections
        pionEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.PION)
        protonEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.PROTON)
        kaonEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.KAON)
        inclusiveEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.INCLUSIVE)
        # convert to arrays using FitTPCPionNsigma.prepareData 
        x, y, yerr = FitTPCPionNsigma.prepareData(pionEnhancedTPCnSigma, protonEnhancedTPCnSigma, kaonEnhancedTPCnSigma, inclusiveEnhancedTPCnSigma)
        # fit and extract the optimal fit params
        # start by creating the fitter instance
        fitter = FitTPCPionNsigma(self.analysisType, self.currentRegion, self.currentAssociatedHadronMomentumBin)
        # initialize the default parameters for the analysis type and current associated hadron momentum bin
        fitter.initializeDefaultParameters()
        optimal_params, covariance = fitter.performFit(x, y, yerr)

        chi2OverNDF = fitter.chi2OverNDF(optimal_params, covariance, x, y, yerr)

        if makeIntermediatePlots:
            self.plotTPCPionNsigmaFit(x, y, yerr, optimal_params, covariance, fitter.fittingFunction, fitter.fittingErrorFunction, fitter.pionFittingFunction, fitter.kaonFittingFunction, fitter.protonFittingFunction, fitter.chi2OverNDF, "TPCnSigmaFitPlots")

        if  not hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()
        # compute the PID fractions
        pid_fractions, pid_fraction_errors = fitter.computeAveragePIDFractions(optimal_params, covariance, self.numberOfAssociatedHadronsDictionary)

        return pid_fractions, pid_fraction_errors, chi2OverNDF

    def getEnhancedTPCnSigmaProjection(self, species: ParticleType):
        '''
        Sets the particle type for the jet hadron sparse and returns the projection onto the TPC nsigma axis, then resets the particle type
        '''
        self.setParticleSelectionForJetHadron(species)
        projection = self.getTPCPionNsigma()
        self.setParticleSelectionForJetHadron(self.current_species)
        return projection


    def getTPCPionNsigma(self):
        '''
        Returns the projection onto the TPC pion nsigma axis
        '''
        return self.JetHadron.getProjection(self.JetHadron.Axes.PION_TPC_N_SIGMA)
        

    def plotTPCPionNsigmaFit(self, x, y, yerr, optimal_params, covariance, fitFunction, fitErrorFunction, pionFitFunction, kaonFitFunction, protonFitFunction, chi2OverNDFFunction, save_path=None):

        if save_path is not None:
            if save_path[-1] != "/":
                save_path += "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        y_pion = y[0]
        y_proton = y[1]
        y_kaon = y[2]
        y_inclusive = y[3]

        y_err_pion = yerr[0]
        y_err_proton = yerr[1]
        y_err_kaon = yerr[2]
        y_err_inclusive = yerr[3]

        x_fit = np.linspace(x[0], x[-1], 100)
        y_fit = fitFunction(None, x_fit, *optimal_params)
        y_fit_err = fitErrorFunction(None, x_fit, *optimal_params, pcov=covariance)
        y_fit_pion = y_fit[:len(x_fit)]
        y_fit_proton = y_fit[len(x_fit):2*len(x_fit)]
        y_fit_kaon = y_fit[2*len(x_fit):3*len(x_fit)]
        y_fit_inclusive = y_fit[3*len(x_fit):]
        y_fit_err_pion = y_fit_err[:len(x_fit)]
        y_fit_err_proton = y_fit_err[len(x_fit):2*len(x_fit)]
        y_fit_err_kaon = y_fit_err[2*len(x_fit):3*len(x_fit)]
        y_fit_err_inclusive = y_fit_err[3*len(x_fit):]

        chi2OverNDF = chi2OverNDFFunction(optimal_params, covariance, x, y, yerr)

        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = optimal_params
        y_fit_pion_pion = pionFitFunction(x=x_fit, mu=mupi, sig=sigpi, a=apipi)
        y_fit_pion_proton = protonFitFunction(x_fit, mup, sigp, appi, alphap)
        y_fit_pion_kaon = kaonFitFunction(x_fit, muk, sigk, akpi, alphak)

        y_fit_proton_pion = pionFitFunction(x_fit, mupi, sigpi, apip)
        y_fit_proton_proton = protonFitFunction(x_fit, mup, sigp, app, alphap)
        y_fit_proton_kaon = kaonFitFunction(x_fit, muk, sigk, akp, alphak)

        y_fit_kaon_pion = pionFitFunction(x_fit, mupi, sigpi, apik)
        y_fit_kaon_proton = protonFitFunction(x_fit, mup, sigp, apk, alphap)
        y_fit_kaon_kaon = kaonFitFunction(x_fit, muk, sigk, akk, alphak)

        y_fit_inclusive_pion = pionFitFunction(x_fit, mupi, sigpi, apiinc)
        y_fit_inclusive_proton = protonFitFunction(x_fit, mup, sigp, apinc, alphap)
        y_fit_inclusive_kaon = kaonFitFunction(x_fit, muk, sigk, akinc, alphak)


        x_data_pion = {
            "raw_pion": x,
            "pion_total": x_fit,
            "pion_pion": x_fit,
            "pion_proton": x_fit,
            "pion_kaon": x_fit,
        }
        x_data_proton = {
            "raw_proton": x,
            "proton_total": x_fit,
            "proton_pion": x_fit,
            "proton_proton": x_fit,
            "proton_kaon": x_fit,
        }
        x_data_kaon = {
            "raw_kaon": x,
            "kaon_total": x_fit,
            "kaon_pion": x_fit,
            "kaon_proton": x_fit,
            "kaon_kaon": x_fit,
        }
        x_data_inclusive = {
            "raw_inclusive": x,
            "inclusive_total": x_fit,
            "inclusive_pion": x_fit,
            "inclusive_proton": x_fit,
            "inclusive_kaon": x_fit,
        }
        y_data_pion = {
            "raw_pion": y_pion,
            "pion_total": y_fit_pion,
            "pion_pion": y_fit_pion_pion,
            "pion_proton": y_fit_pion_proton,
            "pion_kaon": y_fit_pion_kaon,
        }
        y_data_proton = {
            "raw_proton": y_proton,
            "proton_total": y_fit_proton,
            "proton_pion": y_fit_proton_pion,
            "proton_proton": y_fit_proton_proton,
            "proton_kaon": y_fit_proton_kaon,
        }
        y_data_kaon = {
            "raw_kaon": y_kaon,
            "kaon_total": y_fit_kaon,
            "kaon_pion": y_fit_kaon_pion,
            "kaon_proton": y_fit_kaon_proton,
            "kaon_kaon": y_fit_kaon_kaon,
        }
        y_data_inclusive = {
            "raw_inclusive": y_inclusive,
            "inclusive_total": y_fit_inclusive,
            "inclusive_pion": y_fit_inclusive_pion,
            "inclusive_proton": y_fit_inclusive_proton,
            "inclusive_kaon": y_fit_inclusive_kaon,
        }

        yerr_data_pion = {
            "raw_pion": y_err_pion,
            "pion_total": y_fit_err_pion,
            "pion_pion": None,
            "pion_proton": None,
            "pion_kaon": None,
        }
        yerr_data_proton = {
            "raw_proton": y_err_proton,
            "proton_total": y_fit_err_proton,
            "proton_pion": None,
            "proton_proton": None,
            "proton_kaon": None,
        }
        yerr_data_kaon = {
            "raw_kaon": y_err_kaon,
            "kaon_total": y_fit_err_kaon,
            "kaon_pion": None,
            "kaon_proton": None,
            "kaon_kaon": None,
        }
        yerr_data_inclusive = {
            "raw_inclusive": y_err_inclusive,
            "inclusive_total": y_fit_err_inclusive,
            "inclusive_pion": None,
            "inclusive_proton": None,
            "inclusive_kaon": None,
        }
        
        data_labels_pion = {
            "raw_pion": "Enhanced Pion",
            "pion_total": "Pion Fit",
            "pion_pion": "Pion Component",
            "pion_proton": "Proton Component",
            "pion_kaon": "Kaon Component",
        }
        data_labels_proton = {
            "raw_proton": "Enhanced Proton",
            "proton_total": "Proton Fit",
            "proton_pion": "Pion Component",
            "proton_proton": "Proton Component",
            "proton_kaon": "Kaon Component",
        }
        data_labels_kaon = {
            "raw_kaon": "Enhanced Kaon",
            "kaon_total": "Kaon Fit",
            "kaon_pion": "Pion Component",
            "kaon_proton": "Proton Component",
            "kaon_kaon": "Kaon Component",
        }
        data_labels_inclusive = {
            "raw_inclusive": "Enhanced Inclusive",
            "inclusive_total": "Inclusive Fit",
            "inclusive_pion": "Pion Component",
            "inclusive_proton": "Proton Component",
            "inclusive_kaon": "Kaon Component",
        }

        format_style_pion = {
            "raw_pion": "o",
            "pion_total": "-",
            "pion_pion": "--",
            "pion_proton": "--",
            "pion_kaon": "--",
        }
        format_style_proton = {
            "raw_proton": "o",
            "proton_total": "-",
            "proton_pion": "--",
            "proton_proton": "--",
            "proton_kaon": "--",
        }
        format_style_kaon = {
            "raw_kaon": "o",
            "kaon_total": "-",
            "kaon_pion": "--",
            "kaon_proton": "--",
            "kaon_kaon": "--",
        }
        format_style_inclusive = {
            "raw_inclusive": "o",
            "inclusive_total": "-",
            "inclusive_pion": "--",
            "inclusive_proton": "--",
            "inclusive_kaon": "--",
        }

        plotArrays(x_data_pion, y_data_pion, yerr_data_pion, data_label=data_labels_pion, format_style=format_style_pion, error_bands=None, error_bands_label=None, title=f"TPC nSigma Fit - Pions Chi^2/NDF = {chi2OverNDF}", xtitle="TPC nSigma", ytitle="Counts", output_path=f"{save_path}TPCnSigmaFit{self.analysisType}_{self.currentRegion}_Pion.png")
        plotArrays(x_data_proton, y_data_proton, yerr_data_proton, data_label=data_labels_proton, format_style=format_style_proton, error_bands=None, error_bands_label=None, title=f"TPC nSigma Fit - Protons Chi^2/NDF = {chi2OverNDF}", xtitle="TPC nSigma", ytitle="Counts", output_path=f"{save_path}TPCnSigmaFit{self.analysisType}_{self.currentRegion}_Proton.png")
        plotArrays(x_data_kaon, y_data_kaon, yerr_data_kaon, data_label=data_labels_kaon, format_style=format_style_kaon, error_bands=None, error_bands_label=None, title=f"TPC nSigma Fit - Kaons Chi^2/NDF = {chi2OverNDF}", xtitle="TPC nSigma", ytitle="Counts", output_path=f"{save_path}TPCnSigmaFit{self.analysisType}_{self.currentRegion}_Kaon.png")
        plotArrays(x_data_inclusive, y_data_inclusive, yerr_data_inclusive, data_label=data_labels_inclusive, format_style=format_style_inclusive, error_bands=None, error_bands_label=None, title=f"TPC nSigma Fit - Inclusive Chi^2/NDF = {chi2OverNDF}", xtitle="TPC nSigma", ytitle="Counts", output_path=f"{save_path}TPCnSigmaFit{self.analysisType}_{self.currentRegion}_Inclusive.png")

        # y_fit_pi = pionFitFunction(None, x_fit, *optimal_params[:3])
        # y_fit_k = kaonFitFunction(None, x_fit, *optimal_params[3:6])
        # y_fit_p = protonFitFunction(None, x_fit, *optimal_params[6:9])

    def computeMixedEventNormalizationFactor(self, mixedEventCorrelationFunction, normMethod: NormalizationMethod, **kwargs):
        '''
        Returns the normalization factor for the mixed event correlation function
        '''
        if normMethod == NormalizationMethod.SLIDING_WINDOW:
            return self.computeSlidingWindowNormalizationFactor(mixedEventCorrelationFunction=mixedEventCorrelationFunction, **kwargs)
        elif normMethod == NormalizationMethod.MAX:
            return self.computeMaxNormalizationFactor(mixedEventCorrelationFunction=mixedEventCorrelationFunction)
        else:
            raise NotImplementedError("Normalization method not yet implemented")
        
    def computeSlidingWindowNormalizationFactor(self, mixedEventCorrelationFunction, windowSize: float = pi, deltaEtaRestriction: float = 0.3):
        '''
        Returns the normalization factor for the mixed event correlation function using the sliding window method after projection onto delta-phi for |delta-eta| < etaRestriction
        '''
        # make a clone to avoid modifying the original
        mixedEventCorrelationFunction = mixedEventCorrelationFunction.Clone()
        # restrict delta-eta range
        mixedEventCorrelationFunction.GetYaxis().SetRangeUser(-deltaEtaRestriction, deltaEtaRestriction)
        # calculate the number of delta-eta bins in the range 
        deltaEtaBinCount = mixedEventCorrelationFunction.GetYaxis().FindBin(deltaEtaRestriction) - mixedEventCorrelationFunction.GetYaxis().FindBin(-deltaEtaRestriction)
        # project onto delta-phi
        mixedEventAzimuthalCorrelationFunction = mixedEventCorrelationFunction.ProjectionX()
        # divide by the number of delta-eta bins to get the average
        mixedEventAzimuthalCorrelationFunction.Scale(1 / deltaEtaBinCount)
        # calculate the number of delta-phi bins in the window
        windowBinCount = int(windowSize // self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_PHI))
        maxWindowAverage = 0
        # slide the window across the mixed event correlation function
        assert windowBinCount < mixedEventAzimuthalCorrelationFunction.GetNbinsX(), f"Window size is too large, must be less than {mixedEventAzimuthalCorrelationFunction.GetNbinsX()}, got {windowBinCount}"
        for i in range(mixedEventAzimuthalCorrelationFunction.GetNbinsX() - windowBinCount):
            # calculate the average in the window
            windowAverage = mixedEventAzimuthalCorrelationFunction.Integral(i, i + windowBinCount) / windowBinCount
            # keep track of the maximum average
            if windowAverage > maxWindowAverage:
                maxWindowAverage = windowAverage
        return maxWindowAverage
    
    def computeMaxNormalizationFactor(self, mixedEventCorrelationFunction, deltaEtaRestriction: float = 0.3):
        '''
        Returns the normalization factor for the mixed event correlation function using the maximum value within the restricted delta-eta range
        '''
        # make a clone to avoid modifying the original
        mixedEventCorrelationFunction = mixedEventCorrelationFunction.Clone()
        # restrict delta-eta range
        mixedEventCorrelationFunction.GetYaxis().SetRangeUser(-deltaEtaRestriction, deltaEtaRestriction)
        # calculate the number of delta-eta bins in the range
        deltaEtaBinCount = mixedEventCorrelationFunction.GetYaxis().FindBin(deltaEtaRestriction) - mixedEventCorrelationFunction.GetYaxis().FindBin(-deltaEtaRestriction)
        # project onto delta-phi
        mixedEventAzimuthalCorrelationFunction = mixedEventCorrelationFunction.ProjectionX()
        # divide by the number of delta-eta bins to get the average
        mixedEventAzimuthalCorrelationFunction.Scale(1 / deltaEtaBinCount)
        # return the maximum value
        return mixedEventAzimuthalCorrelationFunction.GetMaximum()

    def getBackgroundCorrelationFunction(self, per_trigger_normalized=False):
        '''
        Returns the background correlation function
        '''
        # get the positive eta and negative eta background regions of the Jet Hadron correlation function and add them together
        self.JetHadron.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.BACKGROUND_ETANEG])
        self.JetHadron.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.BACKGROUND_ETANEG])

        backgroundCorrelationFunction_etaneg = self.JetHadron.getProjection(self.JetHadron.Axes.DELTA_PHI, self.JetHadron.Axes.DELTA_ETA)

        nbins_delta_eta_neg = backgroundCorrelationFunction_etaneg.GetYaxis().GetNbins()
        backgroundCorrelationFunction_etaneg = backgroundCorrelationFunction_etaneg.ProjectionX()
        backgroundCorrelationFunction_etaneg.Scale(1 / nbins_delta_eta_neg)

        self.JetHadron.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.BACKGROUND_ETAPOS])
        self.JetHadron.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.BACKGROUND_ETAPOS])

        backgroundCorrelationFunction_etapos = self.JetHadron.getProjection(self.JetHadron.Axes.DELTA_PHI, self.JetHadron.Axes.DELTA_ETA)

        nbins_delta_eta_pos = backgroundCorrelationFunction_etapos.GetYaxis().GetNbins()
        backgroundCorrelationFunction_etapos = backgroundCorrelationFunction_etapos.ProjectionX()
        backgroundCorrelationFunction_etapos.Scale(1 / nbins_delta_eta_pos)

        backgroundCorrelationFunction = backgroundCorrelationFunction_etaneg.Clone()
        backgroundCorrelationFunction.Add(backgroundCorrelationFunction_etapos)
        # divide by the delta-phi bin width and the delta-eta bin width to get the average
        backgroundCorrelationFunction.Scale(1 / self.JetHadron.getBinWidth(self.JetHadron.Axes.DELTA_PHI))
        if per_trigger_normalized:
            backgroundCorrelationFunction.Scale(1 / self.getNumberOfTriggerJets())
        # set the delta-phi and delta-eta ranges back to the previous values
        self.setRegion(self.currentRegion)
        return backgroundCorrelationFunction
    
    def getAzimuthalBackgroundFunction(self, backgroundCorrelationFunction: TH1D):
        '''
        Returns the background function. In proton-proton collisions this is a pedestal function estimated as the average value in the background region.
        '''
        # pass backgroundCorrelationFunction to the BackgroundFunction constructor
        backgroundFunction = BackgroundFunction(backgroundCorrelationFunction, self.analysisType)
        # make a TH1D of the background function
        backgroundFunctionHistogram = TH1D("backgroundFunctionHistogram", "backgroundFunctionHistogram", regionDeltaPhiBinCountsDictionary[self.currentRegion], regionDeltaPhiRangeDictionary[self.currentRegion][0], regionDeltaPhiRangeDictionary[self.currentRegion][1])
        # fill the histogram with the background function values
        for deltaPhiBin in range(1, backgroundFunctionHistogram.GetNbinsX() + 1):
            backgroundFunctionHistogram.SetBinContent(
                deltaPhiBin,
                backgroundFunction(backgroundFunctionHistogram.GetBinCenter(deltaPhiBin))
                )
            backgroundFunctionHistogram.SetBinError(
                deltaPhiBin, 
                backgroundFunction.error(backgroundFunctionHistogram.GetBinCenter(deltaPhiBin))
                )

        return backgroundFunctionHistogram

    def fillNumberOfAssociatedHadronsDictionary(self):
        '''
        Returns a dictionary of the number of associated hadrons in current associated hadron momentum bin
        '''
        self.numberOfAssociatedHadronsDictionary = {}
        for species in ParticleType:
            self.setParticleSelectionForJetHadron(species)
            self.numberOfAssociatedHadronsDictionary[species] = self.getNumberOfAssociatedParticles()
        self.setParticleSelectionForJetHadron(self.current_species)

    def getNumberOfAssociatedParticles(self):
        '''
        Returns the number of associated particles
        '''
        return self.JetHadron.getNumberOfAssociatedParticles()

    def getNumberOfTriggerJets(self):
        '''
        Returns the number of trigger jets
        '''
        return self.Trigger.getNumberOfTriggerJets()



