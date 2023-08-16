# © Patrick John Steffanic 2023
# This file contains the class whose responsibility it is to manage the analysis and its configuration 

import numpy as np
import uncertainties
from JetHadronAnalysis.Sparse import TriggerSparse, MixedEventSparse, JetHadronSparse
from JetHadronAnalysis.Types import AnalysisType, ParticleType, NormalizationMethod
from JetHadronAnalysis.Background import BackgroundFunction
from JetHadronAnalysis.TPCPionNsigmaFit import FitTPCPionNsigma
from ROOT import TFile, TH1D # type: ignore
from enum import Enum
from math import pi

class Region(Enum):
    NEAR_SIDE_SIGNAL = 1
    AWAY_SIDE_SIGNAL = 2
    BACKGROUND_ETAPOS = 3
    BACKGROUND_ETANEG = 4
    INCLUSIVE = 5

class AssociatedHadronMomentumBin(Enum):
    PT_1_15 = 1
    PT_15_2 = 2
    PT_2_3 = 3
    PT_3_4 = 4
    PT_4_5 = 5
    PT_5_6 = 6
    PT_6_10 = 7
    
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

    def setAssociatedHadronMomentumBin(self, associatedHadronMomentumBin: AssociatedHadronMomentumBin):
        '''
        Sets the associated hadron momentum bin for the JetHadron sparse
        '''
        self.currentAssociatedHadronMomentumBin = associatedHadronMomentumBin
        self.JetHadron.setAssociatedHadronMomentumRange(*associatedHadronMomentumBinRangeDictionary[associatedHadronMomentumBin])

    def setParticleSelectionForJetHadron(self, species: ParticleType):
        '''
        Sets the pion, kaon, and proton TOF ranges to get the particle type specified in JetHadron sparse
        TODO: Unclear how to cleanly select ParticleType.OTHER
        '''
        if species == ParticleType.OTHER:
            raise NotImplementedError("ParticleType.OTHER not yet implemented")
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
    
    def getPIDFractions(self):
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
        fitter = FitTPCPionNsigma()
        # initialize the default parameters for the analysis type and current associated hadron momentum bin
        fitter.initializeDefaultParameters(self.analysisType, self.currentAssociatedHadronMomentumBin)
        optimal_params, covariance = fitter.performFit(x, y, yerr)
        if  not hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()
        # compute the PID fractions
        pid_fractions, pid_fraction_errors = fitter.computeAveragePIDFractions(optimal_params, covariance, self.numberOfAssociatedHadronsDictionary)

        return pid_fractions, pid_fraction_errors

    def getEnhancedTPCnSigmaProjection(self, species: ParticleType):
        '''
        Sets the particle type for the jet hadron sparse and returns the projection onto the TPC nsigma axis, then resets the particle type
        '''
        self.setParticleSelectionForJetHadron(species)
        projection = self.getTPCPionNsigma()
        self.setParticleSelectionForJetHadron(ParticleType.INCLUSIVE)
        return projection


    def getTPCPionNsigma(self):
        '''
        Returns the projection onto the TPC pion nsigma axis
        '''
        return self.JetHadron.getProjection(self.JetHadron.Axes.PION_TPC_N_SIGMA)
        

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
            if species==ParticleType.OTHER:
                continue
            self.setParticleSelectionForJetHadron(species)
            self.numberOfAssociatedHadronsDictionary[species] = self.getNumberOfAssociatedParticles()
        self.setParticleSelectionForJetHadron(ParticleType.INCLUSIVE)

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



