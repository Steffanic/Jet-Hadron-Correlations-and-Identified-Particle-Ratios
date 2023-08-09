# © Patrick John Steffanic 2023
# This file contains the class whose responsibility it is to manage the analysis and its configuration 

from JetHadronAnalysis.Sparse import TriggerSparse, MixedEventSparse, JetHadronSparse
from JetHadronAnalysis.Types import AnalysisType, ParticleType, NormalizationMethod
from ROOT import TFile
from enum import Enum
from math import pi

class Region(Enum):
    NEAR_SIDE_SIGNAL = 1
    AWAY_SIDE_SIGNAL = 2
    BACKGROUND = 3
    INCLUSIVE = 4
    
regionDeltaPhiRangeDictionary = {
    Region.NEAR_SIDE_SIGNAL: [-pi / 2, pi / 2],
    Region.AWAY_SIDE_SIGNAL: [pi / 2, 3 * pi / 2],
    Region.BACKGROUND: [-pi / 2, pi / 2],
    Region.INCLUSIVE: [-pi / 2, 3 * pi / 2]
}

regionDeltaEtaRangeDictionary = {
    Region.NEAR_SIDE_SIGNAL: [-0.6, 0.6],
    Region.AWAY_SIDE_SIGNAL: [-1.2, 1.2],
    Region.BACKGROUND: ([-1.2, -0.8], [0.8, 1.2]),
    Region.INCLUSIVE: [-1.4, 1.4]
}

speciesTOFRangeDictionary = {
    ParticleType.PION: ([-2,2], [-5,-2], [-5,-2]),
    ParticleType.KAON: ([-5,5], [-2,2], [-5,-2]),
    ParticleType.PROTON: ([-5,5], [-5,5], [-2,2]),
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

    def setRegionForSparses(self, region: Region):
        '''
        Sets the delta-phi and delta-eta ranges for the JetHadron sparse and the Mixed Event sparse
        '''
        if region == Region.BACKGROUND:
            # TODO: Unclear how to handle background region
            raise NotImplementedError("Background region not yet implemented")
        self.JetHadron.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[region])
        self.JetHadron.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[region])
        self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[region])
        self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[region])


    def setParticleSelectionForSparses(self, species: ParticleType):
        '''
        Sets the pion, kaon, and proton TOF ranges to get the particle type specified in JetHadron sparse
        TODO: Unclear how to cleanly select ParticleType.OTHER
        '''
        if species == ParticleType.OTHER:
            raise NotImplementedError("ParticleType.OTHER not yet implemented")
        self.JetHadron.setPionTOFnSigma(*speciesTOFRangeDictionary[species][0])
        self.JetHadron.setKaonTOFnSigma(*speciesTOFRangeDictionary[species][1])
        self.JetHadron.setProtonTOFnSigma(*speciesTOFRangeDictionary[species][2])

    def getDifferentialCorrelationFunction(self):
        '''
        Returns the differential correlation function
        '''
        correlationFunction = self.JetHadron.getProjection(self.JetHadron.Axes.DELTA_PHI, self.JetHadron.Axes.DELTA_ETA)
        correlationFunction.Scale(1 / self.JetHadron.getBinWidth(self.JetHadron.Axes.DELTA_PHI) / self.JetHadron.getBinWidth(self.JetHadron.Axes.DELTA_ETA))
        return correlationFunction
    
    def getNormalizedDifferentialMixedEventCorrelationFunction(self, normMethod: NormalizationMethod, **kwargs):
        '''
        Returns the differential mixed event correlation function
        '''
        mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
        mixedEventCorrelationFunction.Scale(1 / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_PHI) / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_ETA))
        mixedEventCorrelationFunction.Scale(1 / self.computeMixedEventNormalizationFactor(mixedEventCorrelationFunction, normMethod, **kwargs))
        return mixedEventCorrelationFunction
    
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



