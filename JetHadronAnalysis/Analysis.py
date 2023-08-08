# © Patrick John Steffanic 2023
# This file contains the class whose responsibility it is to manage the analysis and its configuration 

from JetHadronAnalysis.Sparse import TriggerSparse, MixedEventSparse, JetHadronSparse
from JetHadronAnalysis.Types import AnalysisType
from ROOT import TFile
from enum import Enum

class Region(Enum):
    NEAR_SIDE_SIGNAL = 1
    AWAY_SIDE_SIGNAL = 2
    BACKGROUND = 3
    


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
        rootFileListName = f"AliAnalysisTaskJetH_tracks_caloClusters_biased" if self.analysisType == AnalysisType.PP else f"AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2_{'SemiCentral' if self.analysisType == AnalysisType.SEMICENTRAL else 'Central'}q"
        rootFileList = file.Get(rootFileListName)
        self.JetHadron.addSparse(rootFileList.FindObject("fhnJH"))
        self.Trigger.addSparse(rootFileList.FindObject("fhnTrigger"))
        self.MixedEvent.addSparse(rootFileList.FindObject("fhnMixedEvents"))
        file.Close()

    def setRegionForSparses(self, region: Region):
        '''
        Sets the delta-phi and delta-eta ranges for the JetHadron sparse and the Mixed Event sparse
        '''
        pass

    def setParticleSelectionForSparses(self, species: ParticleType)