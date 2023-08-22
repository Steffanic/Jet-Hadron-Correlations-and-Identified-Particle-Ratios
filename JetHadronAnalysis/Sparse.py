# © Patrick John Steffanic 2023
# This file contains the class whose responsibility it is to manage the state of the sparses.
# It also contains the enums for the axes in each different sparse

import warnings
from JetHadronAnalysis.Types import AnalysisType, regionDeltaEtaRangeDictionary, regionDeltaPhiRangeDictionary, Region, OtherTOFRangeDictionary, OtherTOFRangeTags
from math import pi
from enum import Enum

def triggerSparseAxesEnumFactory(analysisType: AnalysisType): 
    class TriggerAxes(Enum):
        if analysisType != AnalysisType.PP:
            EVENT_PLANE_ANGLE = 2
        EVENT_ACTIVITY = 0
        TRIGGER_JET_PT = 1
        Z_VERTEX = 2  if analysisType == AnalysisType.PP else 3

    return TriggerAxes

def mixedEventSparseAxesEnumFactory(analysisType: AnalysisType):
    class MixedEventAxes(Enum):
        if analysisType != AnalysisType.PP:
            EVENT_PLANE_ANGLE = 5
        EVENT_ACTIVITY = 0
        TRIGGER_JET_PT = 1
        ASSOCIATED_HADRON_PT = 2
        DELTA_ETA = 3
        DELTA_PHI = 4
        Z_VERTEX = 5  if analysisType == AnalysisType.PP else 6

    return MixedEventAxes

def jetHadronSparseAxesEnumFactory(analysisType: AnalysisType):
    class JetHadronAxes(Enum):
        if analysisType != AnalysisType.PP:
            EVENT_PLANE_ANGLE = 5
        EVENT_ACTIVITY = 0
        TRIGGER_JET_PT = 1
        ASSOCIATED_HADRON_PT = 2
        DELTA_ETA = 3
        DELTA_PHI = 4
        Z_VERTEX = 5  if analysisType == AnalysisType.PP else 6
        ASSOCIATED_HADRON_ETA = 6  if analysisType == AnalysisType.PP else 7
        PION_TPC_N_SIGMA = 7  if analysisType == AnalysisType.PP else 8
        PION_TOF_N_SIGMA = 8  if analysisType == AnalysisType.PP else 9
        PROTON_TOF_N_SIGMA = 9  if analysisType == AnalysisType.PP else 10
        KAON_TOF_N_SIGMA = 10  if analysisType == AnalysisType.PP else 11

    return JetHadronAxes


class Sparse:
    '''
    This class represents a sparse in my analysis, and is responsible for managing the state of the sparse.
    '''

    def __init__(self, analysisType: AnalysisType):
        self.sparseList = []
        self.analysisType = analysisType
        self.Axes = None

    def getNumberOfSparses(self):
        return len(self.sparseList)

    def addSparse(self, sparse):
        self.sparseList.append(sparse)

    def getSparseList(self):
        return self.sparseList
    
    def getProjection(self, *projectionAxes):
        '''
        return a projection along the projection axes provided by combining the projections from each sparse, axis ordering should be consistent, e.g. x,y,z regaredless of number of axes
        '''
        assert self.getNumberOfSparses() != 0, "Trying to get projection without any sparses in the sparse list"
        if len(projectionAxes)==0:
            raise ValueError("Please specify at least one axis to project along")
        if  len(projectionAxes)>3:
            raise NotImplementedError("The THnSparse Projection method allows for up to three projection axes, I haven't implemented the other method yet")
        if type(projectionAxes[0]) != int:
            projectionAxes = [axis.value for axis in projectionAxes]
        if len(projectionAxes) == 2:
            # For some reason the implementation of ROOT's THnSparse::Projection method for two axes is reversed order, (y_axis, x_axis) instead of (x_axis, y_axis), which is pretty dumb.
            projectionAxes = projectionAxes[::-1]
        # When we clone the projection, we pass the representation of the sparse as the name of the projection, so that it is uniquely identified in ROOT's memory. It's super hacky and dumb that I have to do that...
        projection = self.sparseList[0].Projection(*projectionAxes).Clone(repr(self)+"_".join([str(axis) for axis in projectionAxes]))
        for sparse_ind in range(self.getNumberOfSparses()-1):
            addition_success = projection.Add(self.sparseList[sparse_ind].Projection(*projectionAxes).Clone(repr(self)+"_".join([str(axis) for axis in projectionAxes])))
            if not addition_success:
                raise RuntimeError(f"Adding projection onto axes {[axis.name for axis in projectionAxes]} from sparse {sparse_ind} in {self.__class__.__name__} was not succesful")
            
        return projection
    
    def getBinWidth(self, axis):
        return self.sparseList[0].GetAxis(axis.value).GetBinWidth(1) # here we get the firtst bin because in ROOT, bins are 1-ordered, not 0-ordered, which is pretty dumb
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.analysisType.name})"



class TriggerSparse(Sparse):
    '''
    This class represents a trigger sparse in my analysis, and is responsible for managing the state of the sparse.
    '''

    def __init__(self, analysisType: AnalysisType):
        super().__init__(analysisType)
        self.Axes = triggerSparseAxesEnumFactory(self.analysisType)

        self.minTriggerJetMomentum = 0
        self.maxTriggerJetMomentum = 200

        self.minEventActivity = 0
        self.maxEventActivity = 100 if analysisType != AnalysisType.PP else 200

        self.minEventPlaneAngle = 0 if analysisType != AnalysisType.PP else None
        self.maxEventPlaneAngle = pi / 2 if analysisType != AnalysisType.PP else None

        self.minZVertex = -10
        self.maxZVertex = 10

    def setTriggerJetMomentumRange(self, minTriggerJetMomentum, maxTriggerJetMomentum):
        self.minTriggerJetMomentum = minTriggerJetMomentum
        self.maxTriggerJetMomentum = maxTriggerJetMomentum
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.TRIGGER_JET_PT.value).SetRangeUser(minTriggerJetMomentum, maxTriggerJetMomentum)

    def setEventActivityRange(self, minEventActivity, maxEventActivity):
        self.minEventActivity = minEventActivity
        self.maxEventActivity = maxEventActivity
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.EVENT_ACTIVITY.value).SetRangeUser(minEventActivity, maxEventActivity)

    def setEventPlaneAngleRange(self, minEventPlaneAngle, maxEventPlaneAngle):
        assert self.analysisType != AnalysisType.PP, "Event plane angle is not a valid axis for pp analysis"
        self.minEventPlaneAngle = minEventPlaneAngle
        self.maxEventPlaneAngle = maxEventPlaneAngle
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.EVENT_PLANE_ANGLE.value).SetRangeUser(minEventPlaneAngle, maxEventPlaneAngle)

    def setZVertexRange(self, minZVertex, maxZVertex):
        self.minZVertex = minZVertex
        self.maxZVertex = maxZVertex
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.Z_VERTEX.value).SetRangeUser(minZVertex, maxZVertex)

    def getNumberOfTriggerJets(self):
        '''
        Gets number of trigger jets by projecting onto an axis and getting the number of entries
        '''
        projection = self.getProjection(self.Axes.TRIGGER_JET_PT)
        return projection.GetEntries()

    def __repr__(self) -> str:
        repr_str = "Trigger Sparse:\n"
        repr_str += "Trigger Jet Momentum Range: " + str(self.minTriggerJetMomentum) + " - " + str(self.maxTriggerJetMomentum) + "\n"
        repr_str += "Event Activity Range: " + str(self.minEventActivity) + " - " + str(self.maxEventActivity) + "\n"
        repr_str += "Z Vertex Range: " + str(self.minZVertex) + " - " + str(self.maxZVertex) + "\n"
        if self.analysisType != AnalysisType.PP:
            repr_str += "Event Plane Angle Range: " + str(self.minEventPlaneAngle) + " - " + str(self.maxEventPlaneAngle) + "\n"
        return repr_str

class MixedEventSparse(Sparse):
    '''
    This class represents a mixed event sparse in my analysis, and is responsible for managing the state of the sparse.
    '''

    def __init__(self, analysisType: AnalysisType):
        super().__init__(analysisType)
        self.Axes = mixedEventSparseAxesEnumFactory(self.analysisType)

        self.minTriggerJetMomentum = 0
        self.maxTriggerJetMomentum = 200

        self.minAssociatedHadronMomentum = 0
        self.maxAssociatedHadronMomentum = 10

        self.minEventActivity = 0
        self.maxEventActivity = 100 if analysisType != AnalysisType.PP else 200

        self.minEventPlaneAngle = 0 if analysisType != AnalysisType.PP else None
        self.maxEventPlaneAngle = pi / 2 if analysisType != AnalysisType.PP else None

        self.minZVertex = -10
        self.maxZVertex = 10

        self.minDeltaPhi = -pi / 2
        self.maxDeltaPhi = 3 * pi / 2

        self.minDeltaEta = -1.4
        self.maxDeltaEta = 1.4

    def setTriggerJetMomentumRange(self, minTriggerJetMomentum, maxTriggerJetMomentum):
        self.minTriggerJetMomentum = minTriggerJetMomentum
        self.maxTriggerJetMomentum = maxTriggerJetMomentum
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.TRIGGER_JET_PT.value).SetRangeUser(minTriggerJetMomentum, maxTriggerJetMomentum)

    def setAssociatedHadronMomentumRange(self, minAssociatedHadronMomentum, maxAssociatedHadronMomentum):
        self.minAssociatedHadronMomentum = minAssociatedHadronMomentum
        self.maxAssociatedHadronMomentum = maxAssociatedHadronMomentum
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.ASSOCIATED_HADRON_PT.value).SetRangeUser(minAssociatedHadronMomentum, maxAssociatedHadronMomentum)

    def setEventActivityRange(self, minEventActivity, maxEventActivity):
        self.minEventActivity = minEventActivity
        self.maxEventActivity = maxEventActivity
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.EVENT_ACTIVITY.value).SetRangeUser(minEventActivity, maxEventActivity)

    def setEventPlaneAngleRange(self, minEventPlaneAngle, maxEventPlaneAngle):
        assert self.analysisType != AnalysisType.PP, "Event plane angle is not a valid axis for pp analysis"
        self.minEventPlaneAngle = minEventPlaneAngle
        self.maxEventPlaneAngle = maxEventPlaneAngle
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.EVENT_PLANE_ANGLE.value).SetRangeUser(minEventPlaneAngle, maxEventPlaneAngle)

    def setZVertexRange(self, minZVertex, maxZVertex):
        self.minZVertex = minZVertex
        self.maxZVertex = maxZVertex
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.Z_VERTEX.value).SetRangeUser(minZVertex, maxZVertex)

    def setDeltaPhiRange(self, minDeltaPhi, maxDeltaPhi):
        self.minDeltaPhi = minDeltaPhi
        self.maxDeltaPhi = maxDeltaPhi
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.DELTA_PHI.value).SetRangeUser(minDeltaPhi, maxDeltaPhi)

    def setDeltaEtaRange(self, minDeltaEta, maxDeltaEta):
        self.minDeltaEta = minDeltaEta
        self.maxDeltaEta = maxDeltaEta
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.DELTA_ETA.value).SetRangeUser(minDeltaEta, maxDeltaEta)

    def __repr__(self) -> str:
        repr_str = "Mixed Event Sparse:\n"
        repr_str += "Trigger Jet Momentum Range: " + str(self.minTriggerJetMomentum) + " - " + str(self.maxTriggerJetMomentum) + "\n"
        repr_str += "Associated Hadron Momentum Range: " + str(self.minAssociatedHadronMomentum) + " - " + str(self.maxAssociatedHadronMomentum) + "\n"
        repr_str += "Event Activity Range: " + str(self.minEventActivity) + " - " + str(self.maxEventActivity) + "\n"
        repr_str += "Z Vertex Range: " + str(self.minZVertex) + " - " + str(self.maxZVertex) + "\n"
        repr_str += "Delta Phi Range: " + str(self.minDeltaPhi) + " - " + str(self.maxDeltaPhi) + "\n"
        repr_str += "Delta Eta Range: " + str(self.minDeltaEta) + " - " + str(self.maxDeltaEta) + "\n"
        if self.analysisType != AnalysisType.PP:
            repr_str += "Event Plane Angle Range: " + str(self.minEventPlaneAngle) + " - " + str(self.maxEventPlaneAngle) + "\n"
        return repr_str

class JetHadronSparse(Sparse):
    '''
    This class represents a jet hadron sparse in my analysis, and is responsible for managing the state of the sparse.
    '''

    def __init__(self, analysisType: AnalysisType):
        super().__init__(analysisType)
        self.Axes = jetHadronSparseAxesEnumFactory(self.analysisType)

        self.minTriggerJetMomentum = 0
        self.maxTriggerJetMomentum = 200

        self.minAssociatedHadronMomentum = 0
        self.maxAssociatedHadronMomentum = 10

        self.minEventActivity = 0
        self.maxEventActivity = 100 if analysisType != AnalysisType.PP else 200

        self.minEventPlaneAngle = 0 if analysisType != AnalysisType.PP else None
        self.maxEventPlaneAngle = pi / 2 if analysisType != AnalysisType.PP else None

        self.minZVertex = -10
        self.maxZVertex = 10

        self.minDeltaPhi = -pi / 2
        self.maxDeltaPhi = 3 * pi / 2

        self.minDeltaEta = -1.4
        self.maxDeltaEta = 1.4

        self.minPionTPCnSigma = -10
        self.maxPionTPCnSigma = 10

        self.minPionTOFnSigma = -5
        self.maxPionTOFnSigma = 5

        self.minProtonTOFnSigma = -5
        self.maxProtonTOFnSigma = 5

        self.minKaonTOFnSigma = -5
        self.maxKaonTOFnSigma = 5

        self.particleTypeIsOther=False
        self.regionIsBackground=False


    def getProjection(self, *projectionAxes):
        '''
        return a projection along the projection axes provided by combining the projections from each sparse, axis ordering should be consistent, e.g. x,y,z regaredless of number of axes
        Reimplementation for JetHadronSparse to account for special handling of ParticleType.OTHER case
        '''
        projections = []
        if self.regionIsBackground:
            # handle background case
            # get the projection for Region.BACKGROUND_ETANEG, Region.BACKGROUND_ETAPOS
            for region in [Region.BACKGROUND_ETANEG, Region.BACKGROUND_ETAPOS]:
                self.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[region])
                self.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[region])
                if self.particleTypeIsOther:
                    projections += self.getProjectionsForOtherTOFRange(*projectionAxes)
                else:
                    projections.append(super().getProjection(*projectionAxes))
                # add the projections together
            # reset the delta eta and delta phi ranges
            warnings.warn(f"Setting the delta eta and delta phi ranges back to the previous values:\n\tDelta Eta: ({self.minDeltaEta}, {self.maxDeltaEta})\n\tDelta Phi: ({self.minDeltaPhi}, {self.maxDeltaPhi})")
            self.setDeltaEtaRange(self.minDeltaEta, self.maxDeltaEta)
            self.setDeltaPhiRange(self.minDeltaPhi, self.maxDeltaPhi)
            projection = projections[0]
            for proj in projections[1:]:
                projection.Add(proj)
            return projection
        else:
            # handle non-background case
            if self.particleTypeIsOther:
                projections =  self.getProjectionsForOtherTOFRange(*projectionAxes)
                projection = projections[0]
                for proj in projections[1:]:
                    projection.Add(proj)
                return projection
            
        return super().getProjection(*projectionAxes)
        

    def getProjectionsForOtherTOFRange(self, *projectionAxes):
        projections = []
        for other_tof_range_tag in OtherTOFRangeDictionary.keys():
            self.setPionTOFnSigma(*OtherTOFRangeDictionary[other_tof_range_tag][0])
            self.setKaonTOFnSigma(*OtherTOFRangeDictionary[other_tof_range_tag][1])
            self.setProtonTOFnSigma(*OtherTOFRangeDictionary[other_tof_range_tag][2])
            projections.append(super().getProjection(*projectionAxes))
        warnings.warn(f"Setting the TOF Nsigma ranges back to the previous values:\n\tPion: ({self.minPionTOFnSigma}, {self.maxPionTOFnSigma})\n\tKaon: ({self.minKaonTOFnSigma}, {self.maxKaonTOFnSigma})\n\tProton: ({self.minProtonTOFnSigma}, {self.maxProtonTOFnSigma})")
        # reset the TOF ranges
        self.setPionTOFnSigma(self.minPionTOFnSigma, self.maxPionTOFnSigma)
        self.setKaonTOFnSigma(self.minKaonTOFnSigma, self.maxKaonTOFnSigma)
        self.setProtonTOFnSigma(self.minProtonTOFnSigma, self.maxProtonTOFnSigma)
        return projections

    def setTriggerJetMomentumRange(self, minTriggerJetMomentum, maxTriggerJetMomentum):
        self.minTriggerJetMomentum = minTriggerJetMomentum
        self.maxTriggerJetMomentum = maxTriggerJetMomentum
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.TRIGGER_JET_PT.value).SetRangeUser(minTriggerJetMomentum, maxTriggerJetMomentum)

    def setAssociatedHadronMomentumRange(self, minAssociatedHadronMomentum, maxAssociatedHadronMomentum):
        self.minAssociatedHadronMomentum = minAssociatedHadronMomentum
        self.maxAssociatedHadronMomentum = maxAssociatedHadronMomentum
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.ASSOCIATED_HADRON_PT.value).SetRangeUser(minAssociatedHadronMomentum, maxAssociatedHadronMomentum)

    def setEventActivityRange(self, minEventActivity, maxEventActivity):
        self.minEventActivity = minEventActivity
        self.maxEventActivity = maxEventActivity
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.EVENT_ACTIVITY.value).SetRangeUser(minEventActivity, maxEventActivity)

    def setEventPlaneAngleRange(self, minEventPlaneAngle, maxEventPlaneAngle):
        assert self.analysisType != AnalysisType.PP, "Event plane angle is not a valid axis for pp analysis"
        self.minEventPlaneAngle = minEventPlaneAngle
        self.maxEventPlaneAngle = maxEventPlaneAngle
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.EVENT_PLANE_ANGLE.value).SetRangeUser(minEventPlaneAngle, maxEventPlaneAngle)

    def setZVertexRange(self, minZVertex, maxZVertex):
        self.minZVertex = minZVertex
        self.maxZVertex = maxZVertex
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.Z_VERTEX.value).SetRangeUser(minZVertex, maxZVertex)

    def setDeltaPhiRange(self, minDeltaPhi, maxDeltaPhi):
        self.minDeltaPhi = minDeltaPhi
        self.maxDeltaPhi = maxDeltaPhi
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.DELTA_PHI.value).SetRangeUser(minDeltaPhi, maxDeltaPhi)

    def setDeltaEtaRange(self, minDeltaEta, maxDeltaEta):
        self.minDeltaEta = minDeltaEta
        self.maxDeltaEta = maxDeltaEta
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.DELTA_ETA.value).SetRangeUser(minDeltaEta, maxDeltaEta)

    def setAssociatedHadronEta(self, minAssociatedHadronEta, maxAssociatedHadronEta):
        self.minAssociatedHadronEta = minAssociatedHadronEta
        self.maxAssociatedHadronEta = maxAssociatedHadronEta
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.ASSOCIATED_HADRON_ETA.value).SetRangeUser(minAssociatedHadronEta, maxAssociatedHadronEta)

    def setPionTPCnSigma(self, minPionTPCnSigma, maxPionTPCnSigma):
        self.minPionTPCnSigma = minPionTPCnSigma
        self.maxPionTPCnSigma = maxPionTPCnSigma
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.PION_TPC_N_SIGMA.value).SetRangeUser(minPionTPCnSigma, maxPionTPCnSigma)

    def setPionTOFnSigma(self, minPionTOFnSigma, maxPionTOFnSigma):
        self.minPionTOFnSigma = minPionTOFnSigma
        self.maxPionTOFnSigma = maxPionTOFnSigma
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.PION_TOF_N_SIGMA.value).SetRangeUser(minPionTOFnSigma, maxPionTOFnSigma)

    def setProtonTOFnSigma(self, minProtonTOFnSigma, maxProtonTOFnSigma):
        self.minProtonTOFnSigma = minProtonTOFnSigma
        self.maxProtonTOFnSigma = maxProtonTOFnSigma
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.PROTON_TOF_N_SIGMA.value).SetRangeUser(minProtonTOFnSigma, maxProtonTOFnSigma)

    def setKaonTOFnSigma(self, minKaonTOFnSigma, maxKaonTOFnSigma):
        self.minKaonTOFnSigma = minKaonTOFnSigma
        self.maxKaonTOFnSigma = maxKaonTOFnSigma
        for sparse_ind in range(self.getNumberOfSparses()):
            # set the pT and event plane angle ranges
            self.sparseList[sparse_ind].GetAxis(self.Axes.KAON_TOF_N_SIGMA.value).SetRangeUser(minKaonTOFnSigma, maxKaonTOFnSigma)

    def setParticleTypeIsOther(self, particleTypeIsOther:bool):
        self.particleTypeIsOther = particleTypeIsOther
    
    def setRegionIsBackground(self, regionIsBackground:bool):
        self.regionIsBackground = regionIsBackground

    def getNumberOfAssociatedParticles(self):
        '''
        Gets number of associated particles by projecting onto an axis and getting the number of entries
        '''
        projection = self.getProjection(self.Axes.TRIGGER_JET_PT)
        return projection.GetEntries()

    def __repr__(self) -> str:
        repr_str = "Mixed Event Sparse:\n"
        repr_str += "Trigger Jet Momentum Range: " + str(self.minTriggerJetMomentum) + " - " + str(self.maxTriggerJetMomentum) + "\n"
        repr_str += "Associated Hadron Momentum Range: " + str(self.minAssociatedHadronMomentum) + " - " + str(self.maxAssociatedHadronMomentum) + "\n"
        repr_str += "Event Activity Range: " + str(self.minEventActivity) + " - " + str(self.maxEventActivity) + "\n"
        repr_str += "Z Vertex Range: " + str(self.minZVertex) + " - " + str(self.maxZVertex) + "\n"
        repr_str += "Delta Phi Range: " + str(self.minDeltaPhi) + " - " + str(self.maxDeltaPhi) + "\n"
        repr_str += "Delta Eta Range: " + str(self.minDeltaEta) + " - " + str(self.maxDeltaEta) + "\n"
        #repr_str += "Associated Hadron Eta Range: " + str(self.minAssociatedHadronEta) + " - " + str(self.maxAssociatedHadronEta) + "\n"
        repr_str += "Pion TPC nSigma Range: " + str(self.minPionTPCnSigma) + " - " + str(self.maxPionTPCnSigma) + "\n"
        repr_str += "Pion TOF nSigma Range: " + str(self.minPionTOFnSigma) + " - " + str(self.maxPionTOFnSigma) + "\n"
        repr_str += "Proton TOF nSigma Range: " + str(self.minProtonTOFnSigma) + " - " + str(self.maxProtonTOFnSigma) + "\n"
        repr_str += "Kaon TOF nSigma Range: " + str(self.minKaonTOFnSigma) + " - " + str(self.maxKaonTOFnSigma) + "\n"
        if self.analysisType != AnalysisType.PP:
            repr_str += "Event Plane Angle Range: " + str(self.minEventPlaneAngle) + " - " + str(self.maxEventPlaneAngle) + "\n"
        return repr_str