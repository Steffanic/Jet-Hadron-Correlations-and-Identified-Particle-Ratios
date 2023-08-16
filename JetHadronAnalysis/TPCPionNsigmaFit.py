
from JetHadronAnalysis.Types import AnalysisType
from JetHadronAnalysis.Analysis import AssociatedHadronMomentumBin

class FitTPCPionNsigma:
    def __init__(self):
        self.initial_parameters = None
        self.bounds = None
        self.fittingFunction = 

    def setInitialParameters(self, initial_parameters):
        self.initial_parameters = initial_parameters

    def setBounds(self, bounds):
        self.bounds = bounds

    def setDefaultInitialParameters(self, analysisType:AnalysisType, current_associated_hadron_momentum_bin: AssociatedHadronMomentumBin):

        inclusive_p0 = [80,15,5]
        inclusive_bounds = [[0,0,0],[100000,100000,100000]]

        generalized_p0 = [0.1, 0.1]
        generalized_bounds = [[-6, -6], [6, 6]]

        if analysisType==AnalysisType.PP:
            if current_associated_hadron_momentum_bin==AssociatedHadronMomentumBin.PT_1_15:
                p0 = [2.5, 0, -.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100]+inclusive_p0 + generalized_p0
                bounds = [[-6, -0.1, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [6, 0.1, 6, 100.0, 100.0, 100.0, 100000,100000,10,100000,100000,100000,10,100000,100000]]
                
                bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1] + generalized_bounds[1]]
            elif current_associated_hadron_momentum_bin.value>1 and current_associated_hadron_momentum_bin.value<5:
                p0 = [-1.0, 0, -2.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100] + inclusive_p0+ generalized_p0
                bounds = [[-6, -0.1, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [0, 0.1, 0, 100.0, 100.0, 100.0, 100000,100000,10,100000,100000,100000,10,100000,100000]]
                
                bounds = [bounds[0]+inclusive_bounds[0]+ generalized_bounds[0], bounds[1]+inclusive_bounds[1] + generalized_bounds[1]]
            else:
                p0 = [-3.5, 0, -2.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100] + inclusive_p0+ generalized_p0
                bounds = [[-6, -0.1, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [0, 0.1, 0, 100.0, 100.0, 100.0, 100000,100000,10000,100000,100000,100000,10000,100000,100000]]
                
                bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]
        else:
            if current_associated_hadron_momentum_bin==AssociatedHadronMomentumBin.PT_1_15:
                p0 = [-0.5, 0.0, 1.0,  0.5, 0.5, 0.5, 100,100, 1.1, 10,100, 10, 1.1,100, 100] + inclusive_p0+ generalized_p0
                bounds = [[-1.5, -0.5, -1.5, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [1.5, 0.5, 1.5, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                
                bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]
            else:
                p0 = [-1.0, 0.0, 1.0,  0.5, 0.5, 0.5, 100,100, 1.1, 10,100, 10, 1.1,100, 100] + inclusive_p0+ generalized_p0
                bounds = [[-1.5, -0.5, -1.5, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [1.5, 0.5, 1.5, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                
                bounds = [bounds[0]+inclusive_bounds[0]+generalized_bounds[0], bounds[1]+inclusive_bounds[1]+generalized_bounds[1]]
            
        self.initial_parameters = p0
        self.bounds = bounds

    def setFittingFunction(self):
        pass