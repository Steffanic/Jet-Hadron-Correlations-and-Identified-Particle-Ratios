from ROOT import TH1 # type: ignore
from JetHadronAnalysis.Types import AnalysisType

class BackgroundFunction:
    def __init__(self, backgroundCorrelationFunction: TH1, analysisType:AnalysisType) -> None:
        self.analysisType = analysisType
        self.backgroundCorrelationFunction = backgroundCorrelationFunction

        # TODO add in RPF stuff, TBD what that is

    def __call__(self, deltaPhi: float):
        '''
        Returns the background function evaluated at deltaPhi
        '''
        # TODO add in RPF stuff, TBD what that is
        if self.analysisType == AnalysisType.PP:
            return self.ppBackgroundFunction() # Because it is just the average value it does not depend on deltaPhi
        else:
            raise NotImplementedError("Analysis type not yet implemented")
        
    def ppBackgroundFunction(self):
        '''
        Returns the background function evaluated at deltaPhi in proton-proton collisions
        '''
        # if this is the first time the function is being called, compute the average value of the background correlation function
        if not hasattr(self, "backgroundAverage"):
            self.backgroundAverage = self.backgroundCorrelationFunction.Integral() / self.backgroundCorrelationFunction.GetNbinsX()
        # return the average value
        return self.backgroundAverage