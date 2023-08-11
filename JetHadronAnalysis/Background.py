from ROOT import TH1D # type: ignore
from JetHadronAnalysis.Types import AnalysisType

class BackgroundFunction:
    def __init__(self, backgroundCorrelationFunction: TH1D, analysisType:AnalysisType) -> None:
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
        
    def error(self, deltaPhi: float):
        '''
        Returns the error on the background function evaluated at deltaPhi
        '''
        # TODO add in RPF stuff, TBD what that is
        if self.analysisType == AnalysisType.PP:
            return self.ppBackgroundError(deltaPhi)
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
    
    def ppBackgroundError(self, deltaPhi: float):
        '''
        Returns the error on the background function evaluated at deltaPhi in proton-proton collisions
        '''
        # if this is the first time the function is being called, compute the error on the avcerage value of the background correlation function
        if not hasattr(self, "backgroundError"):
            sumOfSquaredErrors = 0
            for i in range(1, self.backgroundCorrelationFunction.GetNbinsX() + 1):
                sumOfSquaredErrors += self.backgroundCorrelationFunction.GetBinError(i)**2
            unbiasedNorm = self.backgroundCorrelationFunction.GetNbinsX() * (self.backgroundCorrelationFunction.GetNbinsX() - 1)
            self.backgroundError = (sumOfSquaredErrors / unbiasedNorm)**0.5

        # return the error on the average value
        return self.backgroundError