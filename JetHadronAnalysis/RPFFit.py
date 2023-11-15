from functools import partial
import sqlite3
import pickle
from typing import List
import scipy.optimize as opt
import numpy as np
import uncertainties
from ROOT import TH1D #type:ignore
from JetHadronAnalysis.Types import AnalysisType, AssociatedHadronMomentumBin, TriggerJetMomentumBin, ParticleType
from JetHadronAnalysis.Fitting.RPF import simultaneous_fit, simultaneous_err, initial_parameter_defaults, bounds, resolution_parameters

class RPFFit:
    def __init__(self, analysisType:AnalysisType, currentTriggerJetMomentumBin:TriggerJetMomentumBin, currentAssociatedHadronMomentumBin:AssociatedHadronMomentumBin, currentParticleSelection: ParticleType):
        self.analysisType = analysisType
        self.currentTriggerJetMomentumBin = currentTriggerJetMomentumBin
        self.currentAssociatedHadronMomentumBin = currentAssociatedHadronMomentumBin
        self.currentParticleSelection = currentParticleSelection

        self.initialParameters = None 
        self.bounds = bounds 
        self.fittingFunction = simultaneous_fit 
        self.fittingErrorFunction = simultaneous_err

        self.databaseConnection = None 
        self.initializeDatabase()


    def initializeDatabase(self):
        '''
        establish connection to the parameter and particle fraction database named RPF.db
        '''
        self.databaseConnection = sqlite3.connect("RPF.db")
        cursor = self.databaseConnection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS fit_parameters(analysis_type TEXT, trigger_momentum_bin INTEGER, associated_momentum_bin INTEGER, particle_species TEXT, reduced_chi2 REAL, background_level REAL, v2 REAL, v3 REAL, v4 REAL, va2 REAL, va4 REAL, background_level_error REAL, v2_error REAL, v3_error REAL, v4_error REAL, va2_error REAL, va4_error REAL, covariance_matrix BLOB, PRIMARY KEY(analysis_type, trigger_momentum_bin, associated_momentum_bin, particle_species))")
        self.databaseConnection.commit()

    def setInitialParameters(self, initialParameters):
        self.initialParameters = initialParameters

    def setBounds(self, bounds): 
        self.bounds = bounds

    def setDefaultParameters(self):
        self.initialParameters = initial_parameter_defaults[self.analysisType,self.currentTriggerJetMomentumBin,self.currentAssociatedHadronMomentumBin]

    @classmethod
    def prepareData(cls, inPlaneBackgroundAzimuthalCorrelationFunction:TH1D, midPlaneBackgroundAzimuthalCorrelationFunction:TH1D, outOfPlaneBackgroundAzimuthalCorrelationFunction:TH1D):
        '''
        prepare data for fitting

        Args:
            inPlaneBackgroundAzimuthalCorrelationFunction (TH1D): Azimuthal correlation function for in-plane background
            midPlaneBackgroundAzimuthalCorrelationFunction (TH1D): Azimuthal correlation function for mid-plane background
            outOfPlaneBackgroundAzimuthalCorrelationFunction (TH1D): Azimuthal correlation function for out-of-plane background
        Returns:
            x (np.ndarray): numpy array of x values for fitting, common x-axis for each reaction plane bin
            y (List[np.narray]): list of numpy arrays of y values for fitting, y-values from each reaction plane bin
            yerr (List[np.ndarray]): list of numpy arrays of y errors for fitting, y-errors from each reaction plane bin
        '''
        x = np.array([inPlaneBackgroundAzimuthalCorrelationFunction.GetBinCenter(i) for i in range(1, inPlaneBackgroundAzimuthalCorrelationFunction.GetNbinsX()+1)])
        y = [
                np.array([inPlaneBackgroundAzimuthalCorrelationFunction.GetBinContent(i) for i in range(1, inPlaneBackgroundAzimuthalCorrelationFunction.GetNbinsX()+1)]),
                np.array([midPlaneBackgroundAzimuthalCorrelationFunction.GetBinContent(i) for i in range(1, midPlaneBackgroundAzimuthalCorrelationFunction.GetNbinsX()+1)]),
                np.array([outOfPlaneBackgroundAzimuthalCorrelationFunction.GetBinContent(i) for i in range(1, outOfPlaneBackgroundAzimuthalCorrelationFunction.GetNbinsX()+1)])
            ]
        yerr = [
                np.array([inPlaneBackgroundAzimuthalCorrelationFunction.GetBinError(i) for i in range(1, inPlaneBackgroundAzimuthalCorrelationFunction.GetNbinsX()+1)]),
                np.array([midPlaneBackgroundAzimuthalCorrelationFunction.GetBinError(i) for i in range(1, midPlaneBackgroundAzimuthalCorrelationFunction.GetNbinsX()+1)]),
                np.array([outOfPlaneBackgroundAzimuthalCorrelationFunction.GetBinError(i) for i in range(1, outOfPlaneBackgroundAzimuthalCorrelationFunction.GetNbinsX()+1)])
            ]
        return x, y, yerr
    
    def chi2OverNDF(self, optimal_params, covariance, x, y, yerr, non_zero_masks=None):
        '''
        Computes the chi2/ndf of the fit
        '''
        if non_zero_masks is None:
            non_zero_masks = [y[i] != 0 for i in range(len(y))]

            y = np.hstack([y[i][non_zero_masks[i]] for i in range(len(y))])

            yerr = np.hstack([yerr[i][non_zero_masks[i]] for i in range(len(yerr))])
        y_fit = self.fittingFunction(non_zero_masks, *resolution_parameters[self.analysisType].values(), x,  *optimal_params)
        chi2 = np.sum((y-y_fit)**2/yerr**2)
        ndf = len(x)*3-len(optimal_params) # here we multiply by four because we have four different ys that are fitting simultaneously
        return chi2/ndf
    
    def performFit(self, x:np.ndarray, y:List[np.ndarray], yerr:List[np.ndarray]):
        '''
        performs fit using scipy.optimize.curve_fit removing points with y=yerr=0

        Args:
            x (np.ndarray): numpy array of x values for fitting, common x-axis for each reaction plane bin
            y (List[np.narray]): numpy array of y values for fitting, y-values from each reaction plane bin are concatenated using np.hstack
            yerr (List[np.ndarray]): numpy array of y errors for fitting, y-errors from each reaction plane bin are concatenated using np.hstack
        Returns:
            optimalParameters (np.ndarray): numpy array of optimal parameters from fit
            optimalParameterCovariance (np.ndarray): numpy array of optimal parameter covariance errors from fit
            reducedChi2 (float): reduced chi2 from fit
        '''
        non_zero_masks = [y[i] != 0 for i in range(len(y))]
        y = [y[i][non_zero_masks[i]] for i in range(len(y))]
        yerr = [yerr[i][non_zero_masks[i]] for i in range(len(yerr))]
        optimalParameters, covarianceMatrix = opt.curve_fit(partial(self.fittingFunction, non_zero_masks, *resolution_parameters[self.analysisType].values()), x, np.hstack(y), sigma=np.hstack(yerr), p0=self.initialParameters, bounds=self.bounds)#, jac=partial(self.fittingErrorFunction, non_zero_masks, *resolution_parameters[self.analysisType].values()))

        reducedChi2 = self.chi2OverNDF(optimalParameters, covarianceMatrix, x, np.hstack(y), np.hstack(yerr), non_zero_masks)

        B, v2, v3, v4, va2, va4 = uncertainties.correlated_values(optimalParameters, covarianceMatrix)

        cov_pickled = pickle.dumps(covarianceMatrix, pickle.HIGHEST_PROTOCOL)
        cov_pickled_binary = sqlite3.Binary(cov_pickled)


        if self.databaseConnection is not None:
            cursor = self.databaseConnection.cursor()
            cursor.execute("REPLACE INTO fit_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (self.analysisType.name, self.currentTriggerJetMomentumBin.value,  self.currentAssociatedHadronMomentumBin.value, self.currentParticleSelection.name, reducedChi2, B.n, v2.n, v3.n, v4.n, va2.n, va4.n, B.s, v2.s, v3.s, v4.s, va2.s, va4.s, cov_pickled_binary))
            self.databaseConnection.commit()
            
        return optimalParameters, covarianceMatrix, reducedChi2

