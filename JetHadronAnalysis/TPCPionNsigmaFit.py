
import numpy as np
import uncertainties
from JetHadronAnalysis.Types import AnalysisType, ParticleType
from JetHadronAnalysis.Fitting.TPCnSigmaFitting import piKpInc_generalized_fit, piKpInc_generalized_jac, upiKpInc_generalized_fit, gauss, generalized_gauss, ugauss, ugeneralized_gauss

import scipy.optimize as opt
from functools import partial
from typing import List
from ROOT import TH1D #type: ignore

class FitTPCPionNsigma:
    def __init__(self):
        self.initial_parameters = None
        self.bounds = None
        self.fittingFunction = piKpInc_generalized_fit

    def setInitialParameters(self, initial_parameters):
        self.initial_parameters = initial_parameters

    def setBounds(self, bounds):
        self.bounds = bounds

    def initializeDefaultParameters(self, analysisType:AnalysisType, current_associated_hadron_momentum_bin):

        inclusive_p0 = [80,15,5]
        inclusive_bounds = [[0,0,0],[100000,100000,100000]]

        generalized_p0 = [0.1, 0.1]
        generalized_bounds = [[-6, -6], [6, 6]]

        if analysisType==AnalysisType.PP:
            if current_associated_hadron_momentum_bin.value==1:
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
            if current_associated_hadron_momentum_bin.value==1:
                p0 = [-0.5, 0.0, 1.0,  0.5, 0.5, 0.5, 100,100, 1.1, 10,100, 10, 1.1,100, 100] + inclusive_p0+ generalized_p0
                bounds = [[-1.5, -0.5, -1.5, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [1.5, 0.5, 1.5, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                
                bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]
            else:
                p0 = [-1.0, 0.0, 1.0,  0.5, 0.5, 0.5, 100,100, 1.1, 10,100, 10, 1.1,100, 100] + inclusive_p0+ generalized_p0
                bounds = [[-1.5, -0.5, -1.5, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [1.5, 0.5, 1.5, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                
                bounds = [bounds[0]+inclusive_bounds[0]+generalized_bounds[0], bounds[1]+inclusive_bounds[1]+generalized_bounds[1]]
            
        self.initial_parameters = p0
        self.bounds = bounds

    @classmethod
    def prepareData(cls, pionEnhancedTPCnSigma:TH1D, protonEnhancedTPCnSigma:TH1D, kaonEnhancedTPCnSigma:TH1D, inclusiveTPCnSigma:TH1D):
        nbins = pionEnhancedTPCnSigma.GetNbinsX()
        x = np.array([pionEnhancedTPCnSigma.GetBinCenter(i) for i in range(1, nbins+1)])
        y_pi = np.array([pionEnhancedTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])
        y_p = np.array([protonEnhancedTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])
        y_k = np.array([kaonEnhancedTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])
        y_inc = np.array([inclusiveTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])
        y = [y_pi, y_p, y_k, y_inc]
        yerr_pi = np.array([pionEnhancedTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])
        yerr_p = np.array([protonEnhancedTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])
        yerr_k = np.array([kaonEnhancedTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])
        yerr_inc = np.array([inclusiveTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])
        yerr = [yerr_pi, yerr_p, yerr_k, yerr_inc]

        return x, y, yerr

    def performFit(self, x:np.ndarray, y:List[np.ndarray], yerr:List[np.ndarray]):
        non_zero_masks = [y[i] != 0 for i in range(len(y))]

        y = [y[i][non_zero_masks[i]] for i in range(len(y))]

        yerr = [yerr[i][non_zero_masks[i]] for i in range(len(yerr))]
        #breakpoint()
        final_parameters, covariance = opt.curve_fit(partial(self.fittingFunction, non_zero_masks), x, np.hstack(y), p0=self.initial_parameters, sigma=np.hstack(yerr), bounds=self.bounds,jac=partial(piKpInc_generalized_jac, non_zero_masks))

        return final_parameters, covariance
    
    def computeAveragePIDFractions(self, optimal_params, covariance, n_enhanced_associated_hadrons:dict):
        '''
        Computes PID fractions from the optimal fit params
        '''
        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = uncertainties.correlated_values(optimal_params, covariance)

        int_x = np.linspace(-100, 100, 1000)
        int_y = upiKpInc_generalized_fit(None, int_x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak)


        pionEnhNorm  = np.trapz(int_y[:1000], int_x)
        protonEnhNorm = np.trapz(int_y[1000:2000], int_x)
        kaonEnhNorm  = np.trapz(int_y[2000:3000], int_x)
        inclusiveNorm= np.trapz(int_y[3000:], int_x)

        gpp = np.trapz(ugeneralized_gauss(int_x, mup, sigp, app, alphap), int_x)/protonEnhNorm
        gppi = np.trapz(ugauss(int_x, mupi, sigpi, apip), int_x)/protonEnhNorm
        gpk = np.trapz(ugeneralized_gauss(int_x, muk, sigk, akp, alphak), int_x)/protonEnhNorm

        gpip = np.trapz(ugeneralized_gauss(int_x, mup, sigp, appi, alphap), int_x)/pionEnhNorm
        gpipi = np.trapz(ugauss(int_x, mupi, sigpi, apipi), int_x)/pionEnhNorm
        gpik = np.trapz(ugeneralized_gauss(int_x, muk, sigk, akpi, alphak), int_x)/pionEnhNorm

        gkp = np.trapz(ugeneralized_gauss(int_x, mup, sigp, apk, alphap), int_x)/kaonEnhNorm
        gkpi = np.trapz(ugauss(int_x, mupi, sigpi, apik), int_x)/kaonEnhNorm
        gkk = np.trapz(ugeneralized_gauss(int_x, muk, sigk, akk, alphak), int_x)/kaonEnhNorm

        gincp = np.trapz(ugeneralized_gauss(int_x, mup, sigp, apinc, alphap), int_x)/inclusiveNorm
        gincpi = np.trapz(ugauss(int_x, mupi, sigpi, apiinc), int_x)/inclusiveNorm
        ginck = np.trapz(ugeneralized_gauss(int_x, muk, sigk, akinc, alphak), int_x)/inclusiveNorm

        fpp = (gpp/gincp)*(n_enhanced_associated_hadrons[ParticleType.PROTON]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])
        fppi = (gppi/gincpi)*(n_enhanced_associated_hadrons[ParticleType.PROTON]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])
        fpk = (gpk/ginck)*(n_enhanced_associated_hadrons[ParticleType.PROTON]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])

        fpip = (gpip/gincp)*(n_enhanced_associated_hadrons[ParticleType.PION]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])
        fpipi = (gpipi/gincpi)*(n_enhanced_associated_hadrons[ParticleType.PION]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])
        fpik = (gpik/ginck)*(n_enhanced_associated_hadrons[ParticleType.PION]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])

        fkp = (gkp/gincp)*(n_enhanced_associated_hadrons[ParticleType.KAON]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])
        fkpi = (gkpi/gincpi)*(n_enhanced_associated_hadrons[ParticleType.KAON]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])
        fkk = (gkk/ginck)*(n_enhanced_associated_hadrons[ParticleType.KAON]/n_enhanced_associated_hadrons[ParticleType.INCLUSIVE])

        sum_of_particles_used = n_enhanced_associated_hadrons[ParticleType.PION]+n_enhanced_associated_hadrons[ParticleType.PROTON]+n_enhanced_associated_hadrons[ParticleType.KAON] # Hack to get appropriate fractions, replace once the OTHER bin is added
        pionFraction = 1/3*(gpipi*pionEnhNorm/(fpipi)+gppi*protonEnhNorm/(fppi)+gkpi*kaonEnhNorm/(fkpi))/sum_of_particles_used#n_enhanced_associated_hadrons[ParticleType.INCLUSIVE]
        protonFraction = 1/3*(gpip*pionEnhNorm/(fpip)+gpp*protonEnhNorm/(fpp)+gkp*kaonEnhNorm/(fkp))/sum_of_particles_used#n_enhanced_associated_hadrons[ParticleType.INCLUSIVE]
        kaonFraction = 1/3*(gpik*pionEnhNorm/(fpik)+gpk*protonEnhNorm/(fpk)+gkk*kaonEnhNorm/(fkk))/sum_of_particles_used#n_enhanced_associated_hadrons[ParticleType.INCLUSIVE]

        return [pionFraction.n, protonFraction.n, kaonFraction.n], [pionFraction.s, protonFraction.s, kaonFraction.s]
