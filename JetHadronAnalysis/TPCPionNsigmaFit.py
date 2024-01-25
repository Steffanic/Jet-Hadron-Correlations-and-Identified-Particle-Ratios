
import json
import numpy as np
import uncertainties
import sqlite3
from JetHadronAnalysis.Types import AnalysisType, ParticleType, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Fitting.TPCnSigmaFitting import piKpInc_generalized_fit, Inc_generalized_fit, piKpInc_generalized_error, piKpInc_generalized_jac, upiKpInc_generalized_fit, gauss, generalized_gauss, ugauss, ugeneralized_gauss

import scipy.optimize as opt
from functools import partial
from typing import List
from ROOT import TH1D #type: ignore
import logging as lg

fitting_logger = lg.getLogger("fitting")
fitting_logger.addHandler(lg.FileHandler("fitting.log"))

class FitTPCPionNsigma:
    def __init__(self, analysisType:AnalysisType, current_region: Region, current_associated_hadron_momentum_bin:AssociatedHadronMomentumBin):

        self.analysisType = analysisType
        self.current_region = current_region
        self.current_associated_hadron_momentum_bin = current_associated_hadron_momentum_bin

        self.initial_parameters = None
        self.bounds = None
        self.fittingFunction = piKpInc_generalized_fit
        self.yieldFittingFunction = Inc_generalized_fit
        self.fittingErrorFunction = piKpInc_generalized_error
        self.pionFittingFunction = gauss
        self.protonFittingFunction = generalized_gauss
        self.kaonFittingFunction = generalized_gauss
        self.databaseConnection = None
        self.initializeDatabase()

    def __del__(self):
        if self.databaseConnection is not None:
            self.databaseConnection.close()

    def setInitialParameters(self, initial_parameters):
        self.initial_parameters = initial_parameters

    def setBounds(self, bounds):
        self.bounds = bounds

    def initializeDefaultParameters(self, from_file=True):
        if from_file:
            with open("/home/steffanic/Projects/Jet-Hadron-Correlations-and-Identified-Particle-Ratios/JetHadronAnalysis/Fitting/TPCnSigmaFitInitialParams.json", "r") as f:
                initial_parameters = json.load(f)

            if self.analysisType==AnalysisType.PP:
                initial_parameters = initial_parameters["pp"]
            elif self.analysisType==AnalysisType.SEMICENTRAL:
                initial_parameters = initial_parameters["semicentral"]
            elif self.analysisType==AnalysisType.CENTRAL:
                initial_parameters = initial_parameters["central"]
            else:
                raise ValueError("Invalid analysis type, no initial parameters available for this analysis type", self.analysisType)

            if self.current_region==Region.BACKGROUND:
                initial_parameters = initial_parameters["background"]
            elif self.current_region==Region.INCLUSIVE:
                initial_parameters = initial_parameters["inclusive"]
            elif self.current_region==Region.NEAR_SIDE_SIGNAL:
                initial_parameters = initial_parameters["near_side"]
            elif self.current_region==Region.AWAY_SIDE_SIGNAL:
                initial_parameters = initial_parameters["away_side"]
            else:
                raise ValueError("Invalid region, no initial parameters available for this region", self.current_region)
            
            initial_parameters = initial_parameters[self.current_associated_hadron_momentum_bin.name]
            self.initial_parameters = list(initial_parameters["p0"].values())
            self.bounds = initial_parameters["bounds"]


        else:

            if self.analysisType==AnalysisType.PP:
                inclusive_p0 = [8,100,12]
                if self.current_region==Region.BACKGROUND:
                    inclusive_p0 = [0.8, 10, 1.2]
                inclusive_bounds = [[0,0,0],[100000,100000,100000]]

                generalized_p0 = [0.1, 0.1]
                generalized_bounds = [[-4, -4], [4, 4]]
                if self.current_associated_hadron_momentum_bin.value==1:
                    p0 = [2.5, 0, -.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100]+inclusive_p0 + generalized_p0
                    if self.current_region==Region.BACKGROUND:
                        p0 = [2.5, 0, -.5, 0.5, 0.5, 0.5, 10,10, 0.11, 1,10, 1, 0.11,10, 10]+inclusive_p0 + generalized_p0
                    bounds = [[-6, -0.05, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [6, 0.05, 6, 100.0, 100.0, 100.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1] + generalized_bounds[1]]

                elif self.current_associated_hadron_momentum_bin.value>1 and self.current_associated_hadron_momentum_bin.value<4:
                    p0 = [-1.0, 0, -2.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100] + inclusive_p0+ generalized_p0
                    if self.current_region==Region.BACKGROUND:
                        p0 = [-1.0, 0, -2.5, 0.5, 0.5, 0.5, 10,10, 0.11, 1,10, 1, 0.11,10, 10] + inclusive_p0+ generalized_p0
                    bounds = [[-6, -0.05, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [0, 0.05, 0, 100.0, 100.0, 100.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0]+ generalized_bounds[0], bounds[1]+inclusive_bounds[1] + generalized_bounds[1]]
                else:
                    p0 = [-3.5, 0, -2.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100] + inclusive_p0+ generalized_p0
                    if self.current_region==Region.BACKGROUND:
                        p0 = [-3.5, 0, -2.5, 0.5, 0.5, 0.5, 10,10, 0.11, 1,10, 1, 0.11,10, 10] + inclusive_p0+ generalized_p0
                    bounds = [[-6, -0.05, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [0, 0.05, 0, 100.0, 100.0, 100.0, 100000,100000,10000,100000,100000,100000,10000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]
            else:
                inclusive_p0 = [80,1000,120]
                inclusive_bounds = [[0,0,0],[100000,100000,100000]]

                generalized_p0 = [0.5, 0.1]
                generalized_bounds = [[-6, -6], [6, 6]]
                if self.current_associated_hadron_momentum_bin.value==1:
                    p0 = [3.0, 0.0, -2.0,  1.5, 1.5, 1.5, 1000,2000, 100, 100,10000, 100, 100,1000, 1000] + inclusive_p0+ generalized_p0
                    bounds = [[-5.0, -0.5, -5.0, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [5.0, 0.5, 5.0, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]
                if self.current_associated_hadron_momentum_bin.value>5:
                    p0 = [3.0, 0.0, -2.0,  1.5, 1.5, 1.5, 10,20, 1, 1,100, 1, 1,10, 10] + inclusive_p0+ generalized_p0
                    bounds = [[-5.0, -0.5, -5.0, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [5.0, 0.5, 5.0, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]
                else:
                    p0 = [-1.0, 0.0, 1.0,  1.5, 1.5, 1.5, 100,100, 1.1, 10,100, 10, 1.1,100, 100] + inclusive_p0+ generalized_p0
                    bounds = [[-1.5, -0.5, -1.5, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [1.5, 0.5, 1.5, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0]+generalized_bounds[0], bounds[1]+inclusive_bounds[1]+generalized_bounds[1]]
                
            self.initial_parameters = p0
            self.bounds = bounds

    def initializeDatabase(self):
        '''
        establish connection to the parameter and particle fraction database named PID.db
        '''
        fitting_logger.info("Initializing database")
        self.databaseConnection = sqlite3.connect("PID.db")
        cursor = self.databaseConnection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS particle_fractions(analysis_type TEXT, region TEXT, momentum_bin INTEGER, pion_fraction REAL, proton_fraction REAL, kaon_fraction REAL, pion_fraction_error REAL, proton_fraction_error REAL, kaon_fraction_error REAL, PRIMARY KEY(analysis_type, region, momentum_bin))")
        cursor.execute("CREATE TABLE IF NOT EXISTS fit_parameters(analysis_type TEXT, region TEXT, momentum_bin INTEGER, reduced_chi2 REAL, mu_pion REAL, mu_proton REAL, mu_kaon REAL, sigma_pion REAL, sigma_proton REAL, sigma_kaon REAL, alpha_proton REAL, alpha_kaon REAL, pion_enhanced_pion_fraction REAL, pion_enhanced_proton_fraction REAL, pion_enhanced_kaon_fraction REAL, proton_enhanced_pion_fraction REAL, proton_enhanced_proton_fraction REAL, proton_enhanced_kaon_fraction REAL, kaon_enhanced_pion_fraction REAL, kaon_enhanced_proton_fraction REAL, kaon_enhanced_kaon_fraction REAL, inclusive_pion_fraction REAL, inclusive_proton_fraction REAL, inclusive_kaon_fraction REAL, mu_pion_error REAL, mu_proton_error REAL, mu_kaon_error REAL, sigma_pion_error REAL, sigma_proton_error REAL, sigma_kaon_error REAL, alpha_proton_error REAL, alpha_kaon_error REAL, pion_enhanced_pion_fraction_error REAL, pion_enhanced_proton_fraction_error REAL, pion_enhanced_kaon_fraction_error REAL, proton_enhanced_pion_fraction_error REAL, proton_enhanced_proton_fraction_error REAL, proton_enhanced_kaon_fraction_error REAL, kaon_enhanced_pion_fraction_error REAL, kaon_enhanced_proton_fraction_error REAL, kaon_enhanced_kaon_fraction_error REAL, inclusive_pion_fraction_error REAL, inclusive_proton_fraction_error REAL, inclusive_kaon_fraction_error REAL , covariance BLOB, PRIMARY KEY(analysis_type, region, momentum_bin))")
        self.databaseConnection.commit()


    @classmethod
    def prepareData(cls, pionEnhancedTPCnSigma:TH1D, protonEnhancedTPCnSigma:TH1D, kaonEnhancedTPCnSigma:TH1D, inclusiveTPCnSigma:TH1D):
        nbins = pionEnhancedTPCnSigma.GetNbinsX()
        x = np.array([pionEnhancedTPCnSigma.GetBinCenter(i) for i in range(1, nbins+1)])
        y_pi = np.array([pionEnhancedTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])/pionEnhancedTPCnSigma.Integral()
        y_p = np.array([protonEnhancedTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])/protonEnhancedTPCnSigma.Integral()
        y_k = np.array([kaonEnhancedTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])/kaonEnhancedTPCnSigma.Integral()
        y_inc = np.array([inclusiveTPCnSigma.GetBinContent(i) for i in range(1, nbins+1)])/inclusiveTPCnSigma.Integral()
        y = [y_pi, y_p, y_k, y_inc]
        yerr_pi = np.array([pionEnhancedTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])/pionEnhancedTPCnSigma.Integral()
        yerr_p = np.array([protonEnhancedTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])/protonEnhancedTPCnSigma.Integral()
        yerr_k = np.array([kaonEnhancedTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])/kaonEnhancedTPCnSigma.Integral()
        yerr_inc = np.array([inclusiveTPCnSigma.GetBinError(i) for i in range(1, nbins+1)])/inclusiveTPCnSigma.Integral()
        yerr = [yerr_pi, yerr_p, yerr_k, yerr_inc]

        return x, y, yerr

    def performShapeFit(self, x:np.ndarray, y:List[np.ndarray], yerr:List[np.ndarray]):
        non_zero_masks = [y[i] != 0 for i in range(len(y))]

        y = [y[i][non_zero_masks[i]] for i in range(len(y))]

        yerr = [yerr[i][non_zero_masks[i]] for i in range(len(yerr))]
        #breakpoint()
        final_parameters, covariance = opt.curve_fit(partial(self.fittingFunction, non_zero_masks), x, np.hstack(y), p0=self.initial_parameters, sigma=np.hstack(yerr), bounds=self.bounds,jac=partial(piKpInc_generalized_jac, non_zero_masks), absolute_sigma=True, maxfev=10000000)
        # fitting_logger.info(f"Fit parameters: {final_parameters}")
        # fitting_logger.info(f"Fit covariance: {covariance}")
        reduced_chi2 = self.chi2OverNDF(final_parameters, covariance, x, np.hstack(y), np.hstack(yerr), non_zero_masks)
        
        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = uncertainties.correlated_values(final_parameters, covariance)
        # put the parameters into a the database 
        if self.databaseConnection is not None:
            cursor = self.databaseConnection.cursor()
            cursor.execute("REPLACE INTO fit_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (self.analysisType.name, self.current_region.name, self.current_associated_hadron_momentum_bin.value, reduced_chi2, mupi.n, mup.n, muk.n, sigpi.n, sigp.n, sigk.n, alphap.n, alphak.n, apipi.n, appi.n, akpi.n, apip.n, app.n, akp.n, apik.n, apk.n, akk.n, apiinc.n, apinc.n, akinc.n, mupi.s, mup.s, muk.s, sigpi.s, sigp.s, sigk.s, alphap.s, alphak.s, apipi.s, appi.s, akpi.s, apip.s, app.s, akp.s, apik.s, apk.s, akk.s, apiinc.s, apinc.s, akinc.s, covariance.tobytes()))
            self.databaseConnection.commit()
        return final_parameters, covariance
    
    def performYieldFit(self, x:np.ndarray, y:np.ndarray, yerr:np.ndarray, optimal_parameters:np.ndarray, shape_covariance:np.ndarray):
        non_zero_mask = (y != 0)

        y = y[non_zero_mask]

        yerr = yerr[non_zero_mask] 

        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = optimal_parameters
        #breakpoint()
        initialYieldParameters = apinc, apiinc, akinc
        final_parameters, covariance = opt.curve_fit(partial(self.yieldFittingFunction, non_zero_mask, mup, mupi, muk, sigp, sigpi, sigk, alphap, alphak), x, y, p0=initialYieldParameters, sigma=yerr, bounds=(self.bounds[0][-5:-2], self.bounds[1][-5:-2]), absolute_sigma=True, maxfev=10000000)
        # fitting_logger.info(f"Fit parameters: {final_parameters}")
        # fitting_logger.info(f"Fit covariance: {covariance}")
        apinc, apiinc, akinc = uncertainties.correlated_values(final_parameters, covariance)
        y_fit = self.yieldFittingFunction(non_zero_mask, mup, mupi, muk, sigp, sigpi, sigk, alphap, alphak,x, *final_parameters)
        reduced_chi2 = np.sum((y-y_fit)**2/yerr**2)/(len(x)-len(final_parameters)) 

        total_covariance = shape_covariance
        total_covariance[-3:, -3:] = covariance


        if self.databaseConnection is not None:
            cursor = self.databaseConnection.cursor()
            cursor.execute("UPDATE fit_parameters SET inclusive_pion_fraction=?, inclusive_proton_fraction=?, inclusive_kaon_fraction=?, inclusive_pion_fraction_error=?, inclusive_proton_fraction_error=?, inclusive_kaon_fraction_error=?, covariance=? WHERE analysis_type=? AND region=? AND momentum_bin=? ", (apiinc.n, apinc.n, akinc.n, apiinc.s, apinc.s, akinc.s, total_covariance.tobytes(), self.analysisType.name, self.current_region.name, self.current_associated_hadron_momentum_bin.value))
            self.databaseConnection.commit()

        return final_parameters, covariance, reduced_chi2

    
    def chi2OverNDF(self, optimal_params, covariance, x, y, yerr, non_zero_masks=None):
        '''
        Computes the chi2/ndf of the fit
        '''
        if non_zero_masks is None:
            non_zero_masks = [y[i] != 0 for i in range(len(y))]

            y = np.hstack([y[i][non_zero_masks[i]] for i in range(len(y))])

            yerr = np.hstack([yerr[i][non_zero_masks[i]] for i in range(len(yerr))])
        y_fit = self.fittingFunction(non_zero_masks, x, *optimal_params)
        chi2 = np.sum((y-y_fit)**2/yerr**2)
        ndf = len(x)*4-len(optimal_params) # here we multiply by four because we have four different ys that are fitting simultaneously
        return chi2/ndf
    
    def computeAveragePIDFractions(self, optimal_params, covariance, n_enhanced_associated_hadrons:dict):
        '''
        Computes PID fractions from the optimal fit params
        '''

        # get the parameters as ufloats
        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = uncertainties.correlated_values(optimal_params, covariance)

        # generate some bootstrap samples of the fit parameters
        n_samples = 50
        shape_samples = np.random.multivariate_normal(optimal_params, covariance, n_samples) # shape (n_samples, n_params)
        # now just sample apinc, apiinc, akinc with the appropriate subset of covariance matrix
        # this would be covariance[-5:-2, -5:-2] 
        yield_samples = np.random.multivariate_normal([apinc.n, apiinc.n, akinc.n], covariance[-5:-2, -5:-2], n_samples) # shape (n_samples, 3)

        int_x = np.linspace(-100, 100, 1000)
        int_y = upiKpInc_generalized_fit(None, int_x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak) # shape (4*n_x,)

        print("Computing bootstrap PID shape errors")
        int_y_shape_samples = np.array([upiKpInc_generalized_fit(None, int_x, *sample) for sample in shape_samples]) # shape (n_samples, 4*n_x)
        print("Computing bootstrap PID yields errors")
        int_y_yield_samples = np.array([upiKpInc_generalized_fit(None, int_x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, *sample, alphap, alphak) for sample in yield_samples]) # shape (n_samples, 4*n_x)

        inclusiveNorm= np.trapz(int_y[3000:], int_x)

        inclusiveNorm_shape_samples = np.array([np.trapz(int_y_shape_samples[i, 3000:], int_x) for i in range(n_samples)])
        inclusiveNorm_yield_samples = np.array([np.trapz(int_y_yield_samples[i, 3000:], int_x) for i in range(n_samples)])


        gincp = np.trapz(ugeneralized_gauss(int_x, mup, sigp, apinc, alphap), int_x)/inclusiveNorm
        gincpi = np.trapz(ugauss(int_x, mupi, sigpi, apiinc), int_x)/inclusiveNorm
        ginck = np.trapz(ugeneralized_gauss(int_x, muk, sigk, akinc, alphak), int_x)/inclusiveNorm

        gincp_shape_samples = np.array([np.trapz(ugeneralized_gauss(int_x, sample[0], sample[3], sample[15], sample[18]), int_x) for sample in shape_samples])/inclusiveNorm_shape_samples
        gincpi_shape_samples = np.array([np.trapz(ugauss(int_x, sample[1], sample[4], sample[16]), int_x) for sample in shape_samples])/inclusiveNorm_shape_samples
        ginck_shape_samples = np.array([np.trapz(ugeneralized_gauss(int_x, sample[2], sample[5], sample[17], sample[19]), int_x) for sample in shape_samples])/inclusiveNorm_shape_samples

        gincp_yield_samples = np.array([np.trapz(ugeneralized_gauss(int_x, mup, sigp, sample[0], alphap), int_x) for sample in yield_samples])/inclusiveNorm_yield_samples
        gincpi_yield_samples = np.array([np.trapz(ugauss(int_x, mupi, sigpi, sample[1]), int_x) for sample in yield_samples])/inclusiveNorm_yield_samples
        ginck_yield_samples = np.array([np.trapz(ugeneralized_gauss(int_x, muk, sigk, sample[2], alphak), int_x) for sample in yield_samples])/inclusiveNorm_yield_samples

        pionFraction = gincpi#1/3*(gpipi*pionEnhNorm/(fpipi)+gppi*protonEnhNorm/(fppi)+gkpi*kaonEnhNorm/(fkpi))/sum_of_particles_used#n_enhanced_associated_hadrons[ParticleType.INCLUSIVE]
        protonFraction = gincp#1/3*(gpip*pionEnhNorm/(fpip)+gpp*protonEnhNorm/(fpp)+gkp*kaonEnhNorm/(fkp))/sum_of_particles_used#n_enhanced_associated_hadrons[ParticleType.INCLUSIVE]
        kaonFraction = ginck#1/3*(gpik*pionEnhNorm/(fpik)+gpk*protonEnhNorm/(fpk)+gkk*kaonEnhNorm/(fkk))/sum_of_particles_used#n_enhanced_associated_hadrons[ParticleType.INCLUSIVE]

        pionFraction_pid_fit_shape_sys_err = float(np.mean([(sample - gincpi.n)**2 for sample in gincpi_shape_samples]))
        protonFraction_pid_fit_shape_sys_err = float(np.mean([(sample - gincp.n)**2 for sample in gincp_shape_samples]))
        kaonFraction_pid_fit_shape_sys_err = float(np.mean([(sample - ginck.n)**2 for sample in ginck_shape_samples]))

        pionFraction_pid_fit_yield_sys_err = float(np.mean([(sample.n - gincpi.n)**2 for sample in gincpi_yield_samples]))
        protonFraction_pid_fit_yield_sys_err = float(np.mean([(sample.n - gincp.n)**2 for sample in gincp_yield_samples]))
        kaonFraction_pid_fit_yield_sys_err = float(np.mean([(sample.n - ginck.n)**2 for sample in ginck_yield_samples]))

        # save the particle fractions to the database
        if self.databaseConnection is not None:
            cursor = self.databaseConnection.cursor()
            cursor.execute("REPLACE INTO particle_fractions VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (self.analysisType.name, self.current_region.name, self.current_associated_hadron_momentum_bin.value, pionFraction.n, protonFraction.n, kaonFraction.n, pionFraction.s, protonFraction.s, kaonFraction.s, pionFraction_pid_fit_shape_sys_err, protonFraction_pid_fit_shape_sys_err, kaonFraction_pid_fit_shape_sys_err, pionFraction_pid_fit_yield_sys_err, protonFraction_pid_fit_yield_sys_err, kaonFraction_pid_fit_yield_sys_err))
            self.databaseConnection.commit()

        return {ParticleType.PION: pionFraction.n, ParticleType.PROTON:protonFraction.n, ParticleType.KAON:kaonFraction.n}, {ParticleType.PION:pionFraction.s, ParticleType.PROTON:protonFraction.s, ParticleType.KAON:kaonFraction.s}, {ParticleType.PION:pionFraction_pid_fit_shape_sys_err, ParticleType.PROTON:protonFraction_pid_fit_shape_sys_err, ParticleType.KAON:kaonFraction_pid_fit_shape_sys_err}, {ParticleType.PION:pionFraction_pid_fit_yield_sys_err, ParticleType.PROTON:protonFraction_pid_fit_yield_sys_err, ParticleType.KAON:kaonFraction_pid_fit_yield_sys_err}
