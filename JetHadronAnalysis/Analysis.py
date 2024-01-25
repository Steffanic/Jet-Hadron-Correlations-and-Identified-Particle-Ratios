# © Patrick John Steffanic 2023
# This file contains the class whose responsibility it is to manage the analysis and its configuration 

import sqlite3
from typing import Optional
import numpy as np
import os
import uncertainties
from JetHadronAnalysis.Sparse import TriggerSparse, MixedEventSparse, JetHadronSparse
from JetHadronAnalysis.Types import AnalysisType, ParticleType, NormalizationMethod, Region, TriggerJetMomentumBin, AssociatedHadronMomentumBin, ReactionPlaneBin, regionDeltaPhiRangeDictionary, regionDeltaEtaRangeDictionary, regionDeltaPhiBinCountsDictionary, speciesTOFRangeDictionary, triggerJetMomentumBinRangeDictionary, associatedHadronMomentumBinRangeDictionary, eventPlaneAngleBinRangeDictionary
from JetHadronAnalysis.Background import BackgroundFunction
from JetHadronAnalysis.TPCPionNsigmaFit import FitTPCPionNsigma
from JetHadronAnalysis.RPFFit import RPFFit
from JetHadronAnalysis.Fitting.RPF import resolution_parameters

from ROOT import TFile, TH1D # type: ignore
from enum import Enum
from math import pi

from JetHadronAnalysis.Plotting import plotArrays
from JetHadronAnalysis.PIDDB import getParticleFractionByMomentum, getParticleFractionForMomentumBin
from JetHadronAnalysis.RPFDB import getParameterByTriggerAndHadronMomentumForParticleSpecies, getParameters, getParameterErrors


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

        self.currentTriggerJetMomentumBin = TriggerJetMomentumBin.PT_20_40

        self.currentAssociatedHadronMomentumBin = AssociatedHadronMomentumBin.PT_1_15

        self.current_species=ParticleType.INCLUSIVE

        # make some dictionaries to keep track of the number of triger jets and associated hadrons in each region
        self.numberOfTriggerJets = {}
        self.numberOfAssociatedHadrons = {}

        if self.analysisType != AnalysisType.PP:
            self.reactionPlaneAngle = ReactionPlaneBin.INCLUSIVE

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
        if region == Region.BACKGROUND:
            self.JetHadron.setRegionIsBackground(True)
        else:
            self.JetHadron.setRegionIsBackground(False)
            self.JetHadron.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[region])
            self.JetHadron.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[region])
            if hasattr(self, "numberOfAssociatedHadronsDictionary"):
                self.fillNumberOfAssociatedHadronsDictionary()

    def setTriggerJetMomentumBin(self, triggerJetMomentumBin: TriggerJetMomentumBin):
        '''
        Sets the trigger jet momentum bin for the JetHadron sparse
        '''
        self.currentTriggerJetMomentumBin = triggerJetMomentumBin
        self.JetHadron.setTriggerJetMomentumRange(*triggerJetMomentumBinRangeDictionary[triggerJetMomentumBin])
        if hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()

    def setAssociatedHadronMomentumBin(self, associatedHadronMomentumBin: AssociatedHadronMomentumBin):
        '''
        Sets the associated hadron momentum bin for the JetHadron sparse
        '''
        self.currentAssociatedHadronMomentumBin = associatedHadronMomentumBin
        self.JetHadron.setAssociatedHadronMomentumRange(*associatedHadronMomentumBinRangeDictionary[associatedHadronMomentumBin])
        if hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()

    def setParticleSelectionForJetHadron(self, species: ParticleType):
        '''
        Sets the pion, kaon, and proton TOF ranges to get the particle type specified in JetHadron sparse
        '''
        self.current_species = species
        if species == ParticleType.OTHER:
            self.JetHadron.setParticleTypeIsOther(True)
        else:
            self.JetHadron.setParticleTypeIsOther(False)
            self.JetHadron.setPionTOFnSigma(*speciesTOFRangeDictionary[species][0])
            self.JetHadron.setKaonTOFnSigma(*speciesTOFRangeDictionary[species][1])
            self.JetHadron.setProtonTOFnSigma(*speciesTOFRangeDictionary[species][2])

    def setReactionPlaneAngleBin(self, reactionPlaneAngle: ReactionPlaneBin):
        '''
        Sets the reaction plane angle bin for the JetHadron sparse
        '''
        self.reactionPlaneAngle = reactionPlaneAngle
        self.JetHadron.setEventPlaneAngleRange(*eventPlaneAngleBinRangeDictionary[reactionPlaneAngle])
        if hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()

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

    def getAzimuthalCorrelationFunctionforParticleType(self, species: ParticleType, acceptanceCorrectedAzimuthalCorrelationFunction: TH1D, loadFractionsFromDB=False):
        '''
        Gets the particle fractions and scales the acceptance corrected azimuthal correlation function by the particle fraction for the specified species
        '''

        particle_fractions, particle_fraction_errors, pid_fit_shape_sys_err, pid_fit_yield_sys_err, chi2OverNDF_shape, Chi2OverNDF_yield = self.getPIDFractions(loadFractionsFromDB=loadFractionsFromDB)
        
        # now get the per species azimuthal correlation functions for each region by scaling
        species_azimuthal_correlation_function = acceptanceCorrectedAzimuthalCorrelationFunction.Clone()
        species_azimuthal_correlation_function.Scale(particle_fractions[species])
        error_band = acceptanceCorrectedAzimuthalCorrelationFunction.Clone()
        error_band.Scale(particle_fraction_errors[species])
        # convert errorbands into numpy arrays of bin contents
        sys_errors = np.array([error_band.GetBinContent(i) for i in range(1, error_band.GetNbinsX()+1)])
        # reset the errors for species_azimuthal_correlation_function to add the statistical errors and systematic errors in quadrature
        for i in range(1, species_azimuthal_correlation_function.GetNbinsX()+1):
            species_azimuthal_correlation_function.SetBinError(i, np.sqrt(species_azimuthal_correlation_function.GetBinError(i)**2 + sys_errors[i-1]**2))
        

        return species_azimuthal_correlation_function, pid_fit_shape_sys_err[species],pid_fit_yield_sys_err[species]
        
    def getYieldFromAzimuthalCorrelationFunction(self, azimuthalCorrelationFunction: TH1D, pid_fit_shape_sys_err: Optional[float] = None, pid_fit_yield_sys_err: Optional[float] = None):
        '''
        Returns the yield from the azimuthal correlation function
        '''
        # scale by  the bin width
        # but first clone it 
        azimuthalCorrelationFunction = azimuthalCorrelationFunction.Clone()
        azimuthalCorrelationFunction.Scale(self.JetHadron.getBinWidth(self.JetHadron.Axes.DELTA_PHI))
        yield_ = azimuthalCorrelationFunction.Integral()
        error_ = np.sqrt(np.sum([azimuthalCorrelationFunction.GetBinError(i)**2 for i in range(1, azimuthalCorrelationFunction.GetNbinsX()+1)]))
        pid_fit_shape_sys_err_ = None if pid_fit_shape_sys_err is None else pid_fit_shape_sys_err * yield_
        pid_fit_yield_sys_err_ = None if pid_fit_yield_sys_err is None else pid_fit_yield_sys_err * yield_
        return yield_, error_, pid_fit_shape_sys_err_, pid_fit_yield_sys_err_

    def getNormalizedDifferentialMixedEventCorrelationFunction(self, normMethod: NormalizationMethod, **kwargs):
        '''
        Returns the differential mixed event correlation function
        '''
        if "TOF" in kwargs:
            if kwargs['TOF']:
                self.MixedEvent.sethasTOF(True)
                mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
            else:
                mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
        else:
            mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
        mixedEventCorrelationFunction.Scale(1 / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_PHI) / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_ETA))

        normalization_factor = self.computeMixedEventNormalizationFactor(mixedEventCorrelationFunction, normMethod, **kwargs)

        if kwargs.get("customRegion", None) is not None:
            customRegion = kwargs.get("customRegion")
            self.MixedEvent.setDeltaPhiRange(*customRegion["DeltaPhi"])
            self.MixedEvent.setDeltaEtaRange(*customRegion["DeltaEta"])
            mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
            self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.INCLUSIVE])
            self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.INCLUSIVE])
        else:
            if self.currentRegion == Region.BACKGROUND:
                self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.BACKGROUND_ETANEG])
                self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.BACKGROUND_ETANEG])
                mixedEventCorrelationFunction_etaneg = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
                self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.BACKGROUND_ETAPOS])
                self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.BACKGROUND_ETAPOS])
                mixedEventCorrelationFunction_etapos = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
                mixedEventCorrelationFunction = mixedEventCorrelationFunction_etaneg.Clone()
                mixedEventCorrelationFunction.Add(mixedEventCorrelationFunction_etapos)
                self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.INCLUSIVE])
                self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.INCLUSIVE])
            else:
                self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[self.currentRegion])
                self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[self.currentRegion])
                mixedEventCorrelationFunction = self.MixedEvent.getProjection(self.MixedEvent.Axes.DELTA_PHI, self.MixedEvent.Axes.DELTA_ETA)
                self.MixedEvent.setDeltaPhiRange(*regionDeltaPhiRangeDictionary[Region.INCLUSIVE])
                self.MixedEvent.setDeltaEtaRange(*regionDeltaEtaRangeDictionary[Region.INCLUSIVE])
        mixedEventCorrelationFunction.Scale(1 / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_PHI) / self.MixedEvent.getBinWidth(self.MixedEvent.Axes.DELTA_ETA))
        mixedEventCorrelationFunction.Scale(1 / normalization_factor)
        return mixedEventCorrelationFunction
    
    def getRPInclusiveBackgroundCorrelationFunctionUsingRPF(self, inPlaneCorrelationFunction: TH1D, midPlaneCorrelationFunction: TH1D, outPlaneCorrelationFunction: TH1D, loadFunctionFromDB=False):
        '''
        Returns the background correlation function using the RPF method

        Check if the background function has already been computed and stored in the database
        If not, compute it and store it in the database

        Returns:
            backgroundCorrelationFunction (TH1D): The background correlation function with the same binning as the in-plane, mid-plane, and out-of-plane correlation functions
        '''
        # create a fitter instance
        fitter = RPFFit(self.analysisType, self.currentTriggerJetMomentumBin, self.currentAssociatedHadronMomentumBin, self.current_species)

        if loadFunctionFromDB:
            print("Loading RPF background from DB")
            # get the background function from the database
            optimal_params, covariance, reduced_chi2 = self.getRPFParamsAndErrorFromDB()
        else:
            fitter.setDefaultParameters()

            # prepare the data for fitting
            x, y, yerr = RPFFit.prepareData(inPlaneCorrelationFunction, midPlaneCorrelationFunction, outPlaneCorrelationFunction)

            # fit and extract the optimal fit params
            optimal_params, covariance, reduced_chi2 = fitter.performFit(x, y, yerr)
            
        # build the background function with the same binning as the correlation functions
        x_background = np.array([inPlaneCorrelationFunction.GetBinCenter(i) for i in range(1, inPlaneCorrelationFunction.GetNbinsX()+1)])
        backgroundCorrelationFunction = TH1D("backgroundCorrelationFunction", "backgroundCorrelationFunction", len(x_background), x_background[0], x_background[-1])
        backgroundContent = fitter.fittingFunction(None, *resolution_parameters[self.analysisType].values(), x_background, *optimal_params)
        backgroundError = fitter.fittingErrorFunction(None, *resolution_parameters[self.analysisType].values(), x_background, *optimal_params, pcov=covariance)

        inPlaneContent = backgroundContent[:len(x_background)]
        midPlaneContent = backgroundContent[len(x_background):2*len(x_background)]
        outPlaneContent = backgroundContent[2*len(x_background):]
        inclusiveContent = inPlaneContent + midPlaneContent + outPlaneContent

        inPlaneError = backgroundError[:len(x_background)]
        midPlaneError = backgroundError[len(x_background):2*len(x_background)]
        outPlaneError = backgroundError[2*len(x_background):]
        inclusiveError = np.sqrt(inPlaneError**2 + midPlaneError**2 + outPlaneError**2)
        # fill the background function with the optimal params
        for i in range(len(x_background)):
            backgroundCorrelationFunction.SetBinContent(i+1, inclusiveContent[i])
            backgroundCorrelationFunction.SetBinError(i+1, inclusiveError[i])
        return backgroundCorrelationFunction



    def getRPFParamsAndErrorFromDB(self):
        # get a db connection
        conn = sqlite3.connect("RPF.db")
        dbCursor = conn.cursor()
        # get the optimal paams and their errors from the database
        optimal_params = getParameters(self.analysisType, self.currentTriggerJetMomentumBin, self.currentAssociatedHadronMomentumBin, self.current_species, dbCursor)
        # get the optimal params errors from the database
        param_errors, covariance = getParameterErrors(self.analysisType, self.currentTriggerJetMomentumBin, self.currentAssociatedHadronMomentumBin, self.current_species, dbCursor)
        # close the connection
        conn.close()
        reduced_chi2 = optimal_params[-1]
        # convert the optimal params and their errors to numpy arrays
        optimal_params = np.array(optimal_params[:-1])
        return optimal_params, covariance, reduced_chi2


    def getPIDFractions(self, makeIntermediatePlots=True, loadFractionsFromDB=False):
        '''
        First checks if the PID fractions have already been computed and stored in the database
        If not, computes them and stores them in the database
        Prepares the projections for each enhanced species
        Converts them into arrays for fitting
        Fits and Extracts the optimal fit params
        Computes the PID fractions
        Returns the PID fractions
        '''
        if loadFractionsFromDB:
            print("Loading particle fractions from database")
            # get the particle fractions from the database
            particle_fractions, particle_fraction_errors, pid_fit_shape_sys_err, pid_fit_yield_sys_err = self.getParticleFractionsFromDB()
            return particle_fractions, particle_fraction_errors, pid_fit_shape_sys_err, pid_fit_yield_sys_err, None, None
        # set the region to inclusive to fit the shape parameters first saving the current region to reset it later
        fitter = FitTPCPionNsigma(self.analysisType, self.currentRegion, self.currentAssociatedHadronMomentumBin)
        currentRegion = self.currentRegion
        self.setRegion(Region.INCLUSIVE)
        # get the projections
        pionEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.PION)
        protonEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.PROTON)
        kaonEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.KAON)
        inclusiveEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.INCLUSIVE)
        # convert to arrays using FitTPCPionNsigma.prepareData 
        x, y, yerr = FitTPCPionNsigma.prepareData(pionEnhancedTPCnSigma, protonEnhancedTPCnSigma, kaonEnhancedTPCnSigma, inclusiveEnhancedTPCnSigma)
        # fit and extract the optimal fit params
        # start by creating the fitter instance
        # initialize the default parameters for the analysis type and current associated hadron momentum bin
        fitter.initializeDefaultParameters()
        optimal_params, covariance = fitter.performShapeFit(x, y, yerr)

        chi2OverNDF_shape = fitter.chi2OverNDF(optimal_params, covariance, x, y, yerr)
        # now set the region back to what it was before so we can fit the inclusive yield
        self.setRegion(currentRegion)
        # get the inclusive yield

        pionEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.PION)
        protonEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.PROTON)
        kaonEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.KAON)
        inclusiveEnhancedTPCnSigma = self.getEnhancedTPCnSigmaProjection(ParticleType.INCLUSIVE)

        x, y, yerr = FitTPCPionNsigma.prepareData(pionEnhancedTPCnSigma, protonEnhancedTPCnSigma, kaonEnhancedTPCnSigma, inclusiveEnhancedTPCnSigma)

        x_inc = x
        y_inc = y[3]
        yerr_inc = yerr[3]

        optimal_inclusive_yield_parameters, covariance_inclusive_yield, reducedChi2_yield  = fitter.performYieldFit(x_inc, y_inc, yerr_inc, optimal_params, covariance)

        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = optimal_params
        apinc, apiinc, akinc = optimal_inclusive_yield_parameters
        optimal_params = mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak
        covariance[-5:-2, -5:-2] = covariance_inclusive_yield # overwrite the covariance matrix with the covariance matrix from the inclusive yield fit

        # lets calculate a bootstrap estimeate of the systematic fit error
        # first lets get the number of bootstrap samples
        number_of_bootstrap_samples = 1000

        # now lets get the bootstrap samples
        param_value_samples = np.random.multivariate_normal(optimal_params, covariance, number_of_bootstrap_samples)
        # and generate the fit functions for each sample
        fit_functions = np.array([fitter.fittingFunction(None, x, *param_values) for param_values in param_value_samples]) # shape is (number_of_bootstrap_samples, 4*len(x))
        # now lets split it into the 4 species
        fit_functions = np.split(fit_functions, 4, axis=1) # shape is (4, number_of_bootstrap_samples, len(x))
        pion_fit_functions = fit_functions[0] # shape is (number_of_bootstrap_samples, len(x))
        proton_fit_functions = fit_functions[1] # shape is (number_of_bootstrap_samples, len(x))
        kaon_fit_functions = fit_functions[2] # shape is (number_of_bootstrap_samples, len(x))
        inclusive_fit_functions = fit_functions[3] # shape is (number_of_bootstrap_samples, len(x))

        # lets get teh correct fits too
        correct_fit_functions = fitter.fittingFunction(None, x, *optimal_params)
        correct_fit_functions = np.split(correct_fit_functions, 4) # shape is (4, len(x))
        pion_correct_fit_function = correct_fit_functions[0] # shape is (len(x),)
        proton_correct_fit_function = correct_fit_functions[1] # shape is (len(x),)
        kaon_correct_fit_function = correct_fit_functions[2] # shape is (len(x),)
        inclusive_correct_fit_function = correct_fit_functions[3] # shape is (len(x),)

        # now lets compute the mean deviation of the fit functions from the correct fit function
        pion_fit_function_mean_error = np.mean((pion_fit_functions-pion_correct_fit_function)**2, axis=0)
        proton_fit_function_mean_error = np.mean((proton_fit_functions-proton_correct_fit_function)**2, axis=0)
        kaon_fit_function_mean_error = np.mean((kaon_fit_functions-kaon_correct_fit_function)**2, axis=0)
        inclusive_fit_function_mean_error = np.mean((inclusive_fit_functions-inclusive_correct_fit_function)**2, axis=0) # shape is (len(x),)

        fit_err_band = np.concatenate((pion_fit_function_mean_error, proton_fit_function_mean_error, kaon_fit_function_mean_error, inclusive_fit_function_mean_error)) # shape is (4*len(x),)

        if makeIntermediatePlots:
            self.plotTPCPionNsigmaFit(x, y, yerr, fit_err_band, optimal_params, covariance, fitter.fittingFunction, fitter.fittingErrorFunction, fitter.pionFittingFunction, fitter.kaonFittingFunction, fitter.protonFittingFunction, fitter.chi2OverNDF, "TPCnSigmaFits")

        if  not hasattr(self, "numberOfAssociatedHadronsDictionary"):
            self.fillNumberOfAssociatedHadronsDictionary()
        
        
        # compute the PID fractions
        print("Computing PID fractions for momentum bin", self.currentAssociatedHadronMomentumBin.name, " and region ", self.currentRegion.name)
        pid_fractions, pid_fraction_errors, pid_fit_shape_sys_err, pid_fit_yield_sys_err = fitter.computeAveragePIDFractions(optimal_params, covariance, self.numberOfAssociatedHadronsBySpecies)

        return pid_fractions, pid_fraction_errors, pid_fit_shape_sys_err, pid_fit_yield_sys_err, chi2OverNDF_shape, reducedChi2_yield

    def getParticleFractionsFromDB(self):
        '''
        Gets the particle fractions from the database
        '''
        conn = sqlite3.connect("PID.db")
        dbCursor = conn.cursor()
        particle_fractions = {}
        particle_fraction_errors = {}
        pid_fit_shape_sys_err = {}
        pid_fit_yield_sys_err = {}
        for species in ParticleType:
            if species is ParticleType.OTHER or species is ParticleType.INCLUSIVE:
                continue

            particle_fractions[species], particle_fraction_errors[species], pid_fit_shape_sys_err[species], pid_fit_yield_sys_err[species] = getParticleFractionForMomentumBin(self.analysisType, self.currentRegion, self.currentAssociatedHadronMomentumBin, species, dbCursor)[0]
        conn.close()
        return particle_fractions, particle_fraction_errors, pid_fit_shape_sys_err, pid_fit_yield_sys_err

    def getEnhancedTPCnSigmaProjection(self, species: ParticleType):
        '''
        Sets the particle type for the jet hadron sparse and returns the projection onto the TPC nsigma axis, then resets the particle type
        '''
        self.setParticleSelectionForJetHadron(species)
        projection = self.getTPCPionNsigma()
        self.setParticleSelectionForJetHadron(self.current_species)
        return projection


    def getTPCPionNsigma(self):
        '''
        Returns the projection onto the TPC pion nsigma axis
        '''
        return self.JetHadron.getProjection(self.JetHadron.Axes.PION_TPC_N_SIGMA)
        

    def plotTPCPionNsigmaFit(self, x, y, yerr, yfit_err_band, optimal_params, covariance, fitFunction, fitErrorFunction, pionFitFunction, kaonFitFunction, protonFitFunction, chi2OverNDFFunction, save_path=None):

        if save_path is not None:
            if save_path[-1] != "/":
                save_path += "/"
            save_path = f"Plots/{self.analysisType.name}/{self.currentAssociatedHadronMomentumBin.name}/" + save_path

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            

        y_pion = y[0]
        y_proton = y[1]
        y_kaon = y[2]
        y_inclusive = y[3]

        y_err_pion = yerr[0]
        y_err_proton = yerr[1]
        y_err_kaon = yerr[2]
        y_err_inclusive = yerr[3]

        x_fit = np.linspace(x[0], x[-1], 100)
        y_fit = fitFunction(None, x_fit, *optimal_params)
        y_fit_err = fitErrorFunction(None, x_fit, *optimal_params, pcov=covariance)
        y_fit_pion = y_fit[:len(x_fit)]
        y_fit_proton = y_fit[len(x_fit):2*len(x_fit)]
        y_fit_kaon = y_fit[2*len(x_fit):3*len(x_fit)]
        y_fit_inclusive = y_fit[3*len(x_fit):]
        y_fit_err_pion = y_fit_err[:len(x_fit)]
        y_fit_err_proton = y_fit_err[len(x_fit):2*len(x_fit)]
        y_fit_err_kaon = y_fit_err[2*len(x_fit):3*len(x_fit)]
        y_fit_err_inclusive = y_fit_err[3*len(x_fit):]

        y_fit_err_band_pion = np.array(yfit_err_band[:len(x_fit)])
        y_fit_err_band_proton = np.array(yfit_err_band[len(x_fit):2*len(x_fit)])
        y_fit_err_band_kaon = np.array(yfit_err_band[2*len(x_fit):3*len(x_fit)])
        y_fit_err_band_inclusive = np.array(yfit_err_band[3*len(x_fit):])


        chi2OverNDF = chi2OverNDFFunction(optimal_params, covariance, x, y, yerr)

        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = optimal_params
        y_fit_pion_pion = pionFitFunction(x=x_fit, mu=mupi, sig=sigpi, a=apipi)
        y_fit_pion_proton = protonFitFunction(x_fit, mup, sigp, appi, alphap)
        y_fit_pion_kaon = kaonFitFunction(x_fit, muk, sigk, akpi, alphak)

        y_fit_proton_pion = pionFitFunction(x_fit, mupi, sigpi, apip)
        y_fit_proton_proton = protonFitFunction(x_fit, mup, sigp, app, alphap)
        y_fit_proton_kaon = kaonFitFunction(x_fit, muk, sigk, akp, alphak)

        y_fit_kaon_pion = pionFitFunction(x_fit, mupi, sigpi, apik)
        y_fit_kaon_proton = protonFitFunction(x_fit, mup, sigp, apk, alphap)
        y_fit_kaon_kaon = kaonFitFunction(x_fit, muk, sigk, akk, alphak)

        y_fit_inclusive_pion = pionFitFunction(x_fit, mupi, sigpi, apiinc)
        y_fit_inclusive_proton = protonFitFunction(x_fit, mup, sigp, apinc, alphap)
        y_fit_inclusive_kaon = kaonFitFunction(x_fit, muk, sigk, akinc, alphak)

        # lets generate a bootstrap error for the components of each fit 

        # first lets get the number of bootstrap samples
        number_of_bootstrap_samples = 1000

        # now lets get the bootstrap samples
        param_value_samples = np.random.multivariate_normal(optimal_params, covariance, number_of_bootstrap_samples) # shape is (number_of_bootstrap_samples, len(optimal_params))
        pion_in_pion_param_indices = [1, 4, 10]
        proton_in_pion_param_indices = [0, 3, 9, 18]
        kaon_in_pion_param_indices = [2, 5, 11, 19]
        pion_in_proton_param_indices = [1, 4, 7]
        proton_in_proton_param_indices = [0, 3, 6, 18]
        kaon_in_proton_param_indices = [2, 5, 8, 19]
        pion_in_kaon_param_indices = [1, 4, 13]
        proton_in_kaon_param_indices = [0, 3, 12, 18]
        kaon_in_kaon_param_indices = [2, 5, 14, 19]
        pion_in_inclusive_param_indices = [1, 4, 16]
        proton_in_inclusive_param_indices = [0, 3, 15, 18]
        kaon_in_inclusive_param_indices = [2, 5, 17, 19]

        # and generate the fit functions for each sample
        pion_in_pion_fit_functions = np.array([pionFitFunction(x_fit, param_values[pion_in_pion_param_indices[0]], param_values[pion_in_pion_param_indices[1]], param_values[pion_in_pion_param_indices[2]]) for param_values in param_value_samples])
        proton_in_pion_fit_functions = np.array([protonFitFunction(x_fit, param_values[proton_in_pion_param_indices[0]], param_values[proton_in_pion_param_indices[1]], param_values[proton_in_pion_param_indices[2]], param_values[proton_in_pion_param_indices[3]]) for param_values in param_value_samples])
        kaon_in_pion_fit_functions = np.array([kaonFitFunction(x_fit, param_values[kaon_in_pion_param_indices[0]], param_values[kaon_in_pion_param_indices[1]], param_values[kaon_in_pion_param_indices[2]], param_values[kaon_in_pion_param_indices[3]]) for param_values in param_value_samples])

        pion_in_proton_fit_functions = np.array([pionFitFunction(x_fit, param_values[pion_in_proton_param_indices[0]], param_values[pion_in_proton_param_indices[1]], param_values[pion_in_proton_param_indices[2]]) for param_values in param_value_samples])
        proton_in_proton_fit_functions = np.array([protonFitFunction(x_fit, param_values[proton_in_proton_param_indices[0]], param_values[proton_in_proton_param_indices[1]], param_values[proton_in_proton_param_indices[2]], param_values[proton_in_proton_param_indices[3]]) for param_values in param_value_samples])
        kaon_in_proton_fit_functions = np.array([kaonFitFunction(x_fit, param_values[kaon_in_proton_param_indices[0]], param_values[kaon_in_proton_param_indices[1]], param_values[kaon_in_proton_param_indices[2]], param_values[kaon_in_proton_param_indices[3]]) for param_values in param_value_samples])

        pion_in_kaon_fit_functions = np.array([pionFitFunction(x_fit, param_values[pion_in_kaon_param_indices[0]], param_values[pion_in_kaon_param_indices[1]], param_values[pion_in_kaon_param_indices[2]]) for param_values in param_value_samples])
        proton_in_kaon_fit_functions = np.array([protonFitFunction(x_fit, param_values[proton_in_kaon_param_indices[0]], param_values[proton_in_kaon_param_indices[1]], param_values[proton_in_kaon_param_indices[2]], param_values[proton_in_kaon_param_indices[3]]) for param_values in param_value_samples])
        kaon_in_kaon_fit_functions = np.array([kaonFitFunction(x_fit, param_values[kaon_in_kaon_param_indices[0]], param_values[kaon_in_kaon_param_indices[1]], param_values[kaon_in_kaon_param_indices[2]], param_values[kaon_in_kaon_param_indices[3]]) for param_values in param_value_samples])

        pion_in_inclusive_fit_functions = np.array([pionFitFunction(x_fit, param_values[pion_in_inclusive_param_indices[0]], param_values[pion_in_inclusive_param_indices[1]], param_values[pion_in_inclusive_param_indices[2]]) for param_values in param_value_samples])
        proton_in_inclusive_fit_functions = np.array([protonFitFunction(x_fit, param_values[proton_in_inclusive_param_indices[0]], param_values[proton_in_inclusive_param_indices[1]], param_values[proton_in_inclusive_param_indices[2]], param_values[proton_in_inclusive_param_indices[3]]) for param_values in param_value_samples])
        kaon_in_inclusive_fit_functions = np.array([kaonFitFunction(x_fit, param_values[kaon_in_inclusive_param_indices[0]], param_values[kaon_in_inclusive_param_indices[1]], param_values[kaon_in_inclusive_param_indices[2]], param_values[kaon_in_inclusive_param_indices[3]]) for param_values in param_value_samples])

        # now lets compute the mean deviation of the fit functions from the correct fit function
        pion_in_pion_fit_function_mean_error = np.mean((pion_in_pion_fit_functions-y_fit_pion_pion)**2, axis=0)
        proton_in_pion_fit_function_mean_error = np.mean((proton_in_pion_fit_functions-y_fit_pion_proton)**2, axis=0)
        kaon_in_pion_fit_function_mean_error = np.mean((kaon_in_pion_fit_functions-y_fit_pion_kaon)**2, axis=0)
        pion_in_proton_fit_function_mean_error = np.mean((pion_in_proton_fit_functions-y_fit_proton_pion)**2, axis=0)
        proton_in_proton_fit_function_mean_error = np.mean((proton_in_proton_fit_functions-y_fit_proton_proton)**2, axis=0)
        kaon_in_proton_fit_function_mean_error = np.mean((kaon_in_proton_fit_functions-y_fit_proton_kaon)**2, axis=0)
        pion_in_kaon_fit_function_mean_error = np.mean((pion_in_kaon_fit_functions-y_fit_kaon_pion)**2, axis=0)
        proton_in_kaon_fit_function_mean_error = np.mean((proton_in_kaon_fit_functions-y_fit_kaon_proton)**2, axis=0)
        kaon_in_kaon_fit_function_mean_error = np.mean((kaon_in_kaon_fit_functions-y_fit_kaon_kaon)**2, axis=0)
        pion_in_inclusive_fit_function_mean_error = np.mean((pion_in_inclusive_fit_functions-y_fit_inclusive_pion)**2, axis=0)
        proton_in_inclusive_fit_function_mean_error = np.mean((proton_in_inclusive_fit_functions-y_fit_inclusive_proton)**2, axis=0)
        kaon_in_inclusive_fit_function_mean_error = np.mean((kaon_in_inclusive_fit_functions-y_fit_inclusive_kaon)**2, axis=0)

        


        x_data_pion = {
            "raw_pion": x,
            "pion_total": x_fit,
            "pion_pion": x_fit,
            "pion_proton": x_fit,
            "pion_kaon": x_fit,
        }
        x_data_proton = {
            "raw_proton": x,
            "proton_total": x_fit,
            "proton_pion": x_fit,
            "proton_proton": x_fit,
            "proton_kaon": x_fit,
        }
        x_data_kaon = {
            "raw_kaon": x,
            "kaon_total": x_fit,
            "kaon_pion": x_fit,
            "kaon_proton": x_fit,
            "kaon_kaon": x_fit,
        }
        x_data_inclusive = {
            "raw_inclusive": x,
            "inclusive_total": x_fit,
            "inclusive_pion": x_fit,
            "inclusive_proton": x_fit,
            "inclusive_kaon": x_fit,
        }
        y_data_pion = {
            "raw_pion": y_pion,
            "pion_total": y_fit_pion,
            "pion_pion": y_fit_pion_pion,
            "pion_proton": y_fit_pion_proton,
            "pion_kaon": y_fit_pion_kaon,
        }
        y_data_proton = {
            "raw_proton": y_proton,
            "proton_total": y_fit_proton,
            "proton_pion": y_fit_proton_pion,
            "proton_proton": y_fit_proton_proton,
            "proton_kaon": y_fit_proton_kaon,
        }
        y_data_kaon = {
            "raw_kaon": y_kaon,
            "kaon_total": y_fit_kaon,
            "kaon_pion": y_fit_kaon_pion,
            "kaon_proton": y_fit_kaon_proton,
            "kaon_kaon": y_fit_kaon_kaon,
        }
        y_data_inclusive = {
            "raw_inclusive": y_inclusive,
            "inclusive_total": y_fit_inclusive,
            "inclusive_pion": y_fit_inclusive_pion,
            "inclusive_proton": y_fit_inclusive_proton,
            "inclusive_kaon": y_fit_inclusive_kaon,
        }

        yerr_data_pion = {
            "raw_pion": y_err_pion,
            "pion_total": y_fit_err_pion,
            "pion_pion": None,
            "pion_proton": None,
            "pion_kaon": None,
        }
        yerr_data_proton = {
            "raw_proton": y_err_proton,
            "proton_total": y_fit_err_proton,
            "proton_pion": None,
            "proton_proton": None,
            "proton_kaon": None,
        }
        yerr_data_kaon = {
            "raw_kaon": y_err_kaon,
            "kaon_total": y_fit_err_kaon,
            "kaon_pion": None,
            "kaon_proton": None,
            "kaon_kaon": None,
        }
        yerr_data_inclusive = {
            "raw_inclusive": y_err_inclusive,
            "inclusive_total": y_fit_err_inclusive,
            "inclusive_pion": None,
            "inclusive_proton": None,
            "inclusive_kaon": None,
        }

        yerr_band_data_pion = {
            "raw_pion": None,
            "pion_total": y_fit_err_band_pion,
            "pion_pion": pion_in_pion_fit_function_mean_error,
            "pion_proton": proton_in_pion_fit_function_mean_error,
            "pion_kaon": kaon_in_pion_fit_function_mean_error,
        }

        yerr_band_data_proton = {
            "raw_proton": None,
            "proton_total": y_fit_err_band_proton,
            "proton_pion": pion_in_proton_fit_function_mean_error,
            "proton_proton": proton_in_proton_fit_function_mean_error,
            "proton_kaon": kaon_in_proton_fit_function_mean_error,
        }

        yerr_band_data_kaon = {
            "raw_kaon": None,
            "kaon_total": y_fit_err_band_kaon,
            "kaon_pion": pion_in_kaon_fit_function_mean_error,
            "kaon_proton": proton_in_kaon_fit_function_mean_error,
            "kaon_kaon": kaon_in_kaon_fit_function_mean_error,
        }

        yerr_band_data_inclusive = {
            "raw_inclusive": None,
            "inclusive_total": y_fit_err_band_inclusive,
            "inclusive_pion": pion_in_inclusive_fit_function_mean_error,
            "inclusive_proton": proton_in_inclusive_fit_function_mean_error,
            "inclusive_kaon": kaon_in_inclusive_fit_function_mean_error,
        }
        
        data_labels_pion = {
            "raw_pion": "Enhanced Pion",
            "pion_total": "Pion Fit",
            "pion_pion": "Pion Component",
            "pion_proton": "Proton Component",
            "pion_kaon": "Kaon Component",
        }
        data_labels_proton = {
            "raw_proton": "Enhanced Proton",
            "proton_total": "Proton Fit",
            "proton_pion": "Pion Component",
            "proton_proton": "Proton Component",
            "proton_kaon": "Kaon Component",
        }
        data_labels_kaon = {
            "raw_kaon": "Enhanced Kaon",
            "kaon_total": "Kaon Fit",
            "kaon_pion": "Pion Component",
            "kaon_proton": "Proton Component",
            "kaon_kaon": "Kaon Component",
        }
        data_labels_inclusive = {
            "raw_inclusive": "Enhanced Inclusive",
            "inclusive_total": "Inclusive Fit",
            "inclusive_pion": "Pion Component",
            "inclusive_proton": "Proton Component",
            "inclusive_kaon": "Kaon Component",
        }

        format_style_pion = {
            "raw_pion": "bo",
            "pion_total": "b-",
            "pion_pion": "b--",
            "pion_proton": "r--",
            "pion_kaon": "g--",
        }
        format_style_proton = {
            "raw_proton": "ro",
            "proton_total": "r-",
            "proton_pion": "b--",
            "proton_proton": "r--",
            "proton_kaon": "g--",
        }
        format_style_kaon = {
            "raw_kaon": "go",
            "kaon_total": "g-",
            "kaon_pion": "b--",
            "kaon_proton": "r--",
            "kaon_kaon": "g--",
        }
        format_style_inclusive = {
            "raw_inclusive": "ko",
            "inclusive_total": "k-",
            "inclusive_pion": "b--",
            "inclusive_proton": "r--",
            "inclusive_kaon": "g--",
        }

        plotArrays(x_data_pion, y_data_pion, yerr_data_pion, data_label=data_labels_pion, format_style=format_style_pion, error_bands=yerr_band_data_pion, error_bands_label=None, title=f"{self.currentRegion.name}, {self.analysisType.name}, p_T^{{assoc.}} {self.currentAssociatedHadronMomentumBin.name}, Pions Chi^2/NDF = {chi2OverNDF:.2f}", xtitle="TPC nSigma", ytitle="Density", output_path=f"{save_path}TPCnSigmaFit_{self.currentRegion}_Pion.png")
        plotArrays(x_data_proton, y_data_proton, yerr_data_proton, data_label=data_labels_proton, format_style=format_style_proton, error_bands=yerr_band_data_proton, error_bands_label=None, title=f"{self.currentRegion}, {self.analysisType.name}, p_T^{{assoc.}} {self.currentAssociatedHadronMomentumBin.name}, Protons Chi^2/NDF = {chi2OverNDF:.2f}", xtitle="TPC nSigma", ytitle="Density", output_path=f"{save_path}TPCnSigmaFit_{self.currentRegion}_Proton.png")
        plotArrays(x_data_kaon, y_data_kaon, yerr_data_kaon, data_label=data_labels_kaon, format_style=format_style_kaon, error_bands=yerr_band_data_kaon, error_bands_label=None, title=f"{self.currentRegion}, {self.analysisType.name}, p_T^{{assoc.}} {self.currentAssociatedHadronMomentumBin.name}, Kaons Chi^2/NDF = {chi2OverNDF:.2f}", xtitle="TPC nSigma", ytitle="Density", output_path=f"{save_path}TPCnSigmaFit_{self.currentRegion}_Kaon.png")
        plotArrays(x_data_inclusive, y_data_inclusive, yerr_data_inclusive, data_label=data_labels_inclusive, format_style=format_style_inclusive, error_bands=yerr_band_data_inclusive, error_bands_label=None, title=f"{self.currentRegion}, {self.analysisType.name}, p_T^{{assoc.}} {self.currentAssociatedHadronMomentumBin.name}, Inclusive Chi^2/NDF = {chi2OverNDF:.2f}", xtitle="TPC nSigma", ytitle="Density", output_path=f"{save_path}TPCnSigmaFit_{self.currentRegion}_Inclusive.png")

        # y_fit_pi = pionFitFunction(None, x_fit, *optimal_params[:3])
        # y_fit_k = kaonFitFunction(None, x_fit, *optimal_params[3:6])
        # y_fit_p = protonFitFunction(None, x_fit, *optimal_params[6:9])

    def computeMixedEventNormalizationFactor(self, mixedEventCorrelationFunction, normMethod: NormalizationMethod, **kwargs):
        '''
        Returns the normalization factor for the mixed event correlation function
        '''
        if normMethod == NormalizationMethod.SLIDING_WINDOW:
            return self.computeSlidingWindowNormalizationFactor(mixedEventCorrelationFunction=mixedEventCorrelationFunction, windowSize=kwargs.get("windowSize", pi))
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
        self.numberOfAssociatedHadronsBySpecies = {}
        for species in ParticleType:
            self.setParticleSelectionForJetHadron(species)
            self.numberOfAssociatedHadronsBySpecies[species] = self.getNumberOfAssociatedParticles()
        self.setParticleSelectionForJetHadron(self.current_species)

    def getNumberOfAssociatedParticles(self):
        '''
        Returns the number of associated particles
        '''
        if repr(self.JetHadron) not in self.numberOfAssociatedHadrons:
            self.numberOfAssociatedHadrons[repr(self.JetHadron)] = self.JetHadron.getNumberOfAssociatedParticles()
        return self.numberOfAssociatedHadrons[repr(self.JetHadron)]

    def getNumberOfTriggerJets(self):
        '''
        Returns the number of trigger jets
        '''
        if repr(self.Trigger) not in self.numberOfTriggerJets:
            self.numberOfTriggerJets[repr(self.Trigger)] = self.Trigger.getNumberOfTriggerJets()
        return self.numberOfTriggerJets[repr(self.Trigger)]



