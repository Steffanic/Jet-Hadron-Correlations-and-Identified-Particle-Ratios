from uncertainties.umath import erf, fsum
from pyunfold import iterative_unfold
import uncertainties
from uncertainties import unumpy as unp
from ROOT import TH1D, TH2D, TH3D
import numpy as np
from sklearn.preprocessing import normalize
from scipy.integrate import quad
from functools import partial
import logging
debug_logger = logging.getLogger('debug')
error_logger = logging.getLogger('error')
info_logger = logging.getLogger('info')
unfolding_logger = logging.getLogger('unfolding')
# let's register a new file for the unfolding logger
unfolding_logger.addHandler(logging.FileHandler('unfolding.log', mode='w'))
unfolding_logger.setLevel(logging.INFO)



def area_under_gaussian_distribution(mean, sigma, x_min, x_max):
  """
  Calculates the area under a Gaussian distribution with custom mean and sigma.

  Args:
    mean: The mean of the Gaussian distribution.
    sigma: The standard deviation of the Gaussian distribution.
    x_min: The lower bound of the integration region.
    x_max: The upper bound of the integration region.

  Returns:
    The area under the Gaussian distribution.
  """

  # Calculate the probability density function of the Gaussian distribution.
  pdf = lambda x: np.exp(-(x - mean)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

  # Calculate the integral of the probability density function.
  integral = np.trapz(pdf, x_min, x_max)

  return integral

def print_function_name_with_description_on_call(description, logging_level=logging.DEBUG):
    """
    Prints the name of the function and a description of what it does
    """

    def function_wrapper(function):
        def method_wrapper(self, *args, **kwargs):
            if logging_level == logging.DEBUG:
                logger = debug_logger
            elif logging_level == logging.INFO:
                logger = info_logger
            elif logging_level == logging.ERROR:
                logger = error_logger
            else:
                raise ValueError(f"Unknown logging level {logging_level}")
            logger.log(level=logging_level, msg=f"{function.__name__} in {self.__class__.__name__}:\n\t{description}")
            return function(self, *args, **kwargs)

        return method_wrapper

    return function_wrapper


class AnalysisMixin:
    def get_SE_correlation_function(self):
        """
        Returns TH2 containing the △eta, △phi distribution from the JH sparse
        or TH3 containing the △eta, △phi, △pion distribution from the JH sparse
        """
    
        TH2Corr = None
        for sparse_ind in range(len(self.JH)):  # type:ignore
            bin_widths = [ 
                    self.JH[sparse_ind].GetAxis(i).GetBinWidth(1)
                    for i in range(1, self.JH[sparse_ind].GetNdimensions())
                ]
            if TH2Corr is None:
                TH2Corr = self.JH[sparse_ind].Projection(4, 3)  # type:ignore
                # divide by the bin widths to get the correct correlation function
            else:
                success = TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore
                # divide by the bin widths to get the correct correlation function
                if not success:
                    warn("Failed to add to TH2Corr")
        TH2Corr.Scale(1 / bin_widths[2] / bin_widths[3])
        self.get_SE_correlation_function_result = TH2Corr
        self.get_SE_correlation_function_has_changed = False
        return TH2Corr.Clone()

    def get_SE_correlation_function_for_species(self, i,j,k,species):
        """
        Returns TH2 containing the △eta, △phi distribution from the JH sparse
        or TH3 containing the △eta, △phi, △pion distribution from the JH sparse
        """
        offset = 1 if self.analysisType == "pp" else 0  # type:ignore
        speciesID = (
            9
            if species == "pion"
            else 10
            if species == "proton"
            else 11
        )

        TH2Corr = None
        for sparse_ind in range(len(self.JH)):  # type:ignore
            bin_widths = [ 
                    self.JH[sparse_ind].GetAxis(i).GetBinWidth(1)
                    for i in range(1, self.JH[sparse_ind].GetNdimensions())
                ]

            if species is not "other":
                if species=="pion":

                    self.JH[sparse_ind].GetAxis(9 - offset).SetRangeUser(
                        -2 , 2
                    )  # type:ignore
                    self.JH[sparse_ind].GetAxis(11 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(11-offset).FindBin(-2 if j<2 else -1))
                    self.JH[sparse_ind].GetAxis(10 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2))
                    if TH2Corr is None:
                        TH2Corr = self.JH[sparse_ind].Projection(4, 3)  # type:ignore
                        
                    else:
                        TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore
                    # self.JH[sparse_ind].GetAxis(11-offset).SetRange(0,0)
                    # TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore

                elif species=="kaon":

                    self.JH[sparse_ind].GetAxis(11 - offset).SetRangeUser(
                        -2 if j<2 else -1 , 2
                    )  # type:ignore
                    self.JH[sparse_ind].GetAxis(10 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2))
                    if TH2Corr is None:
                        TH2Corr = self.JH[sparse_ind].Projection(4, 3)  # type:ignore
                        
                    else:
                        TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore
                    # self.JH[sparse_ind].GetAxis(10-offset).SetRange(0,0)
                    # TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore

                elif species=="proton":

                    self.JH[sparse_ind].GetAxis(10 - offset).SetRangeUser(
                        -2 , 2
                    )
                    if TH2Corr is None:
                        TH2Corr = self.JH[sparse_ind].Projection(4, 3)  # type:ignore
                        
                    else:
                        TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore



                
                self.JH[sparse_ind].GetAxis(9 - offset).SetRange(
                  0,0
                )  # type:ignore
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                  0,0
                )  # type:ignore
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                  0,0
                )  # type:ignore
            else:
                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                     self.JH[sparse_ind].GetAxis(10 - offset).FindBin(2), self.JH[sparse_ind].GetAxis(10 - offset).GetNbins()+1
                )  # type:ignore
                if TH2Corr is None:
                    TH2Corr = self.JH[sparse_ind].Projection(4, 3)  # type:ignore
                else:
                    TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore

                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                    0, self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2)
                )  # type:ignore
                # k
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                     self.JH[sparse_ind].GetAxis(11 - offset).FindBin(2), self.JH[sparse_ind].GetAxis(11 - offset).GetNbins()+1
                )  # type:ignore
                TH2Corr.Add(self.JH[sparse_ind].Projection(4,3))

                # pi
                self.JH[sparse_ind].GetAxis(9 - offset).SetRange(
                    0, self.JH[sparse_ind].GetAxis(9-offset).FindBin(-2)
                )  # type:ignore
                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                     0,self.JH[sparse_ind].GetAxis(10 - offset).FindBin(-2)
                )  # type:ignore
                # k
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                     0, self.JH[sparse_ind].GetAxis(11 - offset).FindBin(-2)
                )  # type:ignore
                TH2Corr.Add(self.JH[sparse_ind].Projection(4,3))

                # pi
                self.JH[sparse_ind].GetAxis(9 - offset).SetRange(
                     self.JH[sparse_ind].GetAxis(9-offset).FindBin(2), self.JH[sparse_ind].GetAxis(9-offset).GetNbins()+1
                )  # type:ignore
                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                     0, self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2)
                )  # type:ignore
                # k
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                     0, self.JH[sparse_ind].GetAxis(11 - offset).FindBin(-2)
                )  # type:ignore
                TH2Corr.Add(self.JH[sparse_ind].Projection(4,3))
                

                self.JH[sparse_ind].GetAxis(9 - offset).SetRange(
                  0,0
                )  # type:ignore
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                  0,0
                )  # type:ignore
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                  0,0
                )  # type:ignore
        TH2Corr.Scale(1 / bin_widths[2] / bin_widths[3])
        return TH2Corr.Clone()

    def get_SE_correlation_function_w_pionTPCnSigma(self, species):
        """
        Returns TH2 containing the △eta, △phi distribution from the JH sparse
        or TH3 containing the △eta, △phi, △pion distribution from the JH sparse
        """
        offset = 1 if self.analysisType == "pp" else 0  # type:ignore
    
        speciesID = (
            9
            if species == "pion"
            else 10
            if species == "proton"
            else 11
        )

        TH3Corr = None
        for sparse_ind in range(len(self.JH)):  # type:ignore
            self.JH[sparse_ind].GetAxis(speciesID - offset).SetRangeUser(
                -2 if self.analysisType=='pp' else -2, 2 if self.analysisType=='pp' else 2
            )  # type:ignore
            if TH3Corr is None:
                TH3Corr = self.JH[sparse_ind].Projection(
                    3, 4, 7 - offset
                )  # type:ignore
            else:
                TH3Corr.Add(
                    self.JH[sparse_ind].Projection(3, 4, 7 - offset)
                )  # type:ignore
            self.JH[sparse_ind].GetAxis(speciesID - offset).SetRangeUser(
                -10, 10
            )  # type:ignore
        self.get_SE_correlation_function_w_Pion_result[
            species
        ] = TH3Corr  # type:ignore
        self.get_SE_correlation_function_w_Pion_has_changed[
            species
        ] = False  # type:ignore
        return TH3Corr.Clone()
        

    def get_ME_correlation_function(self):
        """
        Rrturns TH2 containing the △eta, △phi distribution from the Mixed Event sparse
        """
        TH2Corr = None
        for sparse_ind in range(len(self.MixedEvent)):  # type:ignore
            if TH2Corr is None:
                TH2Corr = self.MixedEvent[sparse_ind].Projection(4, 3)
            else:
                TH2Corr.Add(self.MixedEvent[sparse_ind].Projection(4, 3))
        bin_widths = [
            self.MixedEvent[0].GetAxis(i).GetBinWidth(1)
            for i in range(1, self.MixedEvent[0].GetNdimensions())
        ]
        TH2Corr.Scale(1 / bin_widths[2] / bin_widths[3])
        self.get_ME_correlation_function_result = TH2Corr
        self.get_ME_correlation_function_has_changed = False
        return TH2Corr.Clone()


    def get_normalized_ME_correlation_function(self):
        norm, error = self.get_ME_norm_and_systematic_error()
        a0 = 1 / norm
        NormMEC = self.get_ME_correlation_function()
        NormMEC.Scale(a0)
        self.get_normalized_ME_correlation_function_result = NormMEC
        self.get_normalized_ME_correlation_function_has_changed = False
        self.get_ME_normalization_error = error
        self.get_ME_normalization_error_has_changed = False
        return NormMEC.Clone(), error
    

    def get_ME_norm_and_systematic_error(self):
        # compute the normalization constant for the mixed event correlation function
        # using a sliding window average with window size pi
        sliding_window_norm, error = self.ME_norm_sliding_window()
        # using a sliding window average with window size pi/2
        #sliding_window_norm_half_pi = self.ME_norm_sliding_window(windowSize=np.pi / 2)
        # using a sliding window average with window size pi/4
        #sliding_window_norm_quarter_pi = self.ME_norm_sliding_window(windowSize=np.pi/4)
        # using a sliding window average with window size pi/6
        #sliding_window_norm_sixth_pi = self.ME_norm_sliding_window(windowSize=np.pi/6)
        # using a sliding window average with window size pi/12
        #sliding_window_norm_twelfth_pi = self.ME_norm_sliding_window(windowSize=np.pi/12)
        # by taking the max
        #max_norm = self.ME_max()
        #print(f"{sliding_window_norm=}, {max_norm=}, {sliding_window_norm_half_pi}")
        # take the average of the three
        norm = sliding_window_norm
        # compute the error on the normalization constant
        # using the standard deviation of the three values
        #error = np.abs(max_norm-sliding_window_norm)
        return norm, error

    def ME_max(self, etaRestriction=0.3):
        """
        Returns the maximum value of the mixed event correlation function
        """
        fhnMixedEventsCorr = self.get_ME_correlation_function()  # TH2
        fhnMixedEventsCorr.GetXaxis().SetRangeUser(-etaRestriction, etaRestriction)
        eta_bin_width = fhnMixedEventsCorr.GetXaxis().GetBinWidth(1)
        eta_bins = fhnMixedEventsCorr.GetXaxis().FindBin(
            etaRestriction
        ) - fhnMixedEventsCorr.GetXaxis().FindBin(-etaRestriction)
        dPhiME = fhnMixedEventsCorr.ProjectionY()
        dPhiME.Scale(1/eta_bins)
        return dPhiME.GetMaximum()
    
    def ME_norm_sliding_window(self, windowSize=np.pi, etaRestriction=0.3):
        """
        Returns normalization constant for mixed event correlation function

        Restricts |△eta|<etaRestriction and projects onto △phi
        Using a sliding window average, find the highest average and call it the max
        """
        fhnMixedEventsCorr = self.get_ME_correlation_function()  # TH2
        fhnMixedEventsCorr.GetXaxis().SetRangeUser(-etaRestriction, etaRestriction)
        # get  eta bin width
        eta_bin_width = fhnMixedEventsCorr.GetXaxis().GetBinWidth(1)
        # get number of bins in eta between +- etaRestriction
        eta_bins = fhnMixedEventsCorr.GetXaxis().FindBin(
            etaRestriction
        ) - fhnMixedEventsCorr.GetXaxis().FindBin(-etaRestriction)

        dPhiME = fhnMixedEventsCorr.ProjectionY()
        # scale by the number of bins in eta
        dPhiME.Scale(1/eta_bins)
        # Calculate moving max of the mixed event correlation function with window size pi
        maxWindowAve = 0
        totalError = None
        # get bin width
        binWidth = dPhiME.GetBinWidth(1)
        # get number of bins
        nBinsPerWindow = int(windowSize // binWidth)
        for i in range(dPhiME.GetNbinsX() - int(windowSize // binWidth)):
            # get the average of the window for this set of bins
            windowAve = dPhiME.Integral(i, i + int(windowSize // binWidth)) / (
                nBinsPerWindow +1
            )
            # if the average is greater than the current max, set the max to the average
            if windowAve > maxWindowAve:
                maxWindowAve = windowAve
                totalError = np.sqrt(np.sum([dPhiME.GetBinError(bin_no)**2 for bin_no in range(i, i + int(windowSize // binWidth)+1)]))
                

        del dPhiME
        del fhnMixedEventsCorr

        return maxWindowAve, totalError

    def get_acceptance_corrected_correlation_function(self, in_z_vertex_bins=False):

        if in_z_vertex_bins:
            # set the user range for JH, ME, and Trigger sparses to be the same as the z vertex bins
            AccCorr = self.get_SE_correlation_function().Clone()
            AccCorr.Reset()
            total_norm=0
            z_bin_width = self.z_vertex_bins[1]-self.z_vertex_bins[0]
            for z_bin_no in range(len(self.z_vertex_bins)-1):
                self.set_z_vertex_bin(z_bin_no)
                if self.get_ME_correlation_function().Integral() == 0:
                    debug_logger.info(f"Skipping z bin {z_bin_no}")
                    continue
                Corr = self.get_SE_correlation_function()
                NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
                Corr.Divide(NormMEC)
                total_norm +=Corr.GetEntries()
                Corr.Scale(Corr.GetEntries())
                AccCorr.Add(Corr)
            AccCorr.Scale(z_bin_width/total_norm)
            self.reset_z_vertex_bin()
            return AccCorr.Clone()

        else:
            # get the raw correlation functions
            Corr = self.get_SE_correlation_function()
            # get normalized mixed event correlation function
            NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
            # divide the raw correlation function by the mixed event correlation function
            Corr.Divide(NormMEC)
            # Corr.GetXaxis().SetRangeUser(-0.9, 0.9)
            del NormMEC
            self.get_acceptance_corrected_correlation_function_result = Corr
            self.get_acceptance_corrected_correlation_function_has_changed = False
            return Corr.Clone()

    def get_acceptance_corrected_correlation_function_for_enhanced_species(self, i,j,k,species, in_z_vertex_bins=False):
        if in_z_vertex_bins:
            # set the user range for JH, ME, and Trigger sparses to be the same as the z vertex bins
            EnhCorr = self.get_SE_correlation_function_for_species(i,j,k,species).Clone()
           
            EnhCorr.Reset()
            
            total_norm=0

            z_bin_width = self.z_vertex_bins[1]-self.z_vertex_bins[0]
            for z_bin_no in range(len(self.z_vertex_bins)-1):
                self.set_z_vertex_bin(z_bin_no)
                if self.get_ME_correlation_function().Integral() == 0:
                    debug_logger.info(f"Skipping z bin {z_bin_no}")
                    continue
                Corrs_raw = self.get_SE_correlation_function_for_species(i,j,k,species) 
                

                NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
                Corrs_raw.Divide(NormMEC)
       
                total_norm+=1


                EnhCorr.Add(Corrs_raw)
            EnhCorr.Scale(z_bin_width/total_norm)

            self.reset_z_vertex_bin()

        else:
            # get the raw correlation functions for every species and use the PionTPCNSigmaFitObjs to get the mixing factors
            Corrs_raw =self.get_SE_correlation_function_for_species(i,j,k,species)
            
            EnhCorr = Corrs_raw.Clone()
            
            
            # get normalized mixed event correlation function
            NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
            # divide the raw correlation function by the mixed event correlation function
            EnhCorr.Divide(NormMEC)

        return EnhCorr.Clone()

    def get_acceptance_corrected_correlation_function_for_true_species(self, i,j, k,species, in_z_vertex_bins=False):

        if in_z_vertex_bins:
            # set the user range for JH, ME, and Trigger sparses to be the same as the z vertex bins
            Corrs_raw = [self.get_SE_correlation_function_for_species(i,j,k,enh_species) for enh_species in ["pion", "proton", "kaon", "other"]]
            [C.Reset() for C in Corrs_raw]
            pionEnhCorr = self.get_SE_correlation_function_for_species(i,j,k,species).Clone()
            kaonEnhCorr = self.get_SE_correlation_function_for_species(i,j,k,species).Clone()
            protonEnhCorr = self.get_SE_correlation_function_for_species(i,j,k,species).Clone()
            otherEnhCorr = self.get_SE_correlation_function_for_species(i,j,k,species).Clone()
            pionEnhCorr.Reset()
            kaonEnhCorr.Reset()
            protonEnhCorr.Reset()
            otherEnhCorr.Reset()
            total_norm_pion=0
            total_norm_proton=0
            total_norm_kaon=0
            total_norm_other=0
            z_bin_width = self.z_vertex_bins[1]-self.z_vertex_bins[0]
            for z_bin_no in range(len(self.z_vertex_bins)-1):
                self.set_z_vertex_bin(z_bin_no)
                if self.get_ME_correlation_function().Integral() == 0:
                    debug_logger.info(f"Skipping z bin {z_bin_no}")
                    continue
                [C.Add(self.get_SE_correlation_function_for_species(i,j,k,enh_species)) for C, enh_species in zip(Corrs_raw, ["pion", "proton", "kaon", "other"])]
                Corrs_rawZV = [self.get_SE_correlation_function_for_species(i,j,k,enh_species) for enh_species in ["pion", "proton", "kaon", "other"]]
                piEnhCorr = Corrs_rawZV[0]
                pEnhCorr = Corrs_rawZV[1]
                kEnhCorr = Corrs_rawZV[2]
                oEnhCorr = Corrs_rawZV[3]
   

                NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
                
                piEnhCorr.Divide(NormMEC)
                pEnhCorr.Divide(NormMEC)
                kEnhCorr.Divide(NormMEC)
                oEnhCorr.Divide(NormMEC)
                total_norm_pion+=1
                total_norm_proton+=1
                total_norm_kaon+=1
                total_norm_other+=1

                pionEnhCorr.Add(piEnhCorr)
                kaonEnhCorr.Add(kEnhCorr)
                protonEnhCorr.Add(pEnhCorr)
                otherEnhCorr.Add(oEnhCorr)
            pionEnhCorr.Scale(z_bin_width/total_norm_pion)
            kaonEnhCorr.Scale(z_bin_width/total_norm_proton)
            protonEnhCorr.Scale(z_bin_width/total_norm_kaon)
            otherEnhCorr.Scale(z_bin_width/total_norm_other)
            [C.Scale(z_bin_width/total_norm_pion) for C in Corrs_raw]
            self.reset_z_vertex_bin()

        else:
            # get the raw correlation functions for every species and use the PionTPCNSigmaFitObjs to get the mixing factors
            Corrs_raw = [self.get_SE_correlation_function_for_species(i,j,k,enh_species) for enh_species in ["pion", "proton", "kaon", "other"]]

            # if self.analysisType in ['central', 'semicentral']:
            #     assert(self.N_assoc_for_species[species][i,j,k]!=0, f"{self.N_assoc_for_species['pion'][i,j,k]=}")
            #     assert(self.N_assoc_for_species[species][i,j,k]!=0, f"{self.N_assoc_for_species['proton'][i,j,k]=}")
            #     assert(self.N_assoc_for_species[species][i,j,k]!=0, f"{self.N_assoc_for_species['kaon'][i,j,k]=}")
            # else:
            #     assert(self.N_assoc_for_species[species][i,j]!=0, f"{self.N_assoc_for_species['pion'][i,j]=}")
            #     assert(self.N_assoc_for_species[species][i,j]!=0, f"{self.N_assoc_for_species['proton'][i,j]=}")
            #     assert(self.N_assoc_for_species[species][i,j]!=0, f"{self.N_assoc_for_species['kaon'][i,j]=}")
            pionEnhCorr = Corrs_raw[0].Clone()
            protonEnhCorr = Corrs_raw[1].Clone()
            kaonEnhCorr = Corrs_raw[2].Clone()
            otherEnhCorr = Corrs_raw[3].Clone()
            
            # get normalized mixed event correlation function
            NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
            # divide the raw correlation function by the mixed event correlation function
            
            pionEnhCorr.Divide(NormMEC)
            protonEnhCorr.Divide(NormMEC)
            kaonEnhCorr.Divide(NormMEC)
            otherEnhCorr.Divide(NormMEC)
            


        pionEnhCorrBinValues = [[pionEnhCorr.GetBinContent(x_bin, y_bin) for y_bin in range(1, pionEnhCorr.GetNbinsY()+1)] for x_bin in range(1, pionEnhCorr.GetNbinsX()+1)]
        pionEnhCorrBinErrors = [[pionEnhCorr.GetBinError(x_bin, y_bin) for y_bin in range(1, pionEnhCorr.GetNbinsY()+1)] for x_bin in range(1, pionEnhCorr.GetNbinsX()+1)]
        protonEnhCorrBinValues = [[protonEnhCorr.GetBinContent(x_bin, y_bin) for y_bin in range(1, protonEnhCorr.GetNbinsY()+1)] for x_bin in range(1, protonEnhCorr.GetNbinsX()+1)]
        protonEnhCorrBinErrors = [[protonEnhCorr.GetBinError(x_bin, y_bin) for y_bin in range(1, protonEnhCorr.GetNbinsY()+1)] for x_bin in range(1, protonEnhCorr.GetNbinsX()+1)]
        kaonEnhCorrBinValues = [[kaonEnhCorr.GetBinContent(x_bin, y_bin) for y_bin in range(1, kaonEnhCorr.GetNbinsY()+1)] for x_bin in range(1, kaonEnhCorr.GetNbinsX()+1)]
        kaonEnhCorrBinErrors = [[kaonEnhCorr.GetBinError(x_bin, y_bin) for y_bin in range(1, kaonEnhCorr.GetNbinsY()+1)] for x_bin in range(1, kaonEnhCorr.GetNbinsX()+1)]
        otherEnhCorrBinValues = [[otherEnhCorr.GetBinContent(x_bin, y_bin) for y_bin in range(1, otherEnhCorr.GetNbinsY()+1)] for x_bin in range(1, otherEnhCorr.GetNbinsX()+1)]
        otherEnhCorrBinErrors = [[otherEnhCorr.GetBinError(x_bin, y_bin) for y_bin in range(1, otherEnhCorr.GetNbinsY()+1)] for x_bin in range(1, otherEnhCorr.GetNbinsX()+1)]
        enhancedObservations = np.array([pionEnhCorrBinValues, protonEnhCorrBinValues, kaonEnhCorrBinValues, otherEnhCorrBinValues])
        enhancedObservationErrors = np.array([pionEnhCorrBinErrors, protonEnhCorrBinErrors, kaonEnhCorrBinErrors, otherEnhCorrBinErrors])

        # Get Mixing factors 
        if self.analysisType in ['central', 'semicentral']:
            NS_fit_params = self.PionTPCNSigmaFitObjs['NS'][i,j, k].popt
            NS_fit_errors = self.PionTPCNSigmaFitObjs['NS'][i,j, k].pcov
            NS_fit_func = self.PionTPCNSigmaFitObjs['NS'][i,j, k].upiKpInc_generalized_fit
            NS_generalized_gauss = self.PionTPCNSigmaFitObjs['NS'][i,j, k].ugeneralized_gauss
            NS_gauss = self.PionTPCNSigmaFitObjs['NS'][i,j, k].ugauss
            AS_fit_params = self.PionTPCNSigmaFitObjs['AS'][i,j, k].popt
            AS_fit_errors = self.PionTPCNSigmaFitObjs['AS'][i,j, k].pcov
            AS_fit_func = self.PionTPCNSigmaFitObjs['AS'][i,j, k].upiKpInc_generalized_fit
            AS_generalized_gauss = self.PionTPCNSigmaFitObjs['AS'][i,j, k].ugeneralized_gauss
            AS_gauss = self.PionTPCNSigmaFitObjs['AS'][i,j, k].ugauss
            BG_fit_params = self.PionTPCNSigmaFitObjs['BG'][i,j, k].popt
            BG_fit_errors = self.PionTPCNSigmaFitObjs['BG'][i,j, k].pcov
            BG_fit_func = self.PionTPCNSigmaFitObjs['BG'][i,j, k].upiKpInc_generalized_fit
            BG_generalized_gauss = self.PionTPCNSigmaFitObjs['BG'][i,j, k].ugeneralized_gauss
            BG_gauss = self.PionTPCNSigmaFitObjs['BG'][i,j, k].ugauss
        else:
            NS_fit_params = self.PionTPCNSigmaFitObjs['NS'][i,j].popt
            NS_fit_errors = self.PionTPCNSigmaFitObjs['NS'][i,j].pcov
            NS_fit_func = self.PionTPCNSigmaFitObjs['NS'][i,j].upiKpInc_generalized_fit
            NS_generalized_gauss = self.PionTPCNSigmaFitObjs['NS'][i,j].ugeneralized_gauss
            NS_gauss = self.PionTPCNSigmaFitObjs['NS'][i,j].ugauss
            AS_fit_params = self.PionTPCNSigmaFitObjs['AS'][i,j].popt
            AS_fit_errors = self.PionTPCNSigmaFitObjs['AS'][i,j].pcov
            AS_fit_func = self.PionTPCNSigmaFitObjs['AS'][i,j].upiKpInc_generalized_fit
            AS_generalized_gauss = self.PionTPCNSigmaFitObjs['AS'][i,j].ugeneralized_gauss
            AS_gauss = self.PionTPCNSigmaFitObjs['AS'][i,j].ugauss
            BG_fit_params = self.PionTPCNSigmaFitObjs['BG'][i,j].popt
            BG_fit_errors = self.PionTPCNSigmaFitObjs['BG'][i,j].pcov
            BG_fit_func = self.PionTPCNSigmaFitObjs['BG'][i,j].upiKpInc_generalized_fit
            BG_generalized_gauss = self.PionTPCNSigmaFitObjs['BG'][i,j].ugeneralized_gauss
            BG_gauss = self.PionTPCNSigmaFitObjs['BG'][i,j].ugauss
        
        NS_response = self.get_response_matrix(i,j,k,fit_params=NS_fit_params, fit_errors=NS_fit_errors, fit_func=NS_fit_func, generalized_gauss=NS_generalized_gauss, gauss=NS_gauss, region='NS')
        AS_response = self.get_response_matrix(i,j,k,fit_params=AS_fit_params, fit_errors=AS_fit_errors, fit_func=AS_fit_func, generalized_gauss=AS_generalized_gauss, gauss=AS_gauss, region='AS')
        BG_response = self.get_response_matrix(i,j,k,fit_params=BG_fit_params, fit_errors=BG_fit_errors, fit_func=BG_fit_func, generalized_gauss=BG_generalized_gauss, gauss=BG_gauss, region='BG')

        debug_logger.debug(f"{NS_response=}\n {AS_response=}\n {BG_response=}")
        nom = lambda x: x.n
        std = lambda x: x.s
        nom_vec = np.vectorize(nom)
        std_vec = np.vectorize(std)
        _,nphibins, netabins = enhancedObservations.shape
        if self.resetUnfoldingResults:
            self.unfoldedTruthValues = np.zeros((3, nphibins, netabins))
            self.unfoldedTruthErrors = np.zeros((3, nphibins, netabins))
            for x_bin in range(nphibins):
                debug_logger.debug("Unfolding bin ({})".format(x_bin))
                for y_bin in range(netabins):
                    if enhancedObservations[:, x_bin, y_bin].sum() == 0:
                        continue
                    if pionEnhCorr.GetXaxis().GetBinCenter(x_bin+1) > self.dPhiSigNS[0] and pionEnhCorr.GetXaxis().GetBinCenter(x_bin+1) < self.dPhiSigNS[1] and pionEnhCorr.GetYaxis().GetBinCenter(y_bin+1) > self.dEtaSig[0] and pionEnhCorr.GetYaxis().GetBinCenter(y_bin+1) < self.dEtaSig[1]:
                        response=nom_vec(NS_response)
                        response_err=std_vec(NS_response)
                    elif pionEnhCorr.GetXaxis().GetBinCenter(x_bin+1) > self.dPhiSigAS[0] and pionEnhCorr.GetXaxis().GetBinCenter(x_bin+1) < self.dPhiSigAS[1] and pionEnhCorr.GetYaxis().GetBinCenter(y_bin+1) > self.dEtaSigAS[0] and pionEnhCorr.GetYaxis().GetBinCenter(y_bin+1) < self.dEtaSigAS[1]:
                        response=nom_vec(AS_response)
                        response_err=std_vec(AS_response)
                    else:
                        response=nom_vec(BG_response)
                        response_err=std_vec(BG_response)

                        if self.analysisType == 'pp' and j>=3:
                            self.unfoldedTruthValues[:, x_bin, y_bin] = enhancedObservations[:-1, x_bin, y_bin]
                            self.unfoldedTruthErrors[:, x_bin, y_bin] = enhancedObservationErrors[:-1, x_bin, y_bin]  
                            continue
                        
                    unfolded = iterative_unfold(data=enhancedObservations[:, x_bin, y_bin], data_err=enhancedObservationErrors[:, x_bin, y_bin], response=response, response_err=response_err, efficiencies=[1,1,1], efficiencies_err=[0.0,0.0,0.0], max_iter=10, ts='chi2', ts_stopping=1e-6)
                    self.unfoldedTruthValues[:, x_bin, y_bin] = unfolded['unfolded']
                    self.unfoldedTruthErrors[:, x_bin, y_bin] = unfolded['stat_err']  
                    unfolding_logger.info(f"{response}")

                    unfolding_logger.info(f"Unfolded matrix for bin ({x_bin}, {y_bin}) in {unfolded['num_iterations']} iterations")
                    unfolding_logger.info(f"{np.array2string(unfolded['unfolding_matrix'], formatter={'float': lambda x: f'{x:.2f}'})}")
                    # refold the truth values to get the refolded values
                    refolded = np.matmul(unfolded['unfolding_matrix'], unfolded['unfolded'])
                    # print the delta between the refolded values and the original values
                    unfolding_logger.info(f"Percent delta between refolded and original values for bin ({x_bin}, {y_bin})")
                    unfolding_logger.info(f"{refolded=}, {enhancedObservations[:, x_bin, y_bin]=}, {np.array2string((refolded - enhancedObservations[:, x_bin, y_bin])/enhancedObservations[:, x_bin, y_bin], formatter={'float': lambda x: f'{x:.2f}'})}")

                    # log the matrix product of the unfolding matrix and the response matrix
                    # unfolding_logger.info(f"Unfolding matrix times response matrix for bin ({x_bin}, {y_bin})")
                    # unfolding_logger.info(f"{np.array2string(np.matmul(unfolded['unfolding_matrix'], response), formatter={'float': lambda x: f'{x:.2f}'})}")
            self.resetUnfoldingResults = False
        #inv_mat = unp.ulinalg.pinv(A)
   

        if species == 'pion':

            Corr = pionEnhCorr.Clone()
            Corr_err = pionEnhCorr.Clone()
            Corr.Reset()
            Corr_err.Reset()
            for x_bin in range(nphibins):
                for y_bin in range(netabins):
                    Corr.SetBinContent(x_bin, y_bin, self.unfoldedTruthValues[0, x_bin, y_bin])
                    Corr.SetBinError(x_bin, y_bin, self.unfoldedTruthErrors[0, x_bin, y_bin])
                    Corr_err.SetBinContent(x_bin, y_bin, self.unfoldedTruthErrors[0, x_bin, y_bin])



          

        elif species == 'proton':


            Corr = pionEnhCorr.Clone()
            Corr_err = pionEnhCorr.Clone()
            Corr.Reset()
            Corr_err.Reset()
            for x_bin in range(nphibins):
                for y_bin in range(netabins):
                    Corr.SetBinContent(x_bin, y_bin, self.unfoldedTruthValues[1, x_bin, y_bin])
                    Corr.SetBinError(x_bin, y_bin, self.unfoldedTruthErrors[1, x_bin, y_bin])
                    Corr_err.SetBinContent(x_bin, y_bin, self.unfoldedTruthErrors[1, x_bin, y_bin])
            
        elif species == 'kaon':

            Corr = pionEnhCorr.Clone()
            Corr_err = pionEnhCorr.Clone()
            Corr.Reset()
            Corr_err.Reset()
            for x_bin in range(nphibins):
                for y_bin in range(netabins):
                    Corr.SetBinContent(x_bin, y_bin, self.unfoldedTruthValues[2, x_bin, y_bin])
                    Corr.SetBinError(x_bin, y_bin, self.unfoldedTruthErrors[2, x_bin, y_bin])           
                    Corr_err.SetBinContent(x_bin, y_bin, self.unfoldedTruthErrors[2, x_bin, y_bin])
           
        del NormMEC


        return Corr.Clone(), Corr_err.Clone()

        

    def get_acceptance_corrected_correlation_function_w_pionTPCnSigma(self, species):
    
        # get the raw correlation functions
        Corr = self.get_SE_correlation_function_w_pionTPCnSigma(species)
        # get normalized mixed event correlation function
        NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
        # we need to make a compatible 3d histogram. Let's do this be taking NormMEC and making a 3d histogram with the same binning as Corr
        # get the binning of Corr
        xbins = Corr.GetXaxis().GetNbins()
        xlow = Corr.GetXaxis().GetXmin()
        xhigh = Corr.GetXaxis().GetXmax()
        ybins = Corr.GetYaxis().GetNbins()
        ylow = Corr.GetYaxis().GetXmin()
        yhigh = Corr.GetYaxis().GetXmax()
        zbins = Corr.GetZaxis().GetNbins()
        zlow = Corr.GetZaxis().GetXmin()
        zhigh = Corr.GetZaxis().GetXmax()

        # make a 3d histogram with the same binning as Corr
        NormMEC_3d = TH3D(
            "NormMEC_3d",
            "NormMEC_3d",
            xbins,
            xlow,
            xhigh,
            ybins,
            ylow,
            yhigh,
            zbins,
            zlow,
            zhigh,
        )
        # fill the 3d histogram with the 2d histogram and errors from the 2d histogram
        for i in range(0, xbins + 2):
            for j in range(0, ybins + 2):
                for k in range(0, zbins + 2):
                    NormMEC_3d.SetBinContent(
                        i, j, k, NormMEC.GetBinContent(i, j) / zbins
                    )  # yay for github copilot ! 🥳🥳🥂
                    NormMEC_3d.SetBinError(
                        i, j, k, NormMEC.GetBinError(i, j) / zbins
                    )

        # divide the raw correlation function by the mixed event correlation function
        Corr.Divide(NormMEC_3d)
        # Corr.GetXaxis().SetRangeUser(-0.9, 0.9)
        del NormMEC
        self.get_acceptance_corrected_correlation_function_w_pionTPCnSigma_result[
            species
        ] = Corr
        self.get_acceptance_corrected_correlation_function_w_pionTPCnSigma_has_changed[
            species
        ] = False
        return Corr.Clone()
        

    def get_N_trig(self):
        """
        Returns the number of trigger particles in the event
        Assume the Trigger sparse being passed has already been limited to the jet pT bin in question
        """
        N_trig_val = 0
        N_trig_hist = None
        for sparse_ind in range(len(self.Trigger)):
            if N_trig_hist is None:
                N_trig_hist = self.Trigger[sparse_ind].Projection(
                    1
                )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
            else:
                N_trig_hist.Add(
                    self.Trigger[sparse_ind].Projection(1)
                )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
        N_trig_val = N_trig_hist.GetEntries()  # type:ignore
        del N_trig_hist
        return N_trig_val
    
    def get_N_assoc(self, region = None):
        """
        Returns the number of trigger particles in the event
        Assume the Trigger sparse being passed has already been limited to the jet pT bin in question
        """
        N_assoc_val = 0
        N_assoc_hist = None
        
        for sparse_ind in range(len(self.JH)):
            if region is not None:
                if region == "NS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSig[0], self.dEtaSig[1]
                    )  # deta
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )  # dphi

                elif region == "AS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSigAS[0], self.dEtaSigAS[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigAS[0], self.dPhiSigAS[1]
                    )
                if region == "BGHi":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGHi[0], self.dEtaBGHi[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )
                if region == "BGLo":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGLo[0], self.dEtaBGLo[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )
            if N_assoc_hist is None:
                N_assoc_hist = self.JH[sparse_ind].Projection(
                    2
                )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
            else:
                N_assoc_hist.Add(
                    self.JH[sparse_ind].Projection(2)
                )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
        N_assoc_val = N_assoc_hist.GetEntries()  # type:ignore
        del N_assoc_hist
        if region is not None:
            self.JH[sparse_ind].GetAxis(3).SetRange(
                0, self.JH[sparse_ind].GetAxis(3).GetNbins() + 1
            )  # deta # type:ignore
            self.JH[sparse_ind].GetAxis(4).SetRange(
                0, self.JH[sparse_ind].GetAxis(4).GetNbins() + 1
            )  # dphi # type:ignore
        return N_assoc_val
    
    def get_N_assoc_for_species(self,i,j,k,species, region=None):
        """
        Returns the number of trigger particles in the event
        Assume the Trigger sparse being passed has already been limited to the jet pT bin in question
        """
        N_assoc = 0
        N_assoc_hist = None
        # restrict sparse to species in question
        offset = 1 if self.analysisType == "pp" else 0  # type:ignore
    
        speciesID = (
            9
            if species == "pion"
            else 10
            if species == "proton"
            else 11
        )
        for sparse_ind in range(len(self.JH)):
            if region is not None:
                if region == "NS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSig[0], self.dEtaSig[1]
                    )  # deta
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )  # dphi

                elif region == "AS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSigAS[0], self.dEtaSigAS[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigAS[0], self.dPhiSigAS[1]
                    )
                if region == "BGHi":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGHi[0], self.dEtaBGHi[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )
                if region == "BGLo":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGLo[0], self.dEtaBGLo[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )

            if species=='pion':
                self.JH[sparse_ind].GetAxis(9 - offset).SetRangeUser(
                        -2 , 2
                    )  # type:ignore
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(11-offset).FindBin(-2 if j<2 else -1))
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2))
                if N_assoc_hist is None:
                    N_assoc_hist = self.JH[sparse_ind].Projection(
                        2
                    )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
                else:
                    N_assoc_hist.Add(
                        self.JH[sparse_ind].Projection(2)
                    )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
                    #underflow
                # self.JH[sparse_ind].GetAxis(11-offset).SetRange(0, 0)   
                # N_assoc_hist.Add(
                #     self.JH[sparse_ind].Projection(2)
                # ) 
            elif species=='kaon':
                self.JH[sparse_ind].GetAxis(11 - offset).SetRangeUser(
                    -2 if j<2 else -1 , 2
                )  # type:ignore
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2))
                if N_assoc_hist is None:
                    N_assoc_hist = self.JH[sparse_ind].Projection(
                        2
                    )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
                else:
                    N_assoc_hist.Add(
                        self.JH[sparse_ind].Projection(2)
                    )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
                    #underflow
                # self.JH[sparse_ind].GetAxis(10-offset).SetRange(0, 0)   
                # N_assoc_hist.Add(
                #     self.JH[sparse_ind].Projection(2)
                # ) 
            elif species=='proton':
                self.JH[sparse_ind].GetAxis(10-offset).SetRangeUser(-2, 2 )   
                if N_assoc_hist is None:
                    N_assoc_hist = self.JH[sparse_ind].Projection(
                        2
                    )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
                else:
                    N_assoc_hist.Add(
                        self.JH[sparse_ind].Projection(2)
                    )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
            self.JH[sparse_ind].GetAxis(9-offset).SetRange(0, 0)   
            self.JH[sparse_ind].GetAxis(10-offset).SetRange(0, 0)
            self.JH[sparse_ind].GetAxis(11-offset).SetRange(0, 0)   

        N_assoc = N_assoc_hist.GetEntries()  # type:ignore
        del N_assoc_hist
        self.JH[sparse_ind].GetAxis(3).SetRange(
            0, self.JH[sparse_ind].GetAxis(3).GetNbins() + 1
        )  # deta # type:ignore
        self.JH[sparse_ind].GetAxis(4).SetRange(
            0, self.JH[sparse_ind].GetAxis(4).GetNbins() + 1
        )  # dphi # type:ignore
        return N_assoc

    def get_normalized_acceptance_corrected_correlation_function(self, in_z_vertex_bins=False):
        """
        Returns the acceptance corrected correlation function normalized by the number of trigger particles
        """
    
        N_trig = self.get_N_trig()
        AccCorrectedSEcorr = self.get_acceptance_corrected_correlation_function(in_z_vertex_bins=in_z_vertex_bins)
        AccCorrectedSEcorr.Scale(1 / N_trig)
        del N_trig
        self.get_normalized_acceptance_corrected_correlation_function_result = (
            AccCorrectedSEcorr
        )
        self.get_normalized_acceptance_corrected_correlation_function_has_changed = (
            False
        )
        return AccCorrectedSEcorr.Clone()
        
    def get_dPhi_projection_in_dEta_range(self, dEtaRange, rebin=True, scaleUp=False, in_z_vertex_bins=False):
        """
        Returns the dPhi distribution for a given dEta range
        """
        tempHist = self.get_acceptance_corrected_correlation_function(in_z_vertex_bins=in_z_vertex_bins)
        lowerBin, upperBin = (tempHist.GetXaxis().FindBin(bin) for bin in dEtaRange)
        tempHist.GetXaxis().SetRange(lowerBin, upperBin)
        dPhi = tempHist.ProjectionY()
        nbins = upperBin - lowerBin
        binwidth = tempHist.GetXaxis().GetBinWidth(1)
        dPhi.Scale(binwidth) # scale by the number of bins in the dEta range to get the correct normalization
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(
                -0.6
            )
            debug_logger.info(f"Grabbing between bins {upperBin} and {lowerBin}")
            dPhi.Scale((nbins_sig + 1) / (nbins + 1))
        # rebin dPhi
        if rebin:
            dPhi = dPhi.Rebin(2)
            dPhi.Scale(1 / 2)
        return dPhi.Clone(), nbins, binwidth

    def get_dPhi_projection_in_dEta_range_for_enhanced_species(
        self, i,j,k,dEtaRange, TOFcutSpecies, rebin=True, scaleUp=False, in_z_vertex_bins=False
    ):
        """
        Returns the dPhi distribution for a given dEta range
        """
        tempHist = self.get_acceptance_corrected_correlation_function_for_enhanced_species(i,j,k,
            TOFcutSpecies, in_z_vertex_bins=in_z_vertex_bins
        )
        lowerBin, upperBin = (tempHist.GetXaxis().FindBin(bin) for bin in dEtaRange)
        tempHist.GetXaxis().SetRange(lowerBin, upperBin)

        dPhi = tempHist.ProjectionY()

        nbins = upperBin - lowerBin
        binwidth = tempHist.GetXaxis().GetBinWidth(1)
        dPhi.Scale(binwidth) # scale by the number of bins in the dEta range to get the correct normalization
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(
                -0.6
            )
            debug_logger.info(f"Grabbing between bins {upperBin} and {lowerBin}")
            dPhi.Scale((nbins_sig + 1) / (nbins + 1))
        # rebin dPhi
        if rebin:
            dPhi = dPhi.Rebin(2)
            dPhi.Scale(0.5)

        return dPhi.Clone(), nbins, binwidth
    
    def get_dPhi_projection_in_dEta_range_for_true_species(
        self, i,j,k,dEtaRange, TOFcutSpecies, rebin=True, scaleUp=False, in_z_vertex_bins=False
    ):
        """
        Returns the dPhi distribution for a given dEta range
        """
        tempHist, PIDerrTH2 = self.get_acceptance_corrected_correlation_function_for_true_species(i,j,k,
            TOFcutSpecies, in_z_vertex_bins=in_z_vertex_bins
        )
        lowerBin, upperBin = (tempHist.GetXaxis().FindBin(bin) for bin in dEtaRange)
        tempHist.GetXaxis().SetRange(lowerBin, upperBin)
        PIDerrTH2.GetXaxis().SetRange(lowerBin, upperBin)

        dPhi = tempHist.ProjectionY()
        dPhiPIDErr = PIDerrTH2.ProjectionY()

        nbins = upperBin - lowerBin
        binwidth = tempHist.GetXaxis().GetBinWidth(1)
        dPhi.Scale(binwidth) # scale by the number of bins in the dEta range to get the correct normalization
        dPhiPIDErr.Scale(binwidth) # scale by the number of bins in the dEta range to get the correct normalization
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(
                -0.6
            )
            debug_logger.info(f"Grabbing between bins {upperBin} and {lowerBin}")
            dPhi.Scale((nbins_sig + 1) / (nbins + 1))
            dPhiPIDErr.Scale((nbins_sig + 1) / (nbins + 1))
        # rebin dPhi
        if rebin:
            dPhi = dPhi.Rebin(2)
            dPhi.Scale(0.5)
            dPhiPIDErr = dPhiPIDErr.Rebin(2)
            dPhiPIDErr.Scale(0.5)
        return dPhi.Clone(), nbins, binwidth, dPhiPIDErr.Clone()

    def get_dPhi_dpionTPCnSigma_projection_in_dEta_range(
        self, dEtaRange, TOFcutSpecies, rebin=True, scaleUp=False
    ):
        """
        Returns the dPhi distribution for a given dEta range
        """
        tempHist = self.get_acceptance_corrected_correlation_function_w_pionTPCnSigma(
            TOFcutSpecies
        )
        lowerBin, upperBin = (tempHist.GetXaxis().FindBin(bin) for bin in dEtaRange)
        tempHist.GetXaxis().SetRange(lowerBin, upperBin)
        dPhidpionTPCnSigma = tempHist.Project3D("zy")
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            binwidth = tempHist.GetXaxis().GetBinWidth(1)
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(
                -0.6
            )
            debug_logger.info(f"Grabbing between bins {upperBin} and {lowerBin}")
            dPhidpionTPCnSigma.Scale((nbins_sig + 1) / (nbins + 1))
        # rebin dPhi
        if rebin:
            dPhidpionTPCnSigma = dPhidpionTPCnSigma.RebinX(2)
            dPhidpionTPCnSigma.Scale(0.5)
        return dPhidpionTPCnSigma.Clone()

    def get_dEta_projection_NS(self):
        """
        Returns the dPhi distribution for a given dEta range
        """
        tempHist = self.get_acceptance_corrected_correlation_function()
        tempHist.GetYaxis().SetRangeUser(-np.pi / 2, np.pi / 2)
        dEta = tempHist.ProjectionX().Clone()
        return dEta

    def get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(self, i, j, k, in_z_vertex_bins=False):
        """
        Returns the background subtracted dPhi distribution for the signal region
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the dPhi distribution for the signal region
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsZV[i, j, k] 
            else:
                dPhiSig = self.dPhiSigcorrs[i, j, k]  # type:ignore
        elif self.analysisType == "pp":  # type:ignore
            if in_z_vertex_bins: 
                dPhiSig = self.dPhiSigcorrsZV[i, j]
            else:
                dPhiSig = self.dPhiSigcorrs[i, j]  # type:ignore
        # get the bin contents
        dPhiSigBinContents = self.get_bin_contents_as_array(
            dPhiSig, forFitting=True
        )  # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(
            dPhiSig, forFitting=True
        )  # type:ignore
        # get x range
        x_vals = self.get_bin_centers_as_array(dPhiSig, forFitting=True)  # type:ignore
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # generate the background distribution from the RPFObj
            if in_z_vertex_bins:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsZV[i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsZV[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsZV[i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsZV[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                ) 
            else:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjs[i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjs[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjs[i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjs[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            
            if k in [0, 1, 2]:
                dPhiBGRPF = dPhiBGRPFs[:, k]  # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:, k]  # type:ignore

            if k == 3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs*[self.N_trigs[i,0],self.N_trigs[i,1],self.N_trigs[i,2]]/(self.N_trigs[i,0]+self.N_trigs[i,1]+self.N_trigs[i,2]), axis=1)) 
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0, 1, 2, 3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")
            # subtract the background from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - dPhiBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigNSHist",
                "BGSubtracteddPhiSigNSHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone()

        elif self.analysisType == "pp":  # type:ignore
            # get the minimum value of the signal
            minVal = np.min(dPhiSigBinContents)
            # subtract the minimum value from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2 + minVal**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigNSHist",
                "BGSubtracteddPhiSigNSHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone(), minVal

    def get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(self, i, j, k, in_z_vertex_bins=False):
        """
        Returns the background subtracted dPhi distribution for the signal region
        """
        # get the dPhi distribution for the signal region
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsZV[i, j, k]
            else:
                dPhiSig = self.dPhiSigcorrs[i, j, k]  # type:ignore
        elif self.analysisType == "pp":  # type:ignore
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsZV[i, j]
            else:
                dPhiSig = self.dPhiSigcorrs[i, j]  # type:ignore
        # dPhiSig.Scale(0.5) # to put it on the same level as the background
        # get the bin contents
        dPhiSigBinContents = self.get_bin_contents_as_array(
            dPhiSig, forFitting=False, forAS=True
        )  # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(
            dPhiSig, forFitting=False, forAS=True
        )  # type:ignore
        # get x range
        x_vals = self.get_bin_centers_as_array(
            dPhiSig, forFitting=False, forAS=True
        )  # type:ignore
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # generate the background distribution from the RPFObj
            if in_z_vertex_bins:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsZV[i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsZV[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsZV[i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsZV[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            else:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjs[i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjs[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjs[i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjs[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            if k in [0, 1, 2]:
                dPhiBGRPF = dPhiBGRPFs[:, k]  # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:, k]  # type:ignore
            if k == 3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs*[self.N_trigs[i,0],self.N_trigs[i,1],self.N_trigs[i,2]]/(self.N_trigs[i,0]+self.N_trigs[i,1]+self.N_trigs[i,2]), axis=1)) 
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0, 1, 2, 3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")

            # subtract the background from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - dPhiBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigASHist",
                "BGSubtracteddPhiSigASHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone()
        elif self.analysisType == "pp":  # type:ignore
            # compute the minimum value of dPhiSigBinContents
            minVal, minValErr = self.get_minVal_and_systematic_error(dPhiSigBinContents)
            # subtract the minimum value from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal
            # Get the error on the background
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigASHist",
                "BGSubtracteddPhiSigASHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone(), minValErr

    def get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(self, i, j, k, in_z_vertex_bins=False):
        """
        Returns the background subtracted dPhi distribution for the signal region
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsZV[i, j, k] # type:ignore
            else:
                dPhiSig = self.dPhiSigcorrs[i, j, k]  # type:ignore
        elif self.analysisType == "pp":  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsZV[i, j]
            else:
                dPhiSig = self.dPhiSigcorrs[i, j]  # type:ignore

        # get the bin contents
        dPhiSigBinContents = self.get_bin_contents_as_array(
            dPhiSig, forFitting=False
        )  # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(
            dPhiSig, forFitting=False
        )  # type:ignore
        # get x range
        x_vals = self.get_bin_centers_as_array(dPhiSig, forFitting=False)  # type:ignore
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # generate the background distribution from the RPFObj
            if in_z_vertex_bins:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsZV[i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsZV[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsZV[i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsZV[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            else:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjs[i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjs[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjs[i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjs[i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            if k in [0, 1, 2]:
                dPhiBGRPF = dPhiBGRPFs[:, k]  # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:, k]  # type:ignore
            if k == 3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs*[self.N_trigs[i,0],self.N_trigs[i,1],self.N_trigs[i,2]]/(self.N_trigs[i,0]+self.N_trigs[i,1]+self.N_trigs[i,2]), axis=1))  # here we take the weightes average of the 3 RPFObjs, weighted by the number of triggers in each RPFObj
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0, 1, 2, 3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")
            # subtract the background from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - dPhiBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigHist",
                "BGSubtracteddPhiSigHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])

            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution
            BGSubtracteddPhiSigHist.Scale(1 / N_trig)  # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone()

        elif self.analysisType == "pp":  # type:ignore
            # get the minimum value of the signal
            minVal, minValErr = self.get_minVal_and_systematic_error(dPhiSigBinContents)
            # subtract the minimum value from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigHist",
                "BGSubtracteddPhiSigHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution
            BGSubtracteddPhiSigHist.Scale(1 / N_trig)  # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone(), minValErr

    def get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_true_species(
        self, i, j, k, species, in_z_vertex_bins=False
    ):
        """
        Returns the background subtracted dPhi dpionTPCnSigma distribution after a TOF species cut for the signal region
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsForTrueSpeciesZV[species][i, j, k]
            else:
                dPhiSig = self.dPhiSigcorrsForTrueSpecies[species][i, j, k]  # type:ignore
        elif self.analysisType == "pp":  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsForTrueSpeciesZV[species][i, j]
                dPhiBG = self.dPhiBGcorrsForTrueSpeciesZV[species][i, j]
            else:
                dPhiSig = self.dPhiSigcorrsForTrueSpecies[species][i, j]  # type:ignore
                dPhiBG = self.dPhiBGcorrsForTrueSpecies[species][i, j]  # type:ignore

        # get the bin contents
        dPhiSigBinContents = self.get_bin_contents_as_array(
            dPhiSig, forFitting=False
        )  # type:ignore
        
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(
            dPhiSig, forFitting=False
        )  # type:ignore
        # get x range
        x_vals = self.get_bin_centers_as_array(dPhiSig, forFitting=False)  # type:ignore
      
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # generate the background distribution from the RPFObj
            if in_z_vertex_bins:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsForTrueSpeciesZV[species][i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsForTrueSpeciesZV[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsForTrueSpeciesZV[species][i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsForTrueSpeciesZV[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            else:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsForTrueSpecies[species][i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsForTrueSpecies[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsForTrueSpecies[species][i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsForTrueSpecies[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            if k in [0, 1, 2]:
                dPhiBGRPF = dPhiBGRPFs[:, k]  # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:, k]  # type:ignore

            if k == 3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs*[self.N_trigs[i,0],self.N_trigs[i,1],self.N_trigs[i,2]]/(self.N_trigs[i,0]+self.N_trigs[i,1]+self.N_trigs[i,2]), axis=1)) 
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0, 1, 2, 3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")

            #
            # subtract the background from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - dPhiBGRPF 

            # get the error on the bdPhiBGRPFErrackground subtracted signal
            BGSubtracteddPhiSigErr =dPhiBGRPFErr

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigHist",
                "BGSubtracteddPhiSigHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])

            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution
            BGSubtracteddPhiSigHist.Scale(1 / N_trig)  # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone()

        elif self.analysisType == "pp":  # type:ignore
            dPhiBGBinContents = self.get_bin_contents_as_array(
                dPhiBG, forFitting=False
            )  # type:ignore    
            # get the minimum value of the signal
            minVal, minValErr = self.get_minVal_and_systematic_error(dPhiBGBinContents)
            #minVal = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[
            #    i, j
            #]  # type:ignore
            # subtract the minimum value from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigHist",
                "BGSubtracteddPhiSigHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution
            BGSubtracteddPhiSigHist.Scale(1 / N_trig)  # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone(), minValErr
    
    def get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_enhanced_species(
        self, i, j, k, species, in_z_vertex_bins=False
    ):
        """
        Returns the background subtracted dPhi dpionTPCnSigma distribution after a TOF species cut for the signal region
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsForEnhancedSpeciesZV[species][i, j, k]
            else:
                dPhiSig = self.dPhiSigcorrsForEnhancedSpecies[species][i, j, k]  # type:ignore
        elif self.analysisType == "pp":  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsForEnhancedSpeciesZV[species][i, j]
                dPhiBG = self.dPhiBGcorrsForEnhancedSpeciesZV[species][i, j]
            else:
                dPhiSig = self.dPhiSigcorrsForEnhancedSpecies[species][i, j]  # type:ignore
                dPhiBG = self.dPhiBGcorrsForEnhancedSpecies[species][i, j]  # type:ignore

        # get the bin contents
        dPhiSigBinContents = self.get_bin_contents_as_array(
            dPhiSig, forFitting=False
        )  # type:ignore
        
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(
            dPhiSig, forFitting=False
        )  # type:ignore
        # get x range
        x_vals = self.get_bin_centers_as_array(dPhiSig, forFitting=False)  # type:ignore
      
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # generate the background distribution from the RPFObj
            if in_z_vertex_bins:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsForEnhancedSpeciesZV[species][i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsForEnhancedSpeciesZV[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsForEnhancedSpeciesZV[species][i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsForEnhancedSpeciesZV[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            else:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsForEnhancedSpecies[species][i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsForEnhancedSpecies[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsForEnhancedSpecies[species][i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsForEnhancedSpecies[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            if k in [0, 1, 2]:
                dPhiBGRPF = dPhiBGRPFs[:, k]  # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:, k]  # type:ignore

            if k == 3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs*[self.N_trigs[i,0],self.N_trigs[i,1],self.N_trigs[i,2]]/(self.N_trigs[i,0]+self.N_trigs[i,1]+self.N_trigs[i,2]), axis=1)) 
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0, 1, 2, 3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")

            #
            # subtract the background from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - dPhiBGRPF 

            # get the error on the bdPhiBGRPFErrackground subtracted signal
            BGSubtracteddPhiSigErr =dPhiBGRPFErr

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigHist",
                "BGSubtracteddPhiSigHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])

            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution
            BGSubtracteddPhiSigHist.Scale(1 / N_trig)  # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone()

        elif self.analysisType == "pp":  # type:ignore
            dPhiBGBinContents = self.get_bin_contents_as_array(
                dPhiBG, forFitting=False
            )  # type:ignore    
            # get the minimum value of the signal
            minVal, minValErr = self.get_minVal_and_systematic_error(dPhiBGBinContents)
            #minVal = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[
            #    i, j
            #]  # type:ignore
            # subtract the minimum value from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D(
                "BGSubtracteddPhiSigHist",
                "BGSubtracteddPhiSigHist",
                len(x_vals),
                x_vals[0],
                x_vals[-1],
            )
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l + 1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l + 1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution
            BGSubtracteddPhiSigHist.Scale(1 / N_trig)  # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone(), minValErr

    def get_normalized_background_subtracted_dPhi_for_true_species(self, i, j, k, species, in_z_vertex_bins=False):
        """
        Returns the normalized background subtracted dPhi distribution for a given species
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the background subtracted dPhi distribution for a given species
            BGSubtracteddPhiSigForSpeciesHist = (
                self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_true_species(
                    i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
                )
            )
            # project the dPhi distribution onto the x-axis
            # return the background subtracted dPhi distribution
            return BGSubtracteddPhiSigForSpeciesHist.Clone()

        elif self.analysisType == "pp":  # type:ignore
            # get the background subtracted dPhi distribution for a given species
            (
                BGSubtracteddPhiSigForSpeciesHist,
                minValErr,
            ) = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_true_species(
                i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
            )
            # return the background subtracted dPhi distribution
            return BGSubtracteddPhiSigForSpeciesHist.Clone(), minValErr
    
    def get_normalized_background_subtracted_dPhi_for_enhanced_species(self, i, j, k, species, in_z_vertex_bins=False):
        """
        Returns the normalized background subtracted dPhi distribution for a given species
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the background subtracted dPhi distribution for a given species
            BGSubtracteddPhiSigForSpeciesHist = (
                self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_enhanced_species(
                    i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
                )
            )
            # project the dPhi distribution onto the x-axis
            # return the background subtracted dPhi distribution
            return BGSubtracteddPhiSigForSpeciesHist.Clone()

        elif self.analysisType == "pp":  # type:ignore
            # get the background subtracted dPhi distribution for a given species
            (
                BGSubtracteddPhiSigForSpeciesHist,
                minValErr,
            ) = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_enhanced_species(
                i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
            )
            # return the background subtracted dPhi distribution
            return BGSubtracteddPhiSigForSpeciesHist.Clone(), minValErr

    def get_inclusive_yield(self, i, j, k, region=None, in_z_vertex_bins=False):
        """
        Returns the yield for a given species
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            BGSubtracteddPhiSigHist = (
                self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(
                    i, j, k, in_z_vertex_bins=in_z_vertex_bins
                )
            )
        elif self.analysisType == "pp":  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            (
                BGSubtracteddPhiSigHist,
                minValErr,
            ) = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(
                i, j, k, in_z_vertex_bins=in_z_vertex_bins
            )  # type:ignore
        # get the yield for a given species
        bin_width_dphi = BGSubtracteddPhiSigHist.GetBinWidth(1)
        bin_width_ptassoc = self.pTassocBinWidths[j]
        if region is None:
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(1, BGSubtracteddPhiSigHist.GetNbinsX() + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(
                        1, BGSubtracteddPhiSigHist.GetNbinsX() + 1
                    )
                )
            )
        elif region == "NS":
            # get bins at location dPhiSigNS[0] and dPhiSigNS[1]
            (
                low_bin,
                high_bin,
            ) = BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigNS[0]
            ), BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigNS[1]
            )
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(low_bin, high_bin + 1)
                )
            )
        elif region == "AS":
            (
                low_bin,
                high_bin,
            ) = BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigAS[0]
            ), BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigAS[1]
            )
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(low_bin, high_bin + 1)
                )
            )
            # return the yield
        yield_, yield_err_ = yield_ / bin_width_ptassoc, yield_err_ / bin_width_ptassoc
        return yield_, yield_err_

    def get_yield_for_true_species(self, i, j, k, species, region=None, in_z_vertex_bins=False):
        """
        Returns the yield for a given species
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            BGSubtracteddPhiSigHist = (
                self.get_normalized_background_subtracted_dPhi_for_true_species(
                    i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
                )
            )
        elif self.analysisType == "pp":  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            (
                BGSubtracteddPhiSigHist,
                minValErr,
            ) = self.get_normalized_background_subtracted_dPhi_for_true_species(
                i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
            )  # type:ignore
        # get the yield for a given species
        bin_width_dphi = BGSubtracteddPhiSigHist.GetBinWidth(1)
        bin_width_ptassoc = self.pTassocBinWidths[j]
        if region is None:
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(1, BGSubtracteddPhiSigHist.GetNbinsX() + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(
                        1, BGSubtracteddPhiSigHist.GetNbinsX() + 1
                    )
                )
            )
        elif region == "NS":
            # get bins at location dPhiSigNS[0] and dPhiSigNS[1]
            (
                low_bin,
                high_bin,
            ) = BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigNS[0]
            ), BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigNS[1]
            )
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(low_bin, high_bin + 1)
                )
            )
        elif region == "AS":
            (
                low_bin,
                high_bin,
            ) = BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigAS[0]
            ), BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigAS[1]
            )
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(low_bin, high_bin + 1)
                )
            )
            # return the yield
        yield_, yield_err_ = yield_ / bin_width_ptassoc, yield_err_ / bin_width_ptassoc
        return yield_, yield_err_
    
    def get_yield_for_enhanced_species(self, i, j, k, species, region=None, in_z_vertex_bins=False):
        """
        Returns the yield for a given species
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            BGSubtracteddPhiSigHist = (
                self.get_normalized_background_subtracted_dPhi_for_enhanced_species(
                    i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
                )
            )
        elif self.analysisType == "pp":  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            (
                BGSubtracteddPhiSigHist,
                minValErr,
            ) = self.get_normalized_background_subtracted_dPhi_for_enhanced_species(
                i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
            )  # type:ignore
        # get the yield for a given species
        bin_width_dphi = BGSubtracteddPhiSigHist.GetBinWidth(1)
        bin_width_ptassoc = self.pTassocBinWidths[j]
        if region is None:
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(1, BGSubtracteddPhiSigHist.GetNbinsX() + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(
                        1, BGSubtracteddPhiSigHist.GetNbinsX() + 1
                    )
                )
            )
        elif region == "NS":
            # get bins at location dPhiSigNS[0] and dPhiSigNS[1]
            (
                low_bin,
                high_bin,
            ) = BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigNS[0]
            ), BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigNS[1]
            )
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(low_bin, high_bin + 1)
                )
            )
        elif region == "AS":
            (
                low_bin,
                high_bin,
            ) = BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigAS[0]
            ), BGSubtracteddPhiSigHist.GetXaxis().FindBin(
                self.dPhiSigAS[1]
            )
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width_dphi
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width_dphi)
                    ** 2
                    for l in range(low_bin, high_bin + 1)
                )
            )
            # return the yield
        yield_, yield_err_ = yield_ / bin_width_ptassoc, yield_err_ / bin_width_ptassoc
        return yield_, yield_err_

    def get_minVal_and_systematic_error(self, dPhiBinContents):
        """
        Returns the minimum value of the dPhi distribution and the systematic error
        """
        # Get the minimum value in three different ways: standard, sliding window average with a window of 3 points, and a sliding window average with a window of 5 points
        if len(dPhiBinContents.shape) > 1:
            dPhiBinContents = np.mean(dPhiBinContents, axis=1)
        minVal1 = np.min(dPhiBinContents)
        minVal2 = np.min(
            [
                np.mean(dPhiBinContents[i : i + 3])
                for i in range(len(dPhiBinContents) - 3)
            ]
        )
        minVal3 = np.min(
            [
                np.mean(dPhiBinContents[i : i + 5])
                for i in range(len(dPhiBinContents) - 5)
            ]
        )
        # get the average of the three minimum values
        minVal = np.mean([minVal1, minVal2, minVal3])
        # get the standard deviation of the three minimum values
        minValErr = np.std([minVal1, minVal2, minVal3])
        # return the minimum value and the systematic error
        return minVal, minValErr
    
    

    def get_pion_TPC_nSigma_inclusive(self, i,j,k,region=None):
        pion_TPC_signal = None
        offset = 1 if self.analysisType == "pp" else 0 
        for sparse_ind in range(len(self.JH)):  # type:ignore
            if region is not None:
                if region == "NS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSig[0], self.dEtaSig[1]
                    )  # deta
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )  # dphi

                elif region == "AS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSigAS[0], self.dEtaSigAS[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigAS[0], self.dPhiSigAS[1]
                    )
                if region == "BGHi":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGHi[0], self.dEtaBGHi[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )
                if region == "BGLo":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGLo[0], self.dEtaBGLo[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )
            if pion_TPC_signal is None:
                pion_TPC_signal = self.JH[sparse_ind].Projection(8-offset)  # type:ignore
            else:
                pion_TPC_signal.Add(self.JH[sparse_ind].Projection(8-offset))  # type:ignore

            self.JH[sparse_ind].GetAxis(3).SetRange(
                0, self.JH[sparse_ind].GetAxis(3).GetNbins() + 1
            )  # deta # type:ignore
            self.JH[sparse_ind].GetAxis(4).SetRange(
                0, self.JH[sparse_ind].GetAxis(4).GetNbins() + 1
            )  # dphi # type:ignore
        return pion_TPC_signal
    
    def get_pion_TPC_nSigma_inclusive_vs_dphi(self):
        pion_TPC_signal = None
        offset = 1 if self.analysisType == "pp" else 0 
        for sparse_ind in range(len(self.JH)):  # type:ignore
            if pion_TPC_signal is None:
                pion_TPC_signal = self.JH[sparse_ind].Projection(8-offset, 4)  # type:ignore
            else:
                pion_TPC_signal.Add(self.JH[sparse_ind].Projection(8-offset, 4))  # type:ignore
        return pion_TPC_signal

    def get_pion_TPC_nSigma(self, i,j,k,particleType, region=None):
        offset = 1 if self.analysisType == "pp" else 0  # type:ignore
        speciesID = (
            9
            if particleType == "pion"
            else 10
            if particleType == "proton"
            else 11
        )
        pion_TPC_nSigma = None
        for sparse_ind in range(len(self.JH)):  # type:ignore
            if region is not None:
                if region == "NS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSig[0], self.dEtaSig[1]
                    )  # deta
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )  # dphi

                elif region == "AS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSigAS[0], self.dEtaSigAS[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigAS[0], self.dPhiSigAS[1]
                    )
                if region == "BGHi":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGHi[0], self.dEtaBGHi[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )
                if region == "BGLo":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaBGLo[0], self.dEtaBGLo[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )

            if particleType == "pion":
                self.JH[sparse_ind].GetAxis(9 - offset).SetRangeUser(
                        -2 , 2
                    )  # type:ignore
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(11-offset).FindBin(-2 if j<2 else -1))
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2))
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        8 - offset
                    )  # type:ignore
                else:
                    pion_TPC_nSigma.Add(
                        self.JH[sparse_ind].Projection(8 - offset)
                    )  # type:ignore
                # self.JH[sparse_ind].GetAxis(11 - offset).SetRange(0,0)  # type:ignore
                # pion_TPC_nSigma.Add(
                #     self.JH[sparse_ind].Projection(8 - offset)
                # )  # type:ignore

                
            elif particleType == "proton":
                self.JH[sparse_ind].GetAxis(10 - offset).SetRangeUser(
                     -2 , 2
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        8 - offset
                    )  # type:ignore
                else:
                    pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset))


            elif particleType == "kaon":
                self.JH[sparse_ind].GetAxis(11 - offset).SetRangeUser(
                    -2 if j<2 else -1 , 2
                )  # type:ignore
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(0,self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2))
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        8 - offset
                    )  # type:ignore
                else:
                    pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset))

                # self.JH[sparse_ind].GetAxis(10-offset).SetRange(0,0)   
                # pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset))

            
            elif particleType == "other":
                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                     self.JH[sparse_ind].GetAxis(10 - offset).FindBin(2), self.JH[sparse_ind].GetAxis(10 - offset).GetNbins()+1
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        8 - offset
                    )  # type:ignore
                else:
                    pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset))

                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                    0, self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2)
                )  # type:ignore
                # k
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                     self.JH[sparse_ind].GetAxis(11 - offset).FindBin(2), self.JH[sparse_ind].GetAxis(11 - offset).GetNbins()+1
                )  # type:ignore
                pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset))

                # pi
                self.JH[sparse_ind].GetAxis(9 - offset).SetRange(
                    0, self.JH[sparse_ind].GetAxis(9-offset).FindBin(-2)
                )  # type:ignore
                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                     0,self.JH[sparse_ind].GetAxis(10 - offset).FindBin(-2)
                )  # type:ignore
                # k
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                     0, self.JH[sparse_ind].GetAxis(11 - offset).FindBin(-2)
                )  # type:ignore
                pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset))

                # pi
                self.JH[sparse_ind].GetAxis(9 - offset).SetRange(
                     self.JH[sparse_ind].GetAxis(9-offset).FindBin(2), self.JH[sparse_ind].GetAxis(9-offset).GetNbins()+1
                )  # type:ignore
                # p
                self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                     0, self.JH[sparse_ind].GetAxis(10-offset).FindBin(-2)
                )  # type:ignore
                # k
                self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                     0, self.JH[sparse_ind].GetAxis(11 - offset).FindBin(-2)
                )  # type:ignore
                pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset))




            self.JH[sparse_ind].GetAxis(9 - offset).SetRange(
                0,0
            )  # type:ignore
            self.JH[sparse_ind].GetAxis(10 - offset).SetRange(
                0,0
            )  # type:ignore
            self.JH[sparse_ind].GetAxis(11 - offset).SetRange(
                0,0
            )  # type:ignore


            self.JH[sparse_ind].GetAxis(3).SetRange(
                0, self.JH[sparse_ind].GetAxis(3).GetNbins() + 1
            )  # deta # type:ignore
            self.JH[sparse_ind].GetAxis(4).SetRange(
                0, self.JH[sparse_ind].GetAxis(4).GetNbins() + 1
            )  # dphi # type:ignore
        return pion_TPC_nSigma
    
    def get_pion_TPC_nSigma_vs_dphi(self, particleType, region=None):
        offset = 1 if self.analysisType == "pp" else 0  # type:ignore
        speciesID = (
            9
            if particleType == "pion"
            else 10
            if particleType == "proton"
            else 11
        )
        pion_TPC_nSigma = None
        for sparse_ind in range(len(self.JH)):  # type:ignore
            if region is not None:
                if region == "signalNS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSig[0], self.dEtaSig[1]
                    )  # deta
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigNS[0], self.dPhiSigNS[1]
                    )  # dphi

                elif region == "signalAS":
                    # restrict ourselves to the signal region
                    self.JH[sparse_ind].GetAxis(3).SetRangeUser(
                        self.dEtaSigAS[0], self.dEtaSigAS[1]
                    )
                    self.JH[sparse_ind].GetAxis(4).SetRangeUser(
                        self.dPhiSigAS[0], self.dPhiSigAS[1]
                    )
                if region == "background":
                    pass
            if particleType == "pion":
                self.JH[sparse_ind].GetAxis(9 - offset).SetRangeUser(
                     -2 if self.analysisType=='pp' else -2, 2 if self.analysisType=='pp' else 2
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        8 - offset, 4
                    )  # type:ignore
                else:
                    pion_TPC_nSigma.Add(
                        self.JH[sparse_ind].Projection(8 - offset, 4)
                    )  # type:ignore
                self.JH[sparse_ind].GetAxis(9 - offset).SetRangeUser(
                    -100, 100
                )  # type:ignore

            elif particleType == "proton":
                self.JH[sparse_ind].GetAxis(10 - offset).SetRangeUser(
                     -2 if self.analysisType=='pp' else -2, 2 if self.analysisType=='pp' else 2
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        8 - offset, 4
                    )  # type:ignore
                else:
                    pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset, 4))
                self.JH[sparse_ind].GetAxis(10 - offset).SetRangeUser(
                    -100, 100
                )  # type:ignore

            elif particleType == "kaon":
                self.JH[sparse_ind].GetAxis(11 - offset).SetRangeUser(
                     -2 if self.analysisType=='pp' else -2, 2 if self.analysisType=='pp' else 2
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        8 - offset, 4
                    )  # type:ignore
                else:
                    pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(8 - offset, 4))
                self.JH[sparse_ind].GetAxis(11 - offset).SetRangeUser(
                    -100, 100
                )  # type:ignore
            self.JH[sparse_ind].GetAxis(3).SetRange(
                0, self.JH[sparse_ind].GetAxis(3).GetNbins() + 1
            )  # deta # type:ignore
            self.JH[sparse_ind].GetAxis(4).SetRange(
                0, self.JH[sparse_ind].GetAxis(4).GetNbins() + 1
            )  # dphi # type:ignore
        return pion_TPC_nSigma

    def get_response_matrix(self, i,j,k,fit_params, fit_errors, fit_func, gauss, generalized_gauss, region):
        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = uncertainties.correlated_values(fit_params, fit_errors)

        int_x = np.linspace(-10, 10, 1000)
        int_y = fit_func(int_x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak)


        pionEnhNorm  = np.trapz(int_y[:1000], int_x)
        protonEnhNorm = np.trapz(int_y[1000:2000], int_x)
        kaonEnhNorm  = np.trapz(int_y[2000:3000], int_x)
        inclusiveNorm= np.trapz(int_y[3000:], int_x)

        gpp = np.trapz(generalized_gauss(int_x, mup, sigp, app, alphap), int_x)/protonEnhNorm
        gppi = np.trapz(gauss(int_x, mupi, sigpi, apip), int_x)/protonEnhNorm
        gpk = np.trapz(generalized_gauss(int_x, muk, sigk, akp, alphak), int_x)/protonEnhNorm

        gpip = np.trapz(generalized_gauss(int_x, mup, sigp, appi, alphap), int_x)/pionEnhNorm
        gpipi = np.trapz(gauss(int_x, mupi, sigpi, apipi), int_x)/pionEnhNorm
        gpik = np.trapz(generalized_gauss(int_x, muk, sigk, akpi, alphak), int_x)/pionEnhNorm

        gkp = np.trapz(generalized_gauss(int_x, mup, sigp, apk, alphap), int_x)/kaonEnhNorm
        gkpi = np.trapz(gauss(int_x, mupi, sigpi, apik), int_x)/kaonEnhNorm
        gkk = np.trapz(generalized_gauss(int_x, muk, sigk, akk, alphak), int_x)/kaonEnhNorm

        gincp = np.trapz(generalized_gauss(int_x, mup, sigp, apinc, alphap), int_x)/inclusiveNorm
        gincpi = np.trapz(gauss(int_x, mupi, sigpi, apiinc), int_x)/inclusiveNorm
        ginck = np.trapz(generalized_gauss(int_x, muk, sigk, akinc, alphak), int_x)/inclusiveNorm
        

        debug_logger.debug(f"{[[gpipi, gppi, gkpi], [gpip, gpp, gkp], [gpik, gpk, gkk]]=}")
        debug_logger.debug(f"{[gincpi, gincp, ginck]=}")

        if self.analysisType in ['central', 'semicentral']:
            A = np.array([
                [
                    (gpipi/gincpi)*self.N_assoc_for_species[region]['pion'][i,j,k], 
                    (gpip/gincp)*self.N_assoc_for_species[region]['pion'][i,j,k], 
                    (gpik/ginck)*self.N_assoc_for_species[region]['pion'][i,j,k]
                ],
                [
                    (gppi/gincpi)*self.N_assoc_for_species[region]['proton'][i,j,k], 
                    (gpp/gincp)*self.N_assoc_for_species[region]['proton'][i,j,k], 
                    (gpk/ginck)*self.N_assoc_for_species[region]['proton'][i,j,k]
                ],
                [
                    (gkpi/gincpi)*self.N_assoc_for_species[region]['kaon'][i,j,k], 
                    (gkp/gincp)*self.N_assoc_for_species[region]['kaon'][i,j,k], 
                    (gkk/ginck)*self.N_assoc_for_species[region]['kaon'][i,j,k]
                ],
                [
                    self.N_assoc[region][i,j,k]-(gpipi/gincpi)*self.N_assoc_for_species[region]['pion'][i,j,k]- (gppi/gincpi)*self.N_assoc_for_species[region]['proton'][i,j,k]- (gkpi/gincpi)*self.N_assoc_for_species[region]['kaon'][i,j,k], 
                    self.N_assoc[region][i,j,k]-(gpip/gincp)*self.N_assoc_for_species[region]['pion'][i,j,k]- (gpp/gincp)*self.N_assoc_for_species[region]['proton'][i,j,k]- (gkp/gincp)*self.N_assoc_for_species[region]['kaon'][i,j,k],
                    self.N_assoc[region][i,j,k]-(gpik/ginck)*self.N_assoc_for_species[region]['pion'][i,j,k]- (gpk/ginck)*self.N_assoc_for_species[region]['proton'][i,j,k]- (gkk/ginck)*self.N_assoc_for_species[region]['kaon'][i,j,k]
                ]
            ])/self.N_assoc[region][i,j,k]
        else:
            A = np.array([
                [
                    (gpipi/gincpi)*self.N_assoc_for_species[region]['pion'][i,j], 
                    (gpip/gincp)*self.N_assoc_for_species[region]['pion'][i,j], 
                    (gpik/ginck)*self.N_assoc_for_species[region]['pion'][i,j]
                ],
                [
                    (gppi/gincpi)*self.N_assoc_for_species[region]['proton'][i,j], 
                    (gpp/gincp)*self.N_assoc_for_species[region]['proton'][i,j], 
                    (gpk/ginck)*self.N_assoc_for_species[region]['proton'][i,j]
                ],
                [
                    (gkpi/gincpi)*self.N_assoc_for_species[region]['kaon'][i,j], 
                    (gkp/gincp)*self.N_assoc_for_species[region]['kaon'][i,j], 
                    (gkk/ginck)*self.N_assoc_for_species[region]['kaon'][i,j]
                ],
                [
                    self.N_assoc[region][i,j]-(gpipi/gincpi)*self.N_assoc_for_species[region]['pion'][i,j]- (gppi/gincpi)*self.N_assoc_for_species[region]['proton'][i,j]- (gkpi/gincpi)*self.N_assoc_for_species[region]['kaon'][i,j], 
                    self.N_assoc[region][i,j]-(gpip/gincp)*self.N_assoc_for_species[region]['pion'][i,j]- (gpp/gincp)*self.N_assoc_for_species[region]['proton'][i,j]- (gkp/gincp)*self.N_assoc_for_species[region]['kaon'][i,j],
                    self.N_assoc[region][i,j]-(gpik/ginck)*self.N_assoc_for_species[region]['pion'][i,j]- (gpk/ginck)*self.N_assoc_for_species[region]['proton'][i,j]- (gkk/ginck)*self.N_assoc_for_species[region]['kaon'][i,j]
                ]
            ])/self.N_assoc[region][i,j]

        return A