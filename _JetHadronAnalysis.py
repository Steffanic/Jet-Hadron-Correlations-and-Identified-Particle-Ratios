from math import erf
from ROOT import TH1D, TH2D, TH3D
import numpy as np


def print_function_name_with_description_on_call(description):
    """
    Prints the name of the function and a description of what it does
    """

    def function_wrapper(function):
        def method_wrapper(self, *args, **kwargs):
            print(f"{function.__name__} in {self.__class__.__name__}:\n\t{description}")
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
                TH2Corr.Sumw2()
            else:
                success = TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore
                # divide by the bin widths to get the correct correlation function
                if not success:
                    warn("Failed to add to TH2Corr")
        TH2Corr.Scale(1 / bin_widths[2] / bin_widths[3])
        self.get_SE_correlation_function_result = TH2Corr
        self.get_SE_correlation_function_has_changed = False
        return TH2Corr.Clone()

    def get_SE_correlation_function_for_species(self, species):
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
            self.JH[sparse_ind].GetAxis(speciesID - offset).SetRangeUser(
                -2, 2
            )  # type:ignore
            if TH2Corr is None:
                TH2Corr = self.JH[sparse_ind].Projection(4, 3)  # type:ignore
                TH2Corr.Sumw2()
                
            else:
                TH2Corr.Add(self.JH[sparse_ind].Projection(4, 3))  # type:ignore
            self.JH[sparse_ind].GetAxis(speciesID - offset).SetRangeUser(
                -10, 10
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
                -2, 2
            )  # type:ignore
            if TH3Corr is None:
                TH3Corr = self.JH[sparse_ind].Projection(
                    3, 4, 7 - offset
                )  # type:ignore
                TH3Corr.Sumw2()
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
                TH2Corr.Sumw2()
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
        sliding_window_norm = self.ME_norm_sliding_window()
        # using a sliding window average with window size pi/2
        sliding_window_norm_half_pi = self.ME_norm_sliding_window(windowSize=np.pi / 2)
        # using a sliding window average with window size pi/4
        sliding_window_norm_quarter_pi = self.ME_norm_sliding_window(windowSize=np.pi/4)
        # using a sliding window average with window size pi/6
        sliding_window_norm_sixth_pi = self.ME_norm_sliding_window(windowSize=np.pi/6)
        # using a sliding window average with window size pi/12
        sliding_window_norm_twelfth_pi = self.ME_norm_sliding_window(windowSize=np.pi/12)
        # by taking the max
        max_norm = self.ME_max()
        print(f"{sliding_window_norm=}, {max_norm=}, {sliding_window_norm_half_pi}")
        # take the average of the three
        norm = (sliding_window_norm_half_pi+sliding_window_norm_quarter_pi)/2
        # compute the error on the normalization constant
        # using the standard deviation of the three values
        error = np.abs(max_norm-sliding_window_norm)
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

        del dPhiME
        del fhnMixedEventsCorr

        return maxWindowAve

    def get_acceptance_corrected_correlation_function(self, in_z_vertex_bins=False):

        if in_z_vertex_bins:
            # set the user range for JH, ME, and Trigger sparses to be the same as the z vertex bins
            AccCorr = self.get_SE_correlation_function().Clone()
            AccCorr.Reset()
            for z_bin_no in range(len(self.z_vertex_bins)-1):
                self.set_z_vertex_bin(z_bin_no)
                if self.get_ME_correlation_function().Integral() == 0:
                    print(f"Skipping z bin {z_bin_no}")
                    continue
                Corr = self.get_SE_correlation_function()
                NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
                Corr.Divide(NormMEC)
                AccCorr.Add(Corr)
            AccCorr.Scale(1/len(self.z_vertex_bins))
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


    def get_acceptance_corrected_correlation_function_for_species(self, i,j, k,species, in_z_vertex_bins=False):

        if in_z_vertex_bins:
            # set the user range for JH, ME, and Trigger sparses to be the same as the z vertex bins
            pionEnhCorr = self.get_SE_correlation_function_for_species(species).Clone()
            kaonEnhCorr = self.get_SE_correlation_function_for_species(species).Clone()
            protonEnhCorr = self.get_SE_correlation_function_for_species(species).Clone()
            pionEnhCorr.Reset()
            kaonEnhCorr.Reset()
            protonEnhCorr.Reset()
            for z_bin_no in range(len(self.z_vertex_bins)-1):
                self.set_z_vertex_bin(z_bin_no)
                if self.get_ME_correlation_function().Integral() == 0:
                    print(f"Skipping z bin {z_bin_no}")
                    continue
                Corrs_raw = [self.get_SE_correlation_function_for_species(enh_species) for enh_species in ["pion", "proton", "kaon"]]
                piEnhCorr = Corrs_raw[0]
                pEnhCorr = Corrs_raw[1]
                kEnhCorr = Corrs_raw[2]
                if self.analysisType in ['central', 'semicentral']:
                    piEnhCorr.Scale(1/self.N_assoc_for_species['pion'][i,j,k])
                    pEnhCorr.Scale(1/self.N_assoc_for_species['proton'][i,j,k])
                    kEnhCorr.Scale(1/self.N_assoc_for_species['kaon'][i,j,k])
                else:
                    pEnhCorr.Scale(1/self.N_assoc_for_species['pion'][i,j])
                    pEnhCorr.Scale(1/self.N_assoc_for_species['proton'][i,j])
                    kEnhCorr.Scale(1/self.N_assoc_for_species['kaon'][i,j])

                NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
                piEnhCorr.Divide(NormMEC)
                pEnhCorr.Divide(NormMEC)
                kEnhCorr.Divide(NormMEC)
                pionEnhCorr.Add(piEnhCorr)
                kaonEnhCorr.Add(kEnhCorr)
                protonEnhCorr.Add(pEnhCorr)
            pionEnhCorr.Scale(1/len(self.z_vertex_bins))
            kaonEnhCorr.Scale(1/len(self.z_vertex_bins))
            protonEnhCorr.Scale(1/len(self.z_vertex_bins))
            self.reset_z_vertex_bin()

        else:
            # get the raw correlation functions for every species and use the PionTPCNSigmaFitObjs to get the mixing factors
            Corrs_raw = [self.get_SE_correlation_function_for_species(enh_species) for enh_species in ["pion", "proton", "kaon"]]
            if self.analysisType in ['central', 'semicentral']:
                assert(self.N_assoc_for_species[species][i,j,k]!=0, f"{self.N_assoc_for_species['pion'][i,j,k]=}")
                assert(self.N_assoc_for_species[species][i,j,k]!=0, f"{self.N_assoc_for_species['proton'][i,j,k]=}")
                assert(self.N_assoc_for_species[species][i,j,k]!=0, f"{self.N_assoc_for_species['kaon'][i,j,k]=}")
            else:
                assert(self.N_assoc_for_species[species][i,j]!=0, f"{self.N_assoc_for_species['pion'][i,j]=}")
                assert(self.N_assoc_for_species[species][i,j]!=0, f"{self.N_assoc_for_species['proton'][i,j]=}")
                assert(self.N_assoc_for_species[species][i,j]!=0, f"{self.N_assoc_for_species['kaon'][i,j]=}")
            pionEnhCorr = Corrs_raw[0]
            protonEnhCorr = Corrs_raw[1]
            kaonEnhCorr = Corrs_raw[2]
            # now scale each by nassoc 
            if self.analysisType in ['central', 'semicentral']:
                pionEnhCorr.Scale(1/self.N_assoc_for_species['pion'][i,j,k])
                protonEnhCorr.Scale(1/self.N_assoc_for_species['proton'][i,j,k])
                kaonEnhCorr.Scale(1/self.N_assoc_for_species['kaon'][i,j,k])
            else:
                pionEnhCorr.Scale(1/self.N_assoc_for_species['pion'][i,j])
                protonEnhCorr.Scale(1/self.N_assoc_for_species['proton'][i,j])
                kaonEnhCorr.Scale(1/self.N_assoc_for_species['kaon'][i,j])
            # get normalized mixed event correlation function
            NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
            # divide the raw correlation function by the mixed event correlation function
            pionEnhCorr.Divide(NormMEC)
            protonEnhCorr.Divide(NormMEC)
            kaonEnhCorr.Divide(NormMEC)
        # Get Mixing factors 
        if self.analysisType in ['central', 'semicentral']:
            fit_params = self.PionTPCNSigmaFitObjs[i,j, k].popt
        else:
            fit_params = self.PionTPCNSigmaFitObjs[i,j].popt
        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk = fit_params

        protonEnhNorm = -(np.pi/2)**.5*(
            akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
            app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
            apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
            )
        pionEnhNorm = -(np.pi/2)**.5*(
            akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
            appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
            apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
            )
        kaonEnhNorm = -(np.pi/2)**.5*(
            apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
            apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
            akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
            )
        
        fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
        fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
        fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
        fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
        fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
        fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
        fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
        fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
        fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm

        
        determinant_factor = fkpi*fpip*fpk - fkp*fpipi*fpk - fkpi*fpik*fpp + fkk*fpipi*fpp + fkp*fpik*fppi - fkk*fpip*fppi
        print(f"{fpp=}, {fpip=}, {fkp=}, {fppi=}, {fpipi=}, {fkpi=}, {fpk=}, {fpik=}, {fkk=}")
        if species == 'pion':
            pionEnhCorr.Scale((-fkp*fpk+fkk*fpp)/determinant_factor)
            protonEnhCorr.Scale((fkpi*fpk-fppi*fkk)/determinant_factor)
            kaonEnhCorr.Scale((-fkpi*fpp+fppi*fkp)/determinant_factor)
            Corr = pionEnhCorr.Clone()
            Corr.Add(protonEnhCorr)
            Corr.Add(kaonEnhCorr)

        elif species == 'proton':
            pionEnhCorr.Scale((fkp*fpik-fkk*fpip)/determinant_factor)
            protonEnhCorr.Scale((-fkpi*fpik+fpipi*fkk)/determinant_factor)
            kaonEnhCorr.Scale((fkpi*fpip-fpipi*fkp)/determinant_factor)
            Corr = pionEnhCorr.Clone()
            Corr.Add(protonEnhCorr)
            Corr.Add(kaonEnhCorr)
        
        elif species == 'kaon':
            pionEnhCorr.Scale((fpip*fpk-fpik*fpp)/determinant_factor)
            protonEnhCorr.Scale((-fpipi*fpk+fpik*fppi)/determinant_factor)
            kaonEnhCorr.Scale((fpipi*fpp-fppi*fpip)/determinant_factor)
            Corr = pionEnhCorr.Clone()
            Corr.Add(protonEnhCorr)
            Corr.Add(kaonEnhCorr)
        # Corr.GetXaxis().SetRangeUser(-0.9, 0.9)
        del NormMEC
        # scale up the correlation function by the normalization factor
        if self.analysisType in ['central', 'semicentral']:
            Corr.Scale(self.N_assoc[i,j,k])
        else:
            Corr.Scale(self.N_assoc[i,j])
        return Corr.Clone()

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
    
    def get_N_assoc(self):
        """
        Returns the number of trigger particles in the event
        Assume the Trigger sparse being passed has already been limited to the jet pT bin in question
        """
        N_assoc_val = 0
        N_assoc_hist = None
        for sparse_ind in range(len(self.JH)):
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
        return N_assoc_val
    
    def get_N_assoc_for_species(self,species):
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
            self.JH[sparse_ind].GetAxis(speciesID-offset).SetRangeUser(-2, 2)   
            if N_assoc_hist is None:
                N_assoc_hist = self.JH[sparse_ind].Projection(
                    2
                )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
            else:
                N_assoc_hist.Add(
                    self.JH[sparse_ind].Projection(2)
                )  # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will
            self.JH[sparse_ind].GetAxis(speciesID-offset).SetRangeUser(-10, 10)   

        N_assoc = N_assoc_hist.GetEntries()  # type:ignore
        del N_assoc_hist
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
        dPhi.Sumw2()
        nbins = upperBin - lowerBin
        binwidth = tempHist.GetXaxis().GetBinWidth(1)
        dPhi.Scale(binwidth) # scale by the number of bins in the dEta range to get the correct normalization
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(
                -0.6
            )
            print(f"Grabbing between bins {upperBin} and {lowerBin}")
            dPhi.Scale((nbins_sig + 1) / (nbins + 1))
        # rebin dPhi
        if rebin:
            dPhi = dPhi.Rebin(2)
            dPhi.Scale(1 / 2)
        return dPhi.Clone(), nbins, binwidth

    def get_dPhi_projection_in_dEta_dpionTPCnSigma_range(
        self, i,j,k,dEtaRange, TOFcutSpecies, rebin=True, scaleUp=False, in_z_vertex_bins=False
    ):
        """
        Returns the dPhi distribution for a given dEta range
        """
        tempHist = self.get_acceptance_corrected_correlation_function_for_species(i,j,k,
            TOFcutSpecies, in_z_vertex_bins=in_z_vertex_bins
        )
        lowerBin, upperBin = (tempHist.GetXaxis().FindBin(bin) for bin in dEtaRange)
        tempHist.GetXaxis().SetRange(lowerBin, upperBin)
        dPhi = tempHist.ProjectionY()
        dPhi.Sumw2()
        nbins = upperBin - lowerBin
        binwidth = tempHist.GetXaxis().GetBinWidth(1)
        dPhi.Scale(binwidth) # scale by the number of bins in the dEta range to get the correct normalization
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(
                -0.6
            )
            print(f"Grabbing between bins {upperBin} and {lowerBin}")
            dPhi.Scale((nbins_sig + 1) / (nbins + 1))
        # rebin dPhi
        if rebin:
            dPhi = dPhi.Rebin(2)
            dPhi.Scale(0.5)
        return dPhi.Clone(), nbins, binwidth

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
        dPhidpionTPCnSigma.Sumw2()
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            binwidth = tempHist.GetXaxis().GetBinWidth(1)
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(
                -0.6
            )
            print(f"Grabbing between bins {upperBin} and {lowerBin}")
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
                            x_vals[l], *self.RPFObjs[ZVi, j].popt
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
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs, axis=1)) / 3
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
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs, axis=1)) / 3
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
                dPhiSig = self.dPhiSigcorrsZV[i, j, k]
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
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs, axis=1)) 
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

    def get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_species(
        self, i, j, k, species, in_z_vertex_bins=False
    ):
        """
        Returns the background subtracted dPhi dpionTPCnSigma distribution after a TOF species cut for the signal region
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsForSpeciesZV[species][i, j, k]
            else:
                dPhiSig = self.dPhiSigcorrsForSpecies[species][i, j, k]  # type:ignore
        elif self.analysisType == "pp":  # type:ignore
            # get the dPhi distribution
            if in_z_vertex_bins:
                dPhiSig = self.dPhiSigcorrsForSpeciesZV[species][i, j]
                dPhiBG = self.dPhiBGcorrsForSpeciesZV[species][i, j]
            else:
                dPhiSig = self.dPhiSigcorrsForSpecies[species][i, j]  # type:ignore
                dPhiBG = self.dPhiBGcorrsForSpecies[species][i, j]  # type:ignore

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
                        self.RPFObjsForSpeciesZV[species][i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsForSpeciesZV[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsForSpeciesZV[species][i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsForSpeciesZV[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            else:
                dPhiBGRPFs = np.array(
                    [
                        self.RPFObjsForSpecies[species][i, j].simultaneous_fit(
                            x_vals[l], *self.RPFObjsForSpecies[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
                dPhiBGRPFErrs = np.array(
                    [
                        self.RPFObjsForSpecies[species][i, j].simultaneous_fit_err(
                            x_vals[l], x_vals[1] - x_vals[0], *self.RPFObjsForSpecies[species][i, j].popt
                        )
                        for l in range(len(x_vals))
                    ]
                )  # type:ignore
            if k in [0, 1, 2]:
                dPhiBGRPF = dPhiBGRPFs[:, k]  # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:, k]  # type:ignore

            if k == 3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs, axis=1)) 
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

    def get_normalized_background_subtracted_dPhi_for_species(self, i, j, k, species, in_z_vertex_bins=False):
        """
        Returns the normalized background subtracted dPhi distribution for a given species
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the background subtracted dPhi distribution for a given species
            BGSubtracteddPhiSigForSpeciesHist = (
                self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_species(
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
            ) = self.get_normalized_BG_subtracted_AccCorrectedSEdPhiSig_for_species(
                i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
            )
            # return the background subtracted dPhi distribution
            return BGSubtracteddPhiSigForSpeciesHist.Clone(), minValErr

    def get_yield_for_species(self, i, j, k, species, region=None, in_z_vertex_bins=False):
        """
        Returns the yield for a given species
        """
        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            BGSubtracteddPhiSigHist = (
                self.get_normalized_background_subtracted_dPhi_for_species(
                    i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
                )
            )
        elif self.analysisType == "pp":  # type:ignore
            # get the normalized background subtracted dPhi distribution for a given species
            (
                BGSubtracteddPhiSigHist,
                minValErr,
            ) = self.get_normalized_background_subtracted_dPhi_for_species(
                i, j, k, species, in_z_vertex_bins=in_z_vertex_bins
            )  # type:ignore
        # get the yield for a given species
        bin_width = BGSubtracteddPhiSigHist.GetBinWidth(1)
        if region is None:
            yield_ = np.sum(
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width
                for l in range(1, BGSubtracteddPhiSigHist.GetNbinsX() + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width)
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
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width)
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
                BGSubtracteddPhiSigHist.GetBinContent(l) * bin_width
                for l in range(low_bin, high_bin + 1)
            )
            yield_err_ = np.sqrt(
                np.sum(
                    (BGSubtracteddPhiSigHist.GetBinError(l) * bin_width)
                    ** 2
                    for l in range(low_bin, high_bin + 1)
                )
            )
            # return the yield
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
    
    

    def get_pion_TPC_signal(self):
        pion_TPC_signal = None
        for sparse_ind in range(len(self.JH)):  # type:ignore
            if pion_TPC_signal is None:
                pion_TPC_signal = self.JH[sparse_ind].Projection(7)  # type:ignore
                pion_TPC_signal.Sumw2()
            else:
                pion_TPC_signal.Add(self.JH[sparse_ind].Projection(7))  # type:ignore
        return pion_TPC_signal

    def get_pion_TPC_nSigma(self, particleType, region=None):
        offset = 1 if self.analysisType == "pp" else 0  # type:ignore
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
                self.JH[sparse_ind].GetAxis(8 - offset).SetRangeUser(
                    -2, 2
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        7 - offset
                    )  # type:ignore
                    pion_TPC_nSigma.Sumw2()
                else:
                    pion_TPC_nSigma.Add(
                        self.JH[sparse_ind].Projection(7 - offset)
                    )  # type:ignore
                self.JH[sparse_ind].GetAxis(8 - offset).SetRangeUser(
                    -100, 100
                )  # type:ignore
            elif particleType == "proton":
                self.JH[sparse_ind].GetAxis(9 - offset).SetRangeUser(
                    -2, 2
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        7 - offset
                    )  # type:ignore
                    pion_TPC_nSigma.Sumw2()
                else:
                    pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(7 - offset))
                self.JH[sparse_ind].GetAxis(9 - offset).SetRangeUser(
                    -100, 100
                )  # type:ignore
            elif particleType == "kaon":
                self.JH[sparse_ind].GetAxis(10 - offset).SetRangeUser(
                    -2, 2
                )  # type:ignore
                if pion_TPC_nSigma is None:
                    pion_TPC_nSigma = self.JH[sparse_ind].Projection(
                        7 - offset
                    )  # type:ignore
                    pion_TPC_nSigma.Sumw2()
                else:
                    pion_TPC_nSigma.Add(self.JH[sparse_ind].Projection(7 - offset))
                self.JH[sparse_ind].GetAxis(10 - offset).SetRangeUser(
                    -100, 100
                )  # type:ignore
            self.JH[sparse_ind].GetAxis(3).SetRange(
                0, self.JH[sparse_ind].GetAxis(3).GetNbins() + 1
            )  # deta # type:ignore
            self.JH[sparse_ind].GetAxis(4).SetRange(
                0, self.JH[sparse_ind].GetAxis(4).GetNbins() + 1
            )  # dphi # type:ignore
        return pion_TPC_nSigma
