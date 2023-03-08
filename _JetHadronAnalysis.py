from ROOT import TH1D, TH2D, TH3D
import numpy as np

class AnalysisMixin:
    def get_SE_correlation_function(self, includePion=False):
        '''
        Returns TH2 containing the △eta, △phi distribution from the JH sparse
        or TH3 containing the △eta, △phi, △pion distribution from the JH sparse
        '''
        if includePion:
            if self.get_SE_correlation_function_w_Pion_has_changed or self.get_SE_correlation_function_w_Pion_result is None:
                TH3Corr = self.JH.Projection(3,4,7) #type:ignore
                self.get_SE_correlation_function_w_Pion_result = TH3Corr
                self.get_SE_correlation_function_w_Pion_has_changed = False
                TH3Corr.Sumw2()
                return TH3Corr
            else:
                return self.get_SE_correlation_function_w_Pion_result    
        if self.get_SE_correlation_function_has_changed or self.get_SE_correlation_function_result is None:
            
            TH2Corr = self.JH.Projection(4,3) #type:ignore
            self.get_SE_correlation_function_result = TH2Corr
            self.get_SE_correlation_function_has_changed = False
            TH2Corr.Sumw2()
            return TH2Corr
        else:
            return self.get_SE_correlation_function_result
    def get_ME_correlation_function(self):
        '''
        Rrturns TH2 containing the △eta, △phi distribution from the Mixed Event sparse
        '''
        if self.get_ME_correlation_function_has_changed or self.get_ME_correlation_function_result is None:
            TH2Corr = self.MixedEvent.Projection(4,3) #type:ignore
            self.get_ME_correlation_function_result = TH2Corr
            self.get_ME_correlation_function_has_changed = False
            TH2Corr.Sumw2()
            return TH2Corr
        else:
            return self.get_ME_correlation_function_result
    def get_normalized_ME_correlation_function(self):
        if self.get_normalized_ME_correlation_function_has_changed or self.get_normalized_ME_correlation_function_result is None:
            norm, error = self.get_ME_norm_and_systematic_error()
            a0 = 1/norm
            NormMEC = self.get_ME_correlation_function()
            NormMEC.Scale(a0)
            self.get_normalized_ME_correlation_function_result = NormMEC
            self.get_normalized_ME_correlation_function_has_changed = False
            self.get_ME_normalization_error = error
            self.get_ME_normalization_error_has_changed = False
            return NormMEC, error
        else:
            return self.get_normalized_ME_correlation_function_result, self.get_ME_normalization_error
        
    def get_ME_norm_and_systematic_error(self):
        # compute the normalization constant for the mixed event correlation function
        # using a sliding window average with window size pi
        sliding_window_norm = self.ME_norm_sliding_window()
        # using a sliding window average with window size pi/2
        sliding_window_norm_half_pi = self.ME_norm_sliding_window(windowSize = np.pi/2)
        # by taking the max 
        max_norm = self.ME_norm_sliding_window(windowSize = 2*np.pi)
        # take the average of the three
        norm = (sliding_window_norm + sliding_window_norm_half_pi + max_norm)/3
        # compute the error on the normalization constant
        # using the standard deviation of the three values
        error = np.std([sliding_window_norm, sliding_window_norm_half_pi, max_norm])
        return norm, error


    def ME_norm_sliding_window(self, windowSize = np.pi, etaRestriction = 0.3):
        '''
        Returns normalization constant for mixed event correlation function 

        Restricts |△eta|<etaRestriction and projects onto △phi 
        Using a sliding window average, find the highest average and call it the max
        '''
        if self.ME_norm_sliding_window_has_changed or self.ME_norm_sliding_window_result is None:
            fhnMixedEventsCorr = self.get_ME_correlation_function() # TH2
            fhnMixedEventsCorr.GetXaxis().SetRangeUser(-etaRestriction, etaRestriction)
            # get  eta bin width 
            eta_bin_width = fhnMixedEventsCorr.GetXaxis().GetBinWidth(1)
            # get number of bins in eta between +- etaRestriction
            eta_bins = fhnMixedEventsCorr.GetXaxis().FindBin(etaRestriction)-fhnMixedEventsCorr.GetXaxis().FindBin(-etaRestriction)

            dPhiME = fhnMixedEventsCorr.ProjectionY()
            dPhiME.Scale(1/(eta_bins))
            # Calculate moving max of the mixed event correlation function with window size pi 
            maxWindowAve = 0
            # get bin width 
            binWidth = dPhiME.GetBinWidth(1)
            # get number of bins
            nBinsPerWindow = int(windowSize/binWidth)
            for i in range(dPhiME.GetNbinsX()-int(windowSize//binWidth)):
                # get the average of the window for this set of bins 
                windowAve = dPhiME.Integral(i,i+int(windowSize//binWidth))/(nBinsPerWindow+1)
                # if the average is greater than the current max, set the max to the average
                if windowAve > maxWindowAve:
                    maxWindowAve = windowAve

          
            
            del dPhiME
            del fhnMixedEventsCorr
            self.ME_norm_sliding_window_result = maxWindowAve
            self.ME_norm_sliding_window_has_changed = False

            return maxWindowAve
        else:
            return self.ME_norm_sliding_window_result

    def get_acceptance_corrected_correlation_function(self):
        if self.get_acceptance_corrected_correlation_function_has_changed or self.get_acceptance_corrected_correlation_function_result is None:
            # get the raw correlation functions
            Corr = self.get_SE_correlation_function()
            
            # get normalized mixed event correlation function
            NormMEC, norm_systematic = self.get_normalized_ME_correlation_function()
            # divide the raw correlation function by the mixed event correlation function
            Corr.Divide(NormMEC)
            #Corr.GetXaxis().SetRangeUser(-0.9, 0.9)
            del NormMEC
            self.get_acceptance_corrected_correlation_function_result = Corr
            self.get_acceptance_corrected_correlation_function_has_changed = False
            return Corr.Clone()
        else:
            return self.get_acceptance_corrected_correlation_function_result.Clone()

    def get_acceptance_corrected_dPhi_dEta_dPion_distribution(self):
        if self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed or self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_result is None:
            CorrdPion = self.get_SE_correlation_function(includePion=True)
            
            MEdPion, norm_systematic = self.get_normalized_ME_correlation_function()

            # Now make a new TH3D like CorrdPion and fill the bins with MEdPion 
            TH3DNorm = TH3D(CorrdPion)
            TH3DNorm.Reset()
            for x in range(1, TH3DNorm.GetNbinsX()+1):
                for y in range(1,TH3DNorm.GetNbinsY()+1):
                    for z in range(1, TH3DNorm.GetNbinsZ()+1):
                        TH3DNorm.SetBinContent(x, y, z, MEdPion.GetBinContent(x, y))
                        TH3DNorm.SetBinError(x, y, z, MEdPion.GetBinError(x, y))
            TH3DNorm.Sumw2()
            CorrdPion.Divide(TH3DNorm)
            del TH3DNorm
            self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_result = CorrdPion
            self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_has_changed = False
            CorrdPion.Sumw2()
            return CorrdPion
        else:
            return self.get_acceptance_corrected_dPhi_dEta_dPion_distribution_result

    def get_N_trig(self):
        '''
        Returns the number of trigger particles in the event
        Assume the Trigger sparse being passed has already been limited to the jet pT bin in question
        '''
        
        N_trig = self.Trigger.Projection(1) # type:ignore the projection wont count the entries that are outside of the SetRangeUser, whereas the THnSparse will  
        N_trig_ = N_trig.GetEntries()
        del N_trig
        return N_trig_

    def get_normalized_acceptance_corrected_correlation_function(self):
        '''
        Returns the acceptance corrected correlation function normalized by the number of trigger particles
        '''
        if self.get_normalized_acceptance_corrected_correlation_function_has_changed or self.get_normalized_acceptance_corrected_correlation_function_result is None:    
            N_trig = self.get_N_trig()
            AccCorrectedSEcorr = self.get_acceptance_corrected_correlation_function()
            AccCorrectedSEcorr.Scale(1/N_trig)
            del N_trig
            self.get_normalized_acceptance_corrected_correlation_function_result = AccCorrectedSEcorr
            self.get_normalized_acceptance_corrected_correlation_function_has_changed = False
            return AccCorrectedSEcorr.Clone()
        else:
            return self.get_normalized_acceptance_corrected_correlation_function_result.Clone()

    def get_dPhi_projection_in_dEta_range(self, dEtaRange, rebin=True, scaleUp=False):
        '''
        Returns the dPhi distribution for a given dEta range
        '''
        tempHist = self.get_acceptance_corrected_correlation_function()
        lowerBin, upperBin= (tempHist.GetXaxis().FindBin(bin) for bin in dEtaRange)
        tempHist.GetXaxis().SetRange(lowerBin, upperBin)
        dPhi = tempHist.ProjectionY()
        dPhi.Sumw2()
        if scaleUp:
            # Scale background up
            nbins = upperBin - lowerBin
            binwidth = tempHist.GetXaxis().GetBinWidth(1)
            nbins_sig = tempHist.GetXaxis().FindBin(0.6) - tempHist.GetXaxis().FindBin(-0.6)
            print(f"Grabbing between bins {upperBin} and {lowerBin}")
            dPhi.Scale((nbins_sig+1)/(nbins+1))
        # rebin dPhi 
        if rebin:
            dPhi = dPhi.Rebin(2)
        del tempHist
        return dPhi.Clone()

    def get_dPhi_dPion_projection_in_dEta_range(self, dEtaRange, rebin=True, scaleUp=False):
        '''
        Returns the dPhi distribution for a given dEta range
        '''
        tempHist = self.get_acceptance_corrected_dPhi_dEta_dPion_distribution().Clone()
        lowerBin, upperBin= (tempHist.GetXaxis().FindBin(bin) for bin in dEtaRange)
        tempHist.GetXaxis().SetRange(lowerBin, upperBin)
        dPhidPion = tempHist.Project3D("zy")
        dPhidPion.Sumw2()
        dPhidPion.Scale(1/(upperBin-lowerBin+1))
        
        if rebin:
            dPhidPion.RebinX(2)
            dPhidPion.RebinY(2)
        del tempHist
        return dPhidPion

    def get_dEta_projection_NS(self):
        '''
        Returns the dPhi distribution for a given dEta range
        '''
        tempHist = self.get_acceptance_corrected_correlation_function()
        tempHist.GetYaxis().SetRangeUser(-np.pi/2, np.pi/2)
        dEta = tempHist.ProjectionX().Clone()
        del tempHist
        return dEta

    def get_BG_subtracted_AccCorrectedSEdPhiSigNScorr(self, i,j,k):
        '''
        Returns the background subtracted dPhi distribution for the signal region
        '''
        if self.analysisType in ["central", "semicentral"]: # type:ignore
            # get the dPhi distribution for the signal region
            dPhiSig = self.dPhiSigcorrs[i,j,k] # type:ignore
        elif self.analysisType =="pp": #type:ignore
            dPhiSig = self.dPhiSigcorrs[i,j] #type:ignore
        # get the bin contents 
        dPhiSigBinContents = self.get_bin_contents_as_array(dPhiSig, forFitting=True) # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(dPhiSig, forFitting=True) # type:ignore
        # get x range 
        x_vals = self.get_bin_centers_as_array(dPhiSig, forFitting=True) # type:ignore
        if self.analysisType in ["central", "semicentral"]: # type:ignore
            # generate the background distribution from the RPFObj 

            dPhiBGRPFs = np.array([self.RPFObjs[i,j].simultaneous_fit(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))]) # type:ignore
            dPhiBGRPFErrs = np.array([self.RPFObjs[i,j].simultaneous_fit_err(x_vals[l],x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))]) # type:ignore
            if k in [0,1,2]:
                dPhiBGRPF = dPhiBGRPFs[:,k] # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:,k] # type:ignore
        
            if k==3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs, axis=1))/3
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0,1,2,3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")
            # subtract the background from the signal 
            BGSubtracteddPhiSig = dPhiSigBinContents- dPhiBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D("BGSubtracteddPhiSigNSHist", "BGSubtracteddPhiSigNSHist", len(x_vals), x_vals[0], x_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l+1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l+1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone()
        
        elif self.analysisType == "pp": # type:ignore
            # get the minimum value of the signal
            minVal = np.min(dPhiSigBinContents)
            # subtract the minimum value from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2 + minVal**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D("BGSubtracteddPhiSigNSHist", "BGSubtracteddPhiSigNSHist", len(x_vals), x_vals[0], x_vals[-1])
            # fill the histogram with the background subtracted signal 
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l+1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l+1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone(), minVal
        

    
    def get_BG_subtracted_AccCorrectedSEdPhiSigAScorr(self, i,j,k):
        '''
        Returns the background subtracted dPhi distribution for the signal region
        '''
        # get the dPhi distribution for the signal region
        if self.analysisType in ["central", "semicentral"]: # type:ignore
            dPhiSig = self.dPhiSigcorrs[i,j,k] # type:ignore
        elif self.analysisType == "pp":# type:ignore
            dPhiSig = self.dPhiSigcorrs[i,j] # type:ignore
        #dPhiSig.Scale(0.5) # to put it on the same level as the background
        # get the bin contents 
        dPhiSigBinContents = self.get_bin_contents_as_array(dPhiSig, forFitting=False, forAS=True) # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(dPhiSig, forFitting=False, forAS=True) # type:ignore
        # get x range 
        x_vals = self.get_bin_centers_as_array(dPhiSig, forFitting=False, forAS=True) # type:ignore
        if self.analysisType in ["central", "semicentral"]: # type:ignore
            # generate the background distribution from the RPFObj 
            dPhiBGRPFs = np.array([self.RPFObjs[i,j].simultaneous_fit(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))]) # type:ignore
            dPhiBGRPFErrs = np.array([self.RPFObjs[i,j].simultaneous_fit_err(x_vals[l],x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))]) # type:ignore
            if k in [0,1,2]:
                dPhiBGRPF = dPhiBGRPFs[:,k] # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:,k] # type:ignore
            if k==3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs, axis=1))/3
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0,1,2,3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")

            # subtract the background from the signal 
            BGSubtracteddPhiSig = dPhiSigBinContents- dPhiBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D("BGSubtracteddPhiSigASHist", "BGSubtracteddPhiSigASHist", len(x_vals), x_vals[0], x_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l+1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l+1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone()
        elif self.analysisType == "pp":# type:ignore
            # compute the minimum value of dPhiSigBinContents
            minVal, minValErr = self.get_minVal_and_systematic_error(dPhiSigBinContents)
            # subtract the minimum value from the signal 
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal 
            # Get the error on the background 
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2 )
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D("BGSubtracteddPhiSigASHist", "BGSubtracteddPhiSigASHist", len(x_vals), x_vals[0], x_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l+1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l+1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            return BGSubtracteddPhiSigHist.Clone(), minValErr
            
    def get_normalized_BG_subtracted_AccCorrectedSEdPhiSigcorr(self, i,j,k):
        '''
        Returns the background subtracted dPhi distribution for the signal region
        '''
        if self.analysisType in ["central", "semicentral"]: # type:ignore
            # get the dPhi distribution
            dPhiSig = self.dPhiSigcorrs[i,j,k] # type:ignore
        elif self.analysisType =="pp": #type:ignore
            dPhiSig = self.dPhiSigcorrs[i,j] #type:ignore

      
        # get the bin contents 
        dPhiSigBinContents = self.get_bin_contents_as_array(dPhiSig, forFitting=False) # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_bin_errors_as_array(dPhiSig, forFitting=False) # type:ignore
        # get x range 
        x_vals = self.get_bin_centers_as_array(dPhiSig, forFitting=False) # type:ignore
        if self.analysisType in ["central", "semicentral"]: # type:ignore
            # generate the background distribution from the RPFObj 

            dPhiBGRPFs = np.array([self.RPFObjs[i,j].simultaneous_fit(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))]) # type:ignore
            dPhiBGRPFErrs = np.array([self.RPFObjs[i,j].simultaneous_fit_err(x_vals[l],x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))]) # type:ignore
            if k in [0,1,2]:
                dPhiBGRPF = dPhiBGRPFs[:,k] # type:ignore
                dPhiBGRPFErr = dPhiBGRPFErrs[:,k] # type:ignore
        
            if k==3:
                dPhiBGRPF = np.array(np.sum(dPhiBGRPFs, axis=1))/3
                dPhiBGRPFErr = np.array(np.sqrt(np.sum(dPhiBGRPFErrs**2, axis=1)))
            if k not in [0,1,2,3]:
                raise Exception(f"k must be 0, 1, 2, or 3, got {k=}")
            # subtract the background from the signal 
            BGSubtracteddPhiSig = dPhiSigBinContents- dPhiBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D("BGSubtracteddPhiSigHist", "BGSubtracteddPhiSigHist", len(x_vals), x_vals[0], x_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l+1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l+1, BGSubtracteddPhiSigErr[l])

            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers 
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution 
            BGSubtracteddPhiSigHist.Scale(1/N_trig) # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone()
        
        elif self.analysisType == "pp": # type:ignore
            # get the minimum value of the signal
            minVal, minValErr = self.get_minVal_and_systematic_error(dPhiSigBinContents)
            # subtract the minimum value from the signal
            BGSubtracteddPhiSig = dPhiSigBinContents - minVal
            # get the error on the background subtracted signal
            BGSubtracteddPhiSigErr = np.sqrt(dPhiSigBinErrors**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhiSigHist = TH1D("BGSubtracteddPhiSigHist", "BGSubtracteddPhiSigHist", len(x_vals), x_vals[0], x_vals[-1])
            # fill the histogram with the background subtracted signal 
            for l in range(len(x_vals)):
                BGSubtracteddPhiSigHist.SetBinContent(l+1, BGSubtracteddPhiSig[l])
                BGSubtracteddPhiSigHist.SetBinError(l+1, BGSubtracteddPhiSigErr[l])
            # return the background subtracted signal
            BGSubtracteddPhiSigHist.Sumw2()
            # get the number of triggers
            N_trig = self.get_N_trig()
            # normalize the dPhi distribution
            BGSubtracteddPhiSigHist.Scale(1/N_trig) # type:ignore
            # return the background subtracted signal
            return BGSubtracteddPhiSigHist.Clone(), minValErr
        
    def get_minVal_and_systematic_error(self, dPhiBinContents):
        '''
        Returns the minimum value of the dPhi distribution and the systematic error
        '''
        # Get the minimum value in three different ways: standard, sliding window average with a window of 3 points, and a sliding window average with a window of 5 points
        minVal1 = np.min(dPhiBinContents)
        minVal2 = np.min([np.mean(dPhiBinContents[i:i+3]) for i in range(len(dPhiBinContents)-3)])
        minVal3 = np.min([np.mean(dPhiBinContents[i:i+5]) for i in range(len(dPhiBinContents)-5)])
        # get the average of the three minimum values
        minVal = np.mean([minVal1, minVal2, minVal3])
        # get the standard deviation of the three minimum values
        minValErr = np.std([minVal1, minVal2, minVal3])
        # return the minimum value and the systematic error
        return minVal, minValErr
        

    def get_BG_subtracted_AccCorrectedSE_dPhi_dEta_dPion_NS(self, i,j):
        '''
        Returns the background subtracted dPion distribution for the NS signal region
        '''
        # get the dPhi, dEta, dPion distribution for the signal region
        dPhidPionSig = self.get_dPhi_dPion_projection_in_dEta_range(self.dEtaSig) # type:ignore
        dPhidPionSig.Scale(1/dPhidPionSig.GetNbinsY())
        # get the bin contents 
        dPhiSigBinContents = self.get_2D_bin_contents_as_array(dPhidPionSig, forFitting=True) # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_2D_bin_errors_as_array(dPhidPionSig, forFitting=True) # type:ignore
        # get x range 
        x_vals, y_vals = self.get_2D_bin_centers_as_array(dPhidPionSig, forFitting=True) # type:ignore
        if self.analysisType in ["central", "semicentral"]: # type:ignore
            # generate the background distribution from the RPFObj 
            
            
            dPhiIn = [self.RPFObjs[i,j].in_plane_func(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiInErr = [self.RPFObjs[i,j].in_plane_err(x_vals[l],x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiMid = [self.RPFObjs[i,j].mid_plane_func(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiMidErr = [self.RPFObjs[i,j].mid_plane_err(x_vals[l],x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiOut = [self.RPFObjs[i,j].out_of_plane_func(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiOutErr = [self.RPFObjs[i,j].out_of_plane_err(x_vals[l], x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiBGRPF = np.array([dPhiIn[l] + dPhiMid[l] + dPhiOut[l] for l in range(len(x_vals))])/3
            dPhiBGRPFErr = [np.sqrt(dPhiInErr[l]**2 + dPhiMidErr[l]**2 + dPhiOutErr[l]**2) for l in range(len(x_vals))]

            dPhidPionBGRPF = []
            dPhidPionBGRPFErr =[]
            # expand dPhiBGRPF into 2d 
            for i in range(len(dPhiBGRPF)):
                dPhidPionBGRPF.append([dPhiBGRPF[i] for j in range(dPhidPionSig.GetNbinsY())])
                dPhidPionBGRPFErr.append([dPhiBGRPFErr[i] for j in range(dPhidPionSig.GetNbinsY())])
            dPhidPionBGRPF = np.array(dPhidPionBGRPF)
            dPhidPionBGRPFErr = np.array(dPhidPionBGRPFErr)
            # subtract the background from the signal 
            BGSubtracteddPhidPionSig = dPhiSigBinContents - dPhidPionBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhidPionSigErr = np.sqrt(dPhiSigBinErrors**2 + dPhidPionBGRPFErr**2)

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhidPionSigHist = TH2D("BGSubtracteddPhidPionSigNSHist", "BGSubtracteddPhidPionSigNSHist", len(x_vals), x_vals[0], x_vals[-1], len(y_vals), y_vals[0], y_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                for m in range(len(y_vals)):
                    BGSubtracteddPhidPionSigHist.SetBinContent(l+1, m+1, BGSubtracteddPhidPionSig[l, m])
                    BGSubtracteddPhidPionSigHist.SetBinError(l+1, m+1, BGSubtracteddPhidPionSigErr[l, m])
            # return the background subtracted signal
            BGSubtracteddPhidPionSigHist.Sumw2()
            return BGSubtracteddPhidPionSigHist
        elif self.analysisType == "pp": # type:ignore
            # get the minimum value of the signal region
            minVal = np.min(dPhiSigBinContents)
            # subtract the minimum value from the signal region
            BGSubtracteddPhidPionSig = dPhiSigBinContents - minVal
            # get the error on the background subtracted signal
            BGSubtracteddPhidPionSigErr = np.sqrt(dPhiSigBinErrors**2 + minVal**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhidPionSigHist = TH2D("BGSubtracteddPhidPionSigNSHist", "BGSubtracteddPhidPionSigNSHist", len(x_vals), x_vals[0], x_vals[-1], len(y_vals), y_vals[0], y_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                for m in range(len(y_vals)):
                    BGSubtracteddPhidPionSigHist.SetBinContent(l+1, m+1, BGSubtracteddPhidPionSig[l, m])
                    BGSubtracteddPhidPionSigHist.SetBinError(l+1, m+1, BGSubtracteddPhidPionSigErr[l, m])
            # return the background subtracted signal
            BGSubtracteddPhidPionSigHist.Sumw2()
            return BGSubtracteddPhidPionSigHist


    def get_BG_subtracted_AccCorrectedSE_dPhi_dEta_dPion_AS(self, i,j):
        '''
        Returns the background subtracted dPion distribution for the AS signal region
        '''
        # get the dPhi, dEta, dPion distribution for the signal region
        dPhidPionSig = self.get_dPhi_dPion_projection_in_dEta_range(self.dEtaSigAS) # type:ignore
        dPhidPionSig.Scale(1/dPhidPionSig.GetNbinsY())
        # get the bin contents 
        dPhiSigBinContents = self.get_2D_bin_contents_as_array(dPhidPionSig, forFitting=False, forAS=True) # type:ignore
        # get the bin errors
        dPhiSigBinErrors = self.get_2D_bin_errors_as_array(dPhidPionSig, forFitting=False, forAS=True) # type:ignore
        # get x range 
        x_vals, y_vals = self.get_2D_bin_centers_as_array(dPhidPionSig, forFitting=False, forAS=True) # type:ignore
        if self.analysisType in ["central", "semicentral"]:
            # generate the background distribution from the RPFObj 
            
            
            dPhiIn = [self.RPFObjs[i,j].in_plane_func(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiInErr = [self.RPFObjs[i,j].in_plane_err(x_vals[l],x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiMid = [self.RPFObjs[i,j].mid_plane_func(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiMidErr = [self.RPFObjs[i,j].mid_plane_err(x_vals[l],x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiOut = [self.RPFObjs[i,j].out_of_plane_func(x_vals[l], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiOutErr = [self.RPFObjs[i,j].out_of_plane_err(x_vals[l], x_vals[1]-x_vals[0], *self.RPFObjs[i,j].popt) for l in range(len(x_vals))] # type:ignore
            dPhiBGRPF = np.array([dPhiIn[l] + dPhiMid[l] + dPhiOut[l] for l in range(len(x_vals))])/3
            dPhiBGRPFErr = [np.sqrt(dPhiInErr[l]**2 + dPhiMidErr[l]**2 + dPhiOutErr[l]**2) for l in range(len(x_vals))]

            dPhidPionBGRPF = []
            dPhidPionBGRPFErr =[]
            # expand dPhiBGRPF into 2d 
            for i in range(len(dPhiBGRPF)):
                dPhidPionBGRPF.append([dPhiBGRPF[i] for j in range(dPhidPionSig.GetNbinsY())])
                dPhidPionBGRPFErr.append([dPhiBGRPFErr[i] for j in range(dPhidPionSig.GetNbinsY())])
            dPhidPionBGRPF = np.array(dPhidPionBGRPF)
            dPhidPionBGRPFErr = np.array(dPhidPionBGRPFErr)
            # subtract the background from the signal 
            BGSubtracteddPhidPionSig = dPhiSigBinContents - dPhidPionBGRPF
            # get the error on the background subtracted signal
            BGSubtracteddPhidPionSigErr = np.sqrt(dPhiSigBinErrors**2 + dPhidPionBGRPFErr**2)

            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhidPionSigHist = TH2D("BGSubtracteddPhidPionSigASHist", "BGSubtracteddPhidPionSigASHist", len(x_vals), x_vals[0], x_vals[-1], len(y_vals), y_vals[0], y_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                for m in range(len(y_vals)):
                    BGSubtracteddPhidPionSigHist.SetBinContent(l+1, m+1, BGSubtracteddPhidPionSig[l, m])
                    BGSubtracteddPhidPionSigHist.SetBinError(l+1, m+1, BGSubtracteddPhidPionSigErr[l, m])
            # return the background subtracted signal
            BGSubtracteddPhidPionSigHist.Sumw2()
            return BGSubtracteddPhidPionSigHist

        elif self.analysisType == "pp":
            # get the minimum value of the signal distribution
            minSig = np.min(dPhiSigBinContents)
            # subtract the minimum value from the signal distribution
            BGSubtracteddPhidPionSig = dPhiSigBinContents - minSig
            # get the error on the background subtracted signal
            BGSubtracteddPhidPionSigErr = np.sqrt(dPhiSigBinErrors**2 + minSig**2)
            # create a new histogram to hold the background subtracted signal
            BGSubtracteddPhidPionSigHist = TH2D("BGSubtracteddPhidPionSigASHist", "BGSubtracteddPhidPionSigASHist", len(x_vals), x_vals[0], x_vals[-1], len(y_vals), y_vals[0], y_vals[-1])
            # fill the histogram with the background subtracted signal
            for l in range(len(x_vals)):
                for m in range(len(y_vals)):
                    BGSubtracteddPhidPionSigHist.SetBinContent(l+1, m+1, BGSubtracteddPhidPionSig[l, m])
                    BGSubtracteddPhidPionSigHist.SetBinError(l+1, m+1, BGSubtracteddPhidPionSigErr[l, m])
            # return the background subtracted signal
            BGSubtracteddPhidPionSigHist.Sumw2()
            return BGSubtracteddPhidPionSigHist


    def get_pion_TPC_signal_for_BG_subtracted_NS_signal(self, i, j, normalize=False):
        BGSubtracteddPhidPionSigHist = self.get_BG_subtracted_AccCorrectedSE_dPhi_dEta_dPion_NS(i,j)
        dPion = BGSubtracteddPhidPionSigHist.ProjectionY()
        if normalize:
            nTrig = self.get_N_trig()
            dPion.Scale(1/nTrig)
        return dPion

    def get_pion_TPC_signal_for_BG_subtracted_AS_signal(self, i, j, normalize=False):
        BGSubtracteddPhidPionSigHist = self.get_BG_subtracted_AccCorrectedSE_dPhi_dEta_dPion_AS(i,j)
        dPion = BGSubtracteddPhidPionSigHist.ProjectionY()
        if normalize:
            nTrig = self.get_N_trig()
            dPion.Scale(1/nTrig)
        return dPion

    def get_pion_TPC_signal(self):
        pion_TPC_signal = self.JH.Projection(7) # type:ignore
        return pion_TPC_signal