import numpy as np
from pyunfold import iterative_unfold
from typing import Optional

class ResponseMatrix:
    def __init__(self):
        self.matrix = np.zeros((4,3)) # pi, p, k, trash
        self.error = np.zeros((4,3)) # pi, p, k, trash
        # self.root_response = None#RooUnfoldResponse(4, 0.0, 3.0, 3, 0.0, 2.0) # Int_t 	nm, Double_t 	mlo, Double_t 	mhi, Int_t 	nt, Double_t 	tlo, Double_t 	thi, const char * 	name = 0, const char * 	title = 0 

    def buildResponseMatrix(self, pionEffect:dict, protonEffect:dict, kaonEffect:dict, trashEffect:dict, normalize:bool=False, use_ROOT:bool=False):
        '''
        The steps to generating the response matrix are as follows:
        1. Generate the Pion-, Proton-, Kaon-, and Trash- TPC Pion NSigma histograms
        2. Fit them with the appropriate functions\
        3. Integrate the functions from -inf to inf to get the total number of particles in each sample of each type (pi, p, k, trash)
        4. Divide the number of particles in each sample by the total number of particles in each sample to get the probability of each particle being a pion, proton, kaon, or trash
        5. Fill the response matrix with the probabilities
        6. Normalize the response matrix
        7. Profit (hopefully)

        This function should only be responsible for taking the probabilities and filling the response matrix
        It should take the effect probabilities as an argument, e.g. pionEffect, which is a dictionary of the form:
        {
            'pionCause': probability of a pion being reconstructed as a pion,
            'protonCause': probability of a proton being reconstructed as a pion,
            'kaonCause': probability of a kaon being reconstructed as a pion,
            'trashCause': probability of a trash particle being reconstructed as a pion
        }
        etc. for the other effect probabilities
        '''
        # Fill the response matrix
        self.matrix[0][0] = pionEffect['pionCause']
        self.matrix[0][1] = pionEffect['protonCause']
        self.matrix[0][2] = pionEffect['kaonCause']


        self.matrix[1][0] = protonEffect['pionCause']
        self.matrix[1][1] = protonEffect['protonCause']
        self.matrix[1][2] = protonEffect['kaonCause']


        self.matrix[2][0] = kaonEffect['pionCause']
        self.matrix[2][1] = kaonEffect['protonCause']
        self.matrix[2][2] = kaonEffect['kaonCause']


        self.matrix[3][0] = trashEffect['pionCause']
        self.matrix[3][1] = trashEffect['protonCause']
        self.matrix[3][2] = trashEffect['kaonCause']


        if normalize:    
            # Normalize the response matrix
            self.matrix = self.matrix / np.sum(self.matrix, axis=0)

#         if use_ROOT:
#             resp_TH2 = TH2D("resp_TH2", "resp_TH2", 4, 0.0, 3.0, 3, 0.0, 2.0)
#             resp_TH2.SetBinContent(1, 1, self.matrix[0][0])
#             resp_TH2.SetBinContent(1, 2, self.matrix[0][1])
#             resp_TH2.SetBinContent(1, 3, self.matrix[0][2])
# 
#             resp_TH2.SetBinContent(2, 1, self.matrix[1][0])
#             resp_TH2.SetBinContent(2, 2, self.matrix[1][1])
#             resp_TH2.SetBinContent(2, 3, self.matrix[1][2])
# 
#             resp_TH2.SetBinContent(3, 1, self.matrix[2][0])
#             resp_TH2.SetBinContent(3, 2, self.matrix[2][1])
#             resp_TH2.SetBinContent(3, 3, self.matrix[2][2])
# 
#             resp_TH2.SetBinContent(4, 1, self.matrix[3][0])
#             resp_TH2.SetBinContent(4, 2, self.matrix[3][1])
#             resp_TH2.SetBinContent(4, 3, self.matrix[3][2])
# 
#             self.root_response = RooUnfoldResponse(0,0,resp_TH2)
#             breakpoint()
            
            

    def buildResponseMatrixError(self, pionEffectError:dict, protonEffectError:dict, kaonEffectError:dict, trashEffectError:dict):
        '''
        This function should take the effect errors as an argument, e.g. pionEffectError, which is a dictionary of the form:
        {
            'pionCauseError': uncertainty of a pion being reconstructed as a pion,
            'protonCauseError': uncertainty of a proton being reconstructed as a pion,
            'kaonCauseError': uncertainty of a kaon being reconstructed as a pion,
            'trashCauseError': uncertainty of a trash particle being reconstructed as a pion
        }
        etc. for the other effect errors
        '''
        # Fill the response matrix error
        self.error[0][0] = pionEffectError['pionCauseError']
        self.error[0][1] = pionEffectError['protonCauseError']
        self.error[0][2] = pionEffectError['kaonCauseError']

        self.error[1][0] = protonEffectError['pionCauseError']
        self.error[1][1] = protonEffectError['protonCauseError']
        self.error[1][2] = protonEffectError['kaonCauseError']

        self.error[2][0] = kaonEffectError['pionCauseError']
        self.error[2][1] = kaonEffectError['protonCauseError']
        self.error[2][2] = kaonEffectError['kaonCauseError']

        self.error[3][0] = trashEffectError['pionCauseError']
        self.error[3][1] = trashEffectError['protonCauseError']
        self.error[3][2] = trashEffectError['kaonCauseError']

    def getResponseMatrix(self):
        return self.matrix
    
    def getResponseMatrixError(self):
        return self.error

class Unfolder:
    def __init__(self, response_matrix: ResponseMatrix):
        self.response_matrix = response_matrix
        self.unfolding_matrix = None # to be filled by the unfolding algorithm

    def unfold(self, data: np.ndarray, data_errors: np.ndarray, efficiency: Optional[np.ndarray]=None, efficiency_errors: Optional[np.ndarray]=None, test_statistic: str='ks', test_statistic_stopping: float=0.01):
        '''
        This function should take the data and return the unfolded data
        '''
        if efficiency is None and efficiency_errors is None:
            efficiency = np.ones(data.shape[0]-1)
            efficiency_errors = np.zeros(data.shape[0]-1)
        unfolded_data = iterative_unfold(
            data=data, data_err=data_errors, response=self.response_matrix.getResponseMatrix(), response_err=self.response_matrix.getResponseMatrixError(), efficiencies=efficiency, efficiencies_err=efficiency_errors, ts=test_statistic, ts_stopping=test_statistic_stopping, cov_type='poisson', max_iter=100)
        self.unfolding_matrix = unfolded_data['unfolding_matrix']
        return unfolded_data

    def refold(self, unfolded_data: np.ndarray):
        '''
        This function should take the unfolded data and return the refolded data
        '''
        if self.unfolding_matrix is None:
            raise ValueError("Unfolding matrix is not set. Please run unfold() first.")
        refolded_data = np.matmul(self.unfolding_matrix, unfolded_data)
        return refolded_data