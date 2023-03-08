import unittest
import numpy as np
from JetHadron import JetHadron
from ROOT import TFile, THnSparseF 
from array import array

def make_mock_JH_ThNsparse():
    nbins = array("i",[2, 24, 100, 28, 72, 2, 3, 100])
    dim_range_low = array("d",[0, 0, 0, -1.4, -np.pi/2, -0.5, 0, -20])
    dim_range_high = array("d",[20, 200, 10, 1.4, 3*np.pi/2, 1.5, np.pi/2, 20])
    JH = THnSparseF("JH", "JH", 8, nbins, dim_range_low, dim_range_high)
    # now fill in 1000000 random entries 
    for i in range(1000000):
        arr = np.random.rand(8)
        JH.Fill(array("d",arr))
    return JH

class JetHadron_test(unittest.TestCase):
    rootFile = TFile('/home/steffanic/Projects/Thesis/python_backend/AnalysisResults_qr_merged.root')

    def test_mock_SE_correlation_function_has_same_number_of_entries_as_sparse(self):
        mockJH = make_mock_JH_ThNsparse()
        
        jhAna = JetHadron(self.rootFile, "central", fill_on_init=False)
        # assign mock JH to jhAna
        jhAna.JH = mockJH
        # test the correlation function
        corr = jhAna.get_SE_correlation_function()
        # check that the correlation function normalizes to JH.GetEntries()
        self.assertEqual(corr.GetEntries(), mockJH.GetEntries(), "Correlation function does not have the same number of entries as mockJH.GetEntries()")
        del mockJH, jhAna, corr

    def test_actual_SE_correlation_function__has_same_number_of_entries_as_sparse(self):
        jhAna = JetHadron(self.rootFile, "central", fill_on_init=False)
        # get number of entries from the actual JH
        JHEntries = jhAna.JH.GetEntries()
        # test the correlation function
        corr = jhAna.get_SE_correlation_function()
        # check that the correlation function normalizes to JH.GetEntries()
        self.assertEqual(corr.GetEntries(), JHEntries, "Correlation function does not have the same number of entries as JH.GetEntries()")
        del jhAna, corr

    def test_mock_SE_correlation_function_integrates_to_Nentries(self):
        mockJH = make_mock_JH_ThNsparse()
        
        jhAna = JetHadron(self.rootFile, "central", fill_on_init=False)
        # assign mock JH to jhAna
        jhAna.JH = mockJH
        # test the correlation function
        corr = jhAna.get_SE_correlation_function()
        # check that the correlation function normalizes to JH.GetEntries()
        self.assertEqual(corr.Integral(), mockJH.GetEntries(), "Correlation function does not integrate to mockJH.GetEntries()")
        del mockJH, jhAna, corr

    def test_actual_SE_correlation_function_integrates_to_Nentries(self):
        jhAna = JetHadron(self.rootFile, "central", fill_on_init=False)
        # get number of entries from the actual JH
        JHEntries = jhAna.JH.GetEntries()
        
        # test the correlation function
        corr = jhAna.get_SE_correlation_function()
        # check that the correlation function normalizes to JH.GetEntries()
        self.assertEqual(corr.Integral(),JHEntries , "Correlation function does not integrate to JH.GetEntries()")
        del jhAna, corr


if __name__ == '__main__':
    unittest.main()