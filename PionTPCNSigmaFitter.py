'''
This is a class that will handle the fitting of the Pion TPC nsigma distributions simultaneously for all species (pi, K, p) and extracting the parameters of the fit.
'''
import numpy as np
from scipy.optimize import curve_fit

class PionTPCNSigmaFitter:
    def __init__(self, p0=None, p0_bounds=None):
        self.p0 = p0
        self.p0_bounds = p0_bounds
        self.popt = None
        self.pcov = None

    def gauss(self, x, mu, sig, a):
        return a*np.exp(-0.5*((x-mu)/sig)**2)

    def three_gaussians(self, x, mup, mupi, muk, sigp, sigpi, sigk, ap, api, ak):
        return api*np.exp(-0.5*((x-mupi)/sigpi)**2)+ap*np.exp(-0.5*((x-mup)/sigp)**2)+ak*np.exp(-0.5*((x-muk)/sigk)**2)
    
    def piKp_enhanced_fit(self, x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk):
        '''
        Returns an hstack of the three gaussians for each enhanced sample: pi, K, and p. 
        x: [array] nsigma values
        mu*: [float] mean of the gaussian
        sig*:  [float] sigma of the gaussian
        a_{@,#}: [float] amplitude of the gaussian for the @ particle in the # enhanced sample
        '''
        return np.hstack([self.three_gaussians(x, mup, mupi, muk, sigp, sigpi, sigk, appi, apipi, akpi), self.three_gaussians(x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp), self.three_gaussians(x, mup, mupi, muk, sigp, sigpi, sigk, apk, apik, akk)])
    
    def piKp_enhanced_error(self, x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, pcov):
        partial_F_app = np.exp(-(x - mup)**2 / (2 * sigp**2))
        partial_F_appi = np.exp(-(x - mup)**2 / (2 * sigp**2))
        partial_F_apk = np.exp(-(x - mup)**2 / (2 * sigp**2))
        partial_F_apip = np.exp(-(x - mupi)**2 / (2 * sigpi**2))
        partial_F_apipi = np.exp(-(x - mupi)**2 / (2 * sigpi**2))
        partial_F_apik = np.exp(-(x - mupi)**2 / (2 * sigpi**2))
        partial_F_akp = np.exp(-(x - muk)**2 / (2 * sigk**2))
        partial_F_akpi = np.exp(-(x - muk)**2 / (2 * sigk**2))
        partial_F_akk = np.exp(-(x - muk)**2 / (2 * sigk**2))

        partial_F_mup = (app+appi+apk) * (x - mup) * np.exp(-(x - mup)**2 / (2 * sigp**2)) / sigp**2
        partial_F_mupi = (apip + apipi + apik) * (x - mupi) * np.exp(-(x - mupi)**2 / (2 * sigpi**2)) / sigpi**2
        partial_F_muk = (akp + akpi + akk) * (x - muk) * np.exp(-(x - muk)**2 / (2 * sigk**2)) / sigk**2

        partial_F_sigp =  (app+appi+apk) * (x - mup)**2 * np.exp(-(x - mup)**2 / (2 * sigp**2)) / sigp**3
        partial_F_sigpi = (apip + apipi + apik) * (x - mupi)**2 * np.exp(-(x - mupi)**2 / (2 * sigpi**2)) / sigpi**3
        partial_F_sigk = (akp + akpi + akk) * (x - muk)**2 * np.exp(-(x - muk)**2 / (2 * sigk**2)) / sigk**3

        cov_mup = pcov[0][0]
        cov_mupi = pcov[1][1]
        cov_muk = pcov[2][2]
        cov_sigp = pcov[3][3]
        cov_sigpi = pcov[4][4]
        cov_sigk = pcov[5][5]
        cov_app = pcov[6][6]
        cov_apip = pcov[7][7]
        cov_akp = pcov[8][8]
        cov_appi = pcov[9][9]
        cov_apipi = pcov[10][10]
        cov_akpi = pcov[11][11]
        cov_apk = pcov[12][12]
        cov_apik = pcov[13][13]
        cov_akk = pcov[14][14]

        cov_mup_mupi = pcov[0][1]
        cov_mup_muk = pcov[0][2]
        cov_mup_sigp = pcov[0][3]
        cov_mup_sigpi = pcov[0][4]
        cov_mup_sigk = pcov[0][5]
        cov_mup_app = pcov[0][6]
        cov_mup_apip = pcov[0][7]
        cov_mup_akp = pcov[0][8]
        cov_mup_appi = pcov[0][9]
        cov_mup_apipi = pcov[0][10]
        cov_mup_akpi = pcov[0][11]
        cov_mup_apk = pcov[0][12]
        cov_mup_apik = pcov[0][13]
        cov_mup_akk = pcov[0][14]

        cov_mupi_muk = pcov[1][2]
        cov_mupi_sigp = pcov[1][3]
        cov_mupi_sigpi = pcov[1][4]
        cov_mupi_sigk = pcov[1][5]
        cov_mupi_app = pcov[1][6]
        cov_mupi_apip = pcov[1][7]
        cov_mupi_akp = pcov[1][8]
        cov_mupi_appi = pcov[1][9]
        cov_mupi_apipi = pcov[1][10]
        cov_mupi_akpi = pcov[1][11]
        cov_mupi_apk = pcov[1][12]
        cov_mupi_apik = pcov[1][13]
        cov_mupi_akk = pcov[1][14]

        cov_muk_sigp = pcov[2][3]
        cov_muk_sigpi = pcov[2][4]
        cov_muk_sigk = pcov[2][5]
        cov_muk_app = pcov[2][6]
        cov_muk_apip = pcov[2][7]
        cov_muk_akp = pcov[2][8]
        cov_muk_appi = pcov[2][9]
        cov_muk_apipi = pcov[2][10]
        cov_muk_akpi = pcov[2][11]
        cov_muk_apk = pcov[2][12]
        cov_muk_apik = pcov[2][13]
        cov_muk_akk = pcov[2][14]

        cov_sigp_sigpi = pcov[3][4]
        cov_sigp_sigk = pcov[3][5]
        cov_sigp_app = pcov[3][6]
        cov_sigp_apip = pcov[3][7]
        cov_sigp_akp = pcov[3][8]
        cov_sigp_appi = pcov[3][9]
        cov_sigp_apipi = pcov[3][10]
        cov_sigp_akpi = pcov[3][11]
        cov_sigp_apk = pcov[3][12]
        cov_sigp_apik = pcov[3][13]
        cov_sigp_akk = pcov[3][14]

        cov_sigpi_sigk = pcov[4][5]
        cov_sigpi_app = pcov[4][6]
        cov_sigpi_apip = pcov[4][7]
        cov_sigpi_akp = pcov[4][8]
        cov_sigpi_appi = pcov[4][9]
        cov_sigpi_apipi = pcov[4][10]
        cov_sigpi_akpi = pcov[4][11]
        cov_sigpi_apk = pcov[4][12]
        cov_sigpi_apik = pcov[4][13]
        cov_sigpi_akk = pcov[4][14]

        cov_sigk_app = pcov[5][6]
        cov_sigk_apip = pcov[5][7]
        cov_sigk_akp = pcov[5][8]
        cov_sigk_appi = pcov[5][9]
        cov_sigk_apipi = pcov[5][10]
        cov_sigk_akpi = pcov[5][11]
        cov_sigk_apk = pcov[5][12]
        cov_sigk_apik = pcov[5][13]
        cov_sigk_akk = pcov[5][14]

        cov_app_apip = pcov[6][7]
        cov_app_akp = pcov[6][8]
        cov_app_appi = pcov[6][9]
        cov_app_apipi = pcov[6][10]
        cov_app_akpi = pcov[6][11]
        cov_app_apk = pcov[6][12]
        cov_app_apik = pcov[6][13]
        cov_app_akk = pcov[6][14]

        cov_apip_akp = pcov[7][8]
        cov_apip_appi = pcov[7][9]
        cov_apip_apipi = pcov[7][10]
        cov_apip_akpi = pcov[7][11]
        cov_apip_apk = pcov[7][12]
        cov_apip_apik = pcov[7][13]
        cov_apip_akk = pcov[7][14]

        cov_akp_appi = pcov[8][9]
        cov_akp_apipi = pcov[8][10]
        cov_akp_akpi = pcov[8][11]
        cov_akp_apk = pcov[8][12]
        cov_akp_apik = pcov[8][13]
        cov_akp_akk = pcov[8][14]

        cov_appi_apipi = pcov[9][10]
        cov_appi_akpi = pcov[9][11]
        cov_appi_apk = pcov[9][12]
        cov_appi_apik = pcov[9][13]
        cov_appi_akk = pcov[9][14]

        cov_apipi_akpi = pcov[10][11]
        cov_apipi_apk = pcov[10][12]
        cov_apipi_apik = pcov[10][13]
        cov_apipi_akk = pcov[10][14]

        cov_akpi_apk = pcov[11][12]
        cov_akpi_apik = pcov[11][13]
        cov_akpi_akk = pcov[11][14]

        cov_apk_apik = pcov[12][13]
        cov_apk_akk = pcov[12][14]

        cov_apik_akk = pcov[13][14]


        uncertainty_F = np.sqrt(
            partial_F_mup**2*cov_mup + 
            partial_F_mupi**2*cov_mupi +
            partial_F_muk**2*cov_muk +
            partial_F_sigp**2*cov_sigp +
            partial_F_sigpi**2*cov_sigpi +
            partial_F_sigk**2*cov_sigk +
            partial_F_app**2*cov_app +
            partial_F_apip**2*cov_apip +
            partial_F_akp**2*cov_akp +
            partial_F_appi**2*cov_appi +
            partial_F_apipi**2*cov_apipi +
            partial_F_akpi**2*cov_akpi +
            partial_F_apk**2*cov_apk +
            partial_F_apik**2*cov_apik +
            partial_F_akk**2*cov_akk +
            2*partial_F_mup*partial_F_mupi*cov_mup_mupi +
            2*partial_F_mup*partial_F_muk*cov_mup_muk +
            2*partial_F_mup*partial_F_sigp*cov_mup_sigp +
            2*partial_F_mup*partial_F_sigpi*cov_mup_sigpi +
            2*partial_F_mup*partial_F_sigk*cov_mup_sigk +
            2*partial_F_mup*partial_F_app*cov_mup_app +
            2*partial_F_mup*partial_F_apip*cov_mup_apip +
            2*partial_F_mup*partial_F_akp*cov_mup_akp +
            2*partial_F_mup*partial_F_appi*cov_mup_appi +
            2*partial_F_mup*partial_F_apipi*cov_mup_apipi +
            2*partial_F_mup*partial_F_akpi*cov_mup_akpi +
            2*partial_F_mup*partial_F_apk*cov_mup_apk +
            2*partial_F_mup*partial_F_apik*cov_mup_apik +
            2*partial_F_mup*partial_F_akk*cov_mup_akk +
            2*partial_F_mupi*partial_F_muk*cov_mupi_muk +
            2*partial_F_mupi*partial_F_sigp*cov_mupi_sigp +
            2*partial_F_mupi*partial_F_sigpi*cov_mupi_sigpi +
            2*partial_F_mupi*partial_F_sigk*cov_mupi_sigk +
            2*partial_F_mupi*partial_F_app*cov_mupi_app +
            2*partial_F_mupi*partial_F_apip*cov_mupi_apip +
            2*partial_F_mupi*partial_F_akp*cov_mupi_akp +
            2*partial_F_mupi*partial_F_appi*cov_mupi_appi +
            2*partial_F_mupi*partial_F_apipi*cov_mupi_apipi +
            2*partial_F_mupi*partial_F_akpi*cov_mupi_akpi +
            2*partial_F_mupi*partial_F_apk*cov_mupi_apk +
            2*partial_F_mupi*partial_F_apik*cov_mupi_apik +
            2*partial_F_mupi*partial_F_akk*cov_mupi_akk +
            2*partial_F_muk*partial_F_sigp*cov_muk_sigp +
            2*partial_F_muk*partial_F_sigpi*cov_muk_sigpi +
            2*partial_F_muk*partial_F_sigk*cov_muk_sigk +
            2*partial_F_muk*partial_F_app*cov_muk_app +
            2*partial_F_muk*partial_F_apip*cov_muk_apip +
            2*partial_F_muk*partial_F_akp*cov_muk_akp +
            2*partial_F_muk*partial_F_appi*cov_muk_appi +
            2*partial_F_muk*partial_F_apipi*cov_muk_apipi +
            2*partial_F_muk*partial_F_akpi*cov_muk_akpi +
            2*partial_F_muk*partial_F_apk*cov_muk_apk +
            2*partial_F_muk*partial_F_apik*cov_muk_apik +
            2*partial_F_muk*partial_F_akk*cov_muk_akk +
            2*partial_F_sigp*partial_F_sigpi*cov_sigp_sigpi +
            2*partial_F_sigp*partial_F_sigk*cov_sigp_sigk +
            2*partial_F_sigp*partial_F_app*cov_sigp_app +
            2*partial_F_sigp*partial_F_apip*cov_sigp_apip +
            2*partial_F_sigp*partial_F_akp*cov_sigp_akp +
            2*partial_F_sigp*partial_F_appi*cov_sigp_appi +
            2*partial_F_sigp*partial_F_apipi*cov_sigp_apipi +
            2*partial_F_sigp*partial_F_akpi*cov_sigp_akpi +
            2*partial_F_sigp*partial_F_apk*cov_sigp_apk +
            2*partial_F_sigp*partial_F_apik*cov_sigp_apik +
            2*partial_F_sigp*partial_F_akk*cov_sigp_akk +
            2*partial_F_sigpi*partial_F_sigk*cov_sigpi_sigk +
            2*partial_F_sigpi*partial_F_app*cov_sigpi_app +
            2*partial_F_sigpi*partial_F_apip*cov_sigpi_apip +
            2*partial_F_sigpi*partial_F_akp*cov_sigpi_akp +
            2*partial_F_sigpi*partial_F_appi*cov_sigpi_appi +
            2*partial_F_sigpi*partial_F_apipi*cov_sigpi_apipi +
            2*partial_F_sigpi*partial_F_akpi*cov_sigpi_akpi +
            2*partial_F_sigpi*partial_F_apk*cov_sigpi_apk +
            2*partial_F_sigpi*partial_F_apik*cov_sigpi_apik +
            2*partial_F_sigpi*partial_F_akk*cov_sigpi_akk +
            2*partial_F_sigk*partial_F_app*cov_sigk_app +
            2*partial_F_sigk*partial_F_apip*cov_sigk_apip +
            2*partial_F_sigk*partial_F_akp*cov_sigk_akp +
            2*partial_F_sigk*partial_F_appi*cov_sigk_appi +
            2*partial_F_sigk*partial_F_apipi*cov_sigk_apipi +
            2*partial_F_sigk*partial_F_akpi*cov_sigk_akpi +
            2*partial_F_sigk*partial_F_apk*cov_sigk_apk +
            2*partial_F_sigk*partial_F_apik*cov_sigk_apik +
            2*partial_F_sigk*partial_F_akk*cov_sigk_akk +
            2*partial_F_app*partial_F_apip*cov_app_apip +
            2*partial_F_app*partial_F_akp*cov_app_akp +
            2*partial_F_app*partial_F_appi*cov_app_appi +
            2*partial_F_app*partial_F_apipi*cov_app_apipi +
            2*partial_F_app*partial_F_akpi*cov_app_akpi +
            2*partial_F_app*partial_F_apk*cov_app_apk +
            2*partial_F_app*partial_F_apik*cov_app_apik +
            2*partial_F_app*partial_F_akk*cov_app_akk +
            2*partial_F_apip*partial_F_akp*cov_apip_akp +
            2*partial_F_apip*partial_F_appi*cov_apip_appi +
            2*partial_F_apip*partial_F_apipi*cov_apip_apipi +
            2*partial_F_apip*partial_F_akpi*cov_apip_akpi +
            2*partial_F_apip*partial_F_apk*cov_apip_apk +
            2*partial_F_apip*partial_F_apik*cov_apip_apik +
            2*partial_F_apip*partial_F_akk*cov_apip_akk +
            2*partial_F_akp*partial_F_appi*cov_akp_appi +
            2*partial_F_akp*partial_F_apipi*cov_akp_apipi +
            2*partial_F_akp*partial_F_akpi*cov_akp_akpi +
            2*partial_F_akp*partial_F_apk*cov_akp_apk +
            2*partial_F_akp*partial_F_apik*cov_akp_apik +
            2*partial_F_akp*partial_F_akk*cov_akp_akk +
            2*partial_F_appi*partial_F_apipi*cov_appi_apipi +
            2*partial_F_appi*partial_F_akpi*cov_appi_akpi +
            2*partial_F_appi*partial_F_apk*cov_appi_apk +
            2*partial_F_appi*partial_F_apik*cov_appi_apik +
            2*partial_F_appi*partial_F_akk*cov_appi_akk +
            2*partial_F_apipi*partial_F_akpi*cov_apipi_akpi +
            2*partial_F_apipi*partial_F_apk*cov_apipi_apk +
            2*partial_F_apipi*partial_F_apik*cov_apipi_apik +
            2*partial_F_apipi*partial_F_akk*cov_apipi_akk +
            2*partial_F_akpi*partial_F_apk*cov_akpi_apk +
            2*partial_F_akpi*partial_F_apik*cov_akpi_apik +
            2*partial_F_akpi*partial_F_akk*cov_akpi_akk +
            2*partial_F_apk*partial_F_apik*cov_apk_apik +
            2*partial_F_apk*partial_F_akk*cov_apk_akk +
            2*partial_F_apik*partial_F_akk*cov_apik_akk
        )

        return uncertainty_F

    def fit(self, x, y, yerr):
        popt_all, pcov_all = curve_fit(self.piKp_enhanced_fit, x, np.hstack(y), p0=self.p0, bounds=self.p0_bounds, maxfev=1000000)
        self.popt = popt_all
        self.pcov = pcov_all

        NDF = len(np.hstack(y)) - len(self.popt) - 1
        chi2=np.sum((np.hstack(y) - self.piKp_enhanced_fit(x, *self.popt))**2)/(np.hstack(yerr))**2
        
        self.chi2OverNDF = chi2/NDF

        return self.popt, self.pcov, self.chi2OverNDF