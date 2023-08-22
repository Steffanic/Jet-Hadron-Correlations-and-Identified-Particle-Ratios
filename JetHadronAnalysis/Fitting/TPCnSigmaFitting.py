import warnings
import numpy as np
import uncertainties.unumpy as unp
from scipy.special import erf

def gauss(x, mu, sig, a):
    return a*np.exp(-0.5*((x-mu)/sig)**2)

def generalized_gauss(x, mu, sig, a, alpha):
    return a*np.exp(-0.5*((x-mu)/sig)**2)*(1 + erf(alpha*(x-mu)/(sig*2**0.5)))

def ugauss(x, mu, sig, a):
    return a*unp.exp(-0.5*((x-mu)/sig)**2)

def ugeneralized_gauss(x, mu, sig, a, alpha):
    return a*unp.exp(-0.5*((x-mu)/sig)**2)*(1 + unp.erf(alpha*(x-mu)/(sig*2**0.5)))

def two_generalized_gaussians_and_one_gaussian(x, mup, mupi, muk, sigp, sigpi, sigk, ap, api, ak, alphap, alphak):
    return api*np.exp(-0.5*((x-mupi)/sigpi)**2) + ap*np.exp(-0.5*((x-mup)/sigp)**2)*(1 + erf(alphap*(x-mup)/(sigp*2**0.5))) + ak*np.exp(-0.5*((x-muk)/sigk)**2)*(1 + erf(alphak*(x-muk)/(sigk*2**0.5)))

def utwo_generalized_gaussians_and_one_gaussian(x, mup, mupi, muk, sigp, sigpi, sigk, ap, api, ak, alphap, alphak):
    return api*unp.exp(-0.5*((x-mupi)/sigpi)**2) + ap*unp.exp(-0.5*((x-mup)/sigp)**2)*(1 + unp.erf(alphap*(x-mup)/(sigp*2**0.5))) + ak*unp.exp(-0.5*((x-muk)/sigk)**2)*(1 + unp.erf(alphak*(x-muk)/(sigk*2**0.5)))

def piKpInc_generalized_fit(non_zero_masks, x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak):
    """
    Generalized fit function for piKpInc
    """
    non_zero_masks = non_zero_masks or None
    if non_zero_masks is None:
        warnings.warn("No non_zero_masks passed to generalized_fit. Setting to np.ones_like(x)")
        non_zero_masks = [np.ones_like(x, dtype=bool)]*4

    return np.hstack([two_generalized_gaussians_and_one_gaussian(x[non_zero_masks[0]], mup, mupi, muk, sigp, sigpi, sigk, appi, apipi, akpi, alphap, alphak), two_generalized_gaussians_and_one_gaussian(x[non_zero_masks[1]], mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, alphap, alphak), two_generalized_gaussians_and_one_gaussian(x[non_zero_masks[2]], mup, mupi, muk, sigp, sigpi, sigk, apk, apik, akk, alphap, alphak), two_generalized_gaussians_and_one_gaussian(x[non_zero_masks[3]], mup, mupi, muk, sigp, sigpi, sigk, apinc, apiinc, akinc, alphap, alphak)])

def upiKpInc_generalized_fit(non_zero_masks, x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak):
    """
    Generalized fit function for piKpInc
    """
    non_zero_masks = non_zero_masks or None
    if non_zero_masks is None:
        warnings.warn("No non_zero_masks passed to ugeneralized_fit. Setting to np.ones_like(x)")
        non_zero_masks = [np.ones_like(x, dtype=bool)]*4

    return np.hstack([utwo_generalized_gaussians_and_one_gaussian(x[non_zero_masks[0]], mup, mupi, muk, sigp, sigpi, sigk, appi, apipi, akpi, alphap, alphak), utwo_generalized_gaussians_and_one_gaussian(x[non_zero_masks[1]], mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, alphap, alphak), utwo_generalized_gaussians_and_one_gaussian(x[non_zero_masks[2]], mup, mupi, muk, sigp, sigpi, sigk, apk, apik, akk, alphap, alphak), utwo_generalized_gaussians_and_one_gaussian(x[non_zero_masks[3]], mup, mupi, muk, sigp, sigpi, sigk, apinc, apiinc, akinc, alphap, alphak)])
        
    
def piKpInc_generalized_jac(non_zero_masks, x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak):
    non_zero_masks = non_zero_masks or None
    if non_zero_masks is None:
        warnings.warn("No non_zero_masks passed to generalized_jac. Setting to np.ones_like(x)")
        non_zero_masks = [np.ones_like(x,dtype=bool)]*4
    partial_F_appi = np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[0]]-mup)/(sigp*2**0.5)))
    partial_F_apipi = np.exp(-(x[non_zero_masks[0]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akpi = np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[0]]-muk)/(sigk*2**0.5)))

    # now let's hstack with the other masks
    partial_F_appi = np.hstack([partial_F_appi, np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_apipi = np.hstack([partial_F_apipi, np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_akpi = np.hstack([partial_F_akpi, np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])

    partial_F_app = np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[1]]-mup)/(sigp*2**0.5)))
    partial_F_apip = np.exp(-(x[non_zero_masks[1]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akp = np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[1]]-muk)/(sigk*2**0.5)))

    partial_F_app = np.hstack([np.zeros(len(x[non_zero_masks[0]])), partial_F_app, np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_apip = np.hstack([np.zeros(len(x[non_zero_masks[0]])), partial_F_apip, np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_akp = np.hstack([np.zeros(len(x[non_zero_masks[0]])), partial_F_akp, np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    
    partial_F_apk = np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[2]]-mup)/(sigp*2**0.5)))
    partial_F_apik = np.exp(-(x[non_zero_masks[2]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akk = np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[2]]-muk)/(sigk*2**0.5)))

    partial_F_apk = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), partial_F_apk, np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_apik = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), partial_F_apik, np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_akk = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), partial_F_akk, np.zeros(len(x[non_zero_masks[3]]))])

    partial_F_apinc = np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[3]]-mup)/(sigp*2**0.5)))
    partial_F_apiinc = np.exp(-(x[non_zero_masks[3]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akinc = np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[3]]-muk)/(sigk*2**0.5)))

    partial_F_apinc = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), partial_F_apinc])
    partial_F_apiinc = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), partial_F_apiinc])
    partial_F_akinc = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), partial_F_akinc])

    partial_F_mup = np.hstack([
                                (appi) * ((2*(x[non_zero_masks[0]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[0]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[0]]-mup)**2 / (2 * sigp**2))),
                                (app) * ((2*(x[non_zero_masks[1]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[1]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[1]]-mup)**2 / (2 * sigp**2))), 
                                (apk) * ((2*(x[non_zero_masks[2]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[2]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[2]]-mup)**2 / (2 * sigp**2))), 
                                (apinc) * ((2*(x[non_zero_masks[3]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[3]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[3]]-mup)**2 / (2 * sigp**2)))])
    partial_F_mupi = np.hstack([
                                    (apipi) * (x[non_zero_masks[0]] - mupi) * np.exp(-(x[non_zero_masks[0]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2,
                                    (apip) * (x[non_zero_masks[1]] - mupi) * np.exp(-(x[non_zero_masks[1]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2,
                                    (apik) * (x[non_zero_masks[2]] - mupi) * np.exp(-(x[non_zero_masks[2]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2,
                                    (apiinc) * (x[non_zero_masks[3]] - mupi) * np.exp(-(x[non_zero_masks[3]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2])
    partial_F_muk = np.hstack([
        (akpi) * (2*(x[non_zero_masks[0]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[0]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[0]]-muk)**2 / (2 * sigk**2)),
        (akp) * (2*(x[non_zero_masks[1]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[1]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[1]]-muk)**2 / (2 * sigk**2)),
        (akk) * (2*(x[non_zero_masks[2]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[2]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[2]]-muk)**2 / (2 * sigk**2)),
        (akinc) * (2*(x[non_zero_masks[3]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[3]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[3]]-muk)**2 / (2 * sigk**2))])

    partial_F_sigp =  np.hstack([(appi) * ((2*(x[non_zero_masks[0]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[0]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[0]]-mup) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[0]]-mup)**2 / (2 * sigp**2))),
                                    (app) * ((2*(x[non_zero_masks[1]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[1]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[1]]-mup) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[1]]-mup)**2 / (2 * sigp**2))),
                                    (apip) * ((2*(x[non_zero_masks[2]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[2]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[2]]-mup) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[2]]-mup)**2 / (2 * sigp**2))),
                                    (apinc) * ((2*(x[non_zero_masks[3]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[3]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[3]]-mup) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[3]]-mup)**2 / (2 * sigp**2)))
    ])
    partial_F_sigpi = np.hstack([(apipi) * (x[non_zero_masks[0]] - mupi)**2 * np.exp(-(x[non_zero_masks[0]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3,
                                    (apip) * (x[non_zero_masks[1]] - mupi)**2 * np.exp(-(x[non_zero_masks[1]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3,
                                    (apik)* (x[non_zero_masks[2]] - mupi)**2 * np.exp(-(x[non_zero_masks[2]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3,
                                    (apiinc) * (x[non_zero_masks[3]] - mupi)**2 * np.exp(-(x[non_zero_masks[3]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3
                                    ])
    partial_F_sigk = np.hstack([( akpi) * ((2*(x[non_zero_masks[0]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[0]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[0]]-muk) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[0]]-muk)**2 / (2 * sigk**2))),
                                (akp) * ((2*(x[non_zero_masks[1]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[1]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[1]]-muk) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[1]]-muk)**2 / (2 * sigk**2))),
                                (akk) * ((2*(x[non_zero_masks[2]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[2]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[2]]-muk) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[2]]-muk)**2 / (2 * sigk**2))),
                                    (akinc) * ((2*(x[non_zero_masks[3]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[3]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[3]]-muk) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[3]]-muk)**2 / (2 * sigk**2)))
                                ])

    partial_F_alphap = np.hstack([(appi) * (2/np.pi)**.5 * (x[non_zero_masks[0]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[0]]-mup)**2 / (2 * sigp**2)),
                                    (app) * (2/np.pi)**.5 * (x[non_zero_masks[1]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[1]]-mup)**2 / (2 * sigp**2)),
                                    (apk) * (2/np.pi)**.5 * (x[non_zero_masks[2]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[2]]-mup)**2 / (2 * sigp**2)),
                                    (apinc) * (2/np.pi)**.5 * (x[non_zero_masks[3]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[3]]-mup)**2 / (2 * sigp**2))
                                    ])
    partial_F_alphak = np.hstack([( akpi ) * (2/np.pi)**.5 * (x[non_zero_masks[0]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[0]]-muk)**2 / (2 * sigk**2)),
                                    (akp) * (2/np.pi)**.5 * (x[non_zero_masks[1]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[1]]-muk)**2 / (2 * sigk**2)),
                                    (akk) * (2/np.pi)**.5 * (x[non_zero_masks[2]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[2]]-muk)**2 / (2 * sigk**2)),
                                    (akinc) * (2/np.pi)**.5 * (x[non_zero_masks[3]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[3]]-muk)**2 / (2 * sigk**2))
                                    ])

    


    jacobian = np.column_stack((partial_F_mup, partial_F_mupi, partial_F_muk, partial_F_sigp, partial_F_sigpi, partial_F_sigk, partial_F_app, partial_F_apip, partial_F_akp, partial_F_appi, partial_F_apipi, partial_F_akpi, partial_F_apk, partial_F_apik, partial_F_akk, partial_F_apinc, partial_F_apiinc, partial_F_akinc, partial_F_alphap, partial_F_alphak))
    return jacobian

def piKpInc_generalized_error(non_zero_masks, x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak, pcov):
    non_zero_masks = non_zero_masks or None
    if non_zero_masks is None:
        warnings.warn("No non_zero_masks passed to ugeneralized_fit. Setting to np.ones_like(x)")
        non_zero_masks = [np.ones_like(x, dtype=bool)]*4

    partial_F_appi = np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[0]]-mup)/(sigp*2**0.5)))
    partial_F_apipi = np.exp(-(x[non_zero_masks[0]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akpi = np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[0]]-muk)/(sigk*2**0.5)))

    # now let's hstack with the other masks
    partial_F_appi = np.hstack([partial_F_appi, np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_apipi = np.hstack([partial_F_apipi, np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_akpi = np.hstack([partial_F_akpi, np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])

    partial_F_app = np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[1]]-mup)/(sigp*2**0.5)))
    partial_F_apip = np.exp(-(x[non_zero_masks[1]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akp = np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[1]]-muk)/(sigk*2**0.5)))

    partial_F_app = np.hstack([np.zeros(len(x[non_zero_masks[0]])), partial_F_app, np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_apip = np.hstack([np.zeros(len(x[non_zero_masks[0]])), partial_F_apip, np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_akp = np.hstack([np.zeros(len(x[non_zero_masks[0]])), partial_F_akp, np.zeros(len(x[non_zero_masks[2]])), np.zeros(len(x[non_zero_masks[3]]))])
    
    partial_F_apk = np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[2]]-mup)/(sigp*2**0.5)))
    partial_F_apik = np.exp(-(x[non_zero_masks[2]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akk = np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[2]]-muk)/(sigk*2**0.5)))

    partial_F_apk = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), partial_F_apk, np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_apik = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), partial_F_apik, np.zeros(len(x[non_zero_masks[3]]))])
    partial_F_akk = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), partial_F_akk, np.zeros(len(x[non_zero_masks[3]]))])

    partial_F_apinc = np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[3]]-mup)/(sigp*2**0.5)))
    partial_F_apiinc = np.exp(-(x[non_zero_masks[3]] - mupi)**2 / (2 * sigpi**2))
    partial_F_akinc = np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[3]]-muk)/(sigk*2**0.5)))

    partial_F_apinc = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), partial_F_apinc])
    partial_F_apiinc = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), partial_F_apiinc])
    partial_F_akinc = np.hstack([np.zeros(len(x[non_zero_masks[0]])), np.zeros(len(x[non_zero_masks[1]])), np.zeros(len(x[non_zero_masks[2]])), partial_F_akinc])

    partial_F_mup = np.hstack([
                                (appi) * ((2*(x[non_zero_masks[0]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[0]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[0]]-mup)**2 / (2 * sigp**2))),
                                (app) * ((2*(x[non_zero_masks[1]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[1]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[1]]-mup)**2 / (2 * sigp**2))), 
                                (apk) * ((2*(x[non_zero_masks[2]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[2]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[2]]-mup)**2 / (2 * sigp**2))), 
                                (apinc) * ((2*(x[non_zero_masks[3]]-mup))/(2*sigp**2) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[3]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[3]]-mup)**2 / (2 * sigp**2)))])
    partial_F_mupi = np.hstack([
                                    (apipi) * (x[non_zero_masks[0]] - mupi) * np.exp(-(x[non_zero_masks[0]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2,
                                    (apip) * (x[non_zero_masks[1]] - mupi) * np.exp(-(x[non_zero_masks[1]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2,
                                    (apik) * (x[non_zero_masks[2]] - mupi) * np.exp(-(x[non_zero_masks[2]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2,
                                    (apiinc) * (x[non_zero_masks[3]] - mupi) * np.exp(-(x[non_zero_masks[3]] - mupi)**2 / (2 * sigpi**2)) / sigpi**2])
    partial_F_muk = np.hstack([
        (akpi) * (2*(x[non_zero_masks[0]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[0]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[0]]-muk)**2 / (2 * sigk**2)),
        (akp) * (2*(x[non_zero_masks[1]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[1]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[1]]-muk)**2 / (2 * sigk**2)),
        (akk) * (2*(x[non_zero_masks[2]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[2]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[2]]-muk)**2 / (2 * sigk**2)),
        (akinc) * (2*(x[non_zero_masks[3]]-muk))/(2*sigk**2) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[3]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[3]]-muk)**2 / (2 * sigk**2))])

    partial_F_sigp =  np.hstack([(appi) * ((2*(x[non_zero_masks[0]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[0]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[0]]-mup) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[0]]-mup)**2 / (2 * sigp**2))),
                                (app) * ((2*(x[non_zero_masks[1]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[1]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[1]]-mup) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[1]]-mup)**2 / (2 * sigp**2))),
                                    (apip) * ((2*(x[non_zero_masks[2]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[2]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[2]]-mup) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[2]]-mup)**2 / (2 * sigp**2))),
                                    (apinc) * ((2*(x[non_zero_masks[3]]-mup)**2)/(2*sigp) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2))*(1 + erf(alphap*(x[non_zero_masks[3]]-mup)/(sigp*2**0.5))) - (2/np.pi)**.5 * alphap/sigp**2 * (x[non_zero_masks[3]]-mup) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[3]]-mup)**2 / (2 * sigp**2)))
    ])
    partial_F_sigpi = np.hstack([(apipi) * (x[non_zero_masks[0]] - mupi)**2 * np.exp(-(x[non_zero_masks[0]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3,
                                (apip) * (x[non_zero_masks[1]] - mupi)**2 * np.exp(-(x[non_zero_masks[1]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3,
                                (apik)* (x[non_zero_masks[2]] - mupi)**2 * np.exp(-(x[non_zero_masks[2]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3,
                                    (apiinc) * (x[non_zero_masks[3]] - mupi)**2 * np.exp(-(x[non_zero_masks[3]] - mupi)**2 / (2 * sigpi**2)) / sigpi**3
                                ])
    partial_F_sigk = np.hstack([( akpi) * ((2*(x[non_zero_masks[0]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[0]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[0]]-muk) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[0]]-muk)**2 / (2 * sigk**2))),
                                (akp) * ((2*(x[non_zero_masks[1]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[1]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[1]]-muk) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[1]]-muk)**2 / (2 * sigk**2))),
                                (akk) * ((2*(x[non_zero_masks[2]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[2]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[2]]-muk) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[2]]-muk)**2 / (2 * sigk**2))),
                                    (akinc) * ((2*(x[non_zero_masks[3]]-muk)**2)/(2*sigk) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2))*(1 + erf(alphak*(x[non_zero_masks[3]]-muk)/(sigk*2**0.5))) - (2/np.pi)**.5 * alphak/sigk**2 * (x[non_zero_masks[3]]-muk) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[3]]-muk)**2 / (2 * sigk**2)))
                                ])

    partial_F_alphap = np.hstack([(appi) * (2/np.pi)**.5 * (x[non_zero_masks[0]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[0]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[0]]-mup)**2 / (2 * sigp**2)),
                                    (app) * (2/np.pi)**.5 * (x[non_zero_masks[1]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[1]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[1]]-mup)**2 / (2 * sigp**2)),
                                    (apk) * (2/np.pi)**.5 * (x[non_zero_masks[2]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[2]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[2]]-mup)**2 / (2 * sigp**2)),
                                    (apinc) * (2/np.pi)**.5 * (x[non_zero_masks[3]]-mup)/(sigp) * np.exp(-(x[non_zero_masks[3]] - mup)**2 / (2 * sigp**2) - alphap**2 * (x[non_zero_masks[3]]-mup)**2 / (2 * sigp**2))
                                ])
    partial_F_alphak = np.hstack([( akpi ) * (2/np.pi)**.5 * (x[non_zero_masks[0]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[0]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[0]]-muk)**2 / (2 * sigk**2)),
                                (akp) * (2/np.pi)**.5 * (x[non_zero_masks[1]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[1]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[1]]-muk)**2 / (2 * sigk**2)),
                                    (akk) * (2/np.pi)**.5 * (x[non_zero_masks[2]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[2]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[2]]-muk)**2 / (2 * sigk**2)),
                                    (akinc) * (2/np.pi)**.5 * (x[non_zero_masks[3]]-muk)/(sigk) * np.exp(-(x[non_zero_masks[3]] - muk)**2 / (2 * sigk**2) - alphak**2 * (x[non_zero_masks[3]]-muk)**2 / (2 * sigk**2))
                                ])
  
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
    cov_apinc = pcov[15][15]
    cov_apiinc = pcov[16][16]
    cov_akinc = pcov[17][17]
    cov_alphap = pcov[18][18]
    cov_alphak = pcov[19][19]


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
    cov_mup_apinc = pcov[0][15]
    cov_mup_apiinc = pcov[0][16]
    cov_mup_akinc = pcov[0][17]
    cov_mup_alphap = pcov[0][18]
    cov_mup_alphak = pcov[0][19]


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
    cov_mupi_apinc = pcov[1][15]
    cov_mupi_apiinc = pcov[1][16]
    cov_mupi_akinc = pcov[1][17]
    cov_mupi_alphap = pcov[1][18]
    cov_mupi_alphak = pcov[1][19]


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
    cov_muk_apinc = pcov[2][15]
    cov_muk_apiinc = pcov[2][16]
    cov_muk_akinc = pcov[2][17]
    cov_muk_alphap = pcov[2][18]
    cov_muk_alphak = pcov[2][19]


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
    cov_sigp_apinc = pcov[3][15]
    cov_sigp_apiinc = pcov[3][16]
    cov_sigp_akinc = pcov[3][17]
    cov_sigp_alphap = pcov[3][18]
    cov_sigp_alphak = pcov[3][19]


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
    cov_sigpi_apinc = pcov[4][15]
    cov_sigpi_apiinc = pcov[4][16]
    cov_sigpi_akinc = pcov[4][17]
    cov_sigpi_alphap = pcov[4][18]
    cov_sigpi_alphak = pcov[4][19]


    cov_sigk_app = pcov[5][6]
    cov_sigk_apip = pcov[5][7]
    cov_sigk_akp = pcov[5][8]
    cov_sigk_appi = pcov[5][9]
    cov_sigk_apipi = pcov[5][10]
    cov_sigk_akpi = pcov[5][11]
    cov_sigk_apk = pcov[5][12]
    cov_sigk_apik = pcov[5][13]
    cov_sigk_akk = pcov[5][14]
    cov_sigk_apinc = pcov[5][15]
    cov_sigk_apiinc = pcov[5][16]
    cov_sigk_akinc = pcov[5][17]
    cov_sigk_alphap = pcov[5][18]
    cov_sigk_alphak = pcov[5][19]


    cov_app_apip = pcov[6][7]
    cov_app_akp = pcov[6][8]
    cov_app_appi = pcov[6][9]
    cov_app_apipi = pcov[6][10]
    cov_app_akpi = pcov[6][11]
    cov_app_apk = pcov[6][12]
    cov_app_apik = pcov[6][13]
    cov_app_akk = pcov[6][14]
    cov_app_apinc = pcov[6][15]
    cov_app_apiinc = pcov[6][16]
    cov_app_akinc = pcov[6][17]
    cov_app_alphap = pcov[6][18]
    cov_app_alphak = pcov[6][19]


    cov_apip_akp = pcov[7][8]
    cov_apip_appi = pcov[7][9]
    cov_apip_apipi = pcov[7][10]
    cov_apip_akpi = pcov[7][11]
    cov_apip_apk = pcov[7][12]
    cov_apip_apik = pcov[7][13]
    cov_apip_akk = pcov[7][14]
    cov_apip_apinc = pcov[7][15]
    cov_apip_apiinc = pcov[7][16]
    cov_apip_akinc = pcov[7][17]
    cov_apip_alphap = pcov[7][18]
    cov_apip_alphak = pcov[7][19]


    cov_akp_appi = pcov[8][9]
    cov_akp_apipi = pcov[8][10]
    cov_akp_akpi = pcov[8][11]
    cov_akp_apk = pcov[8][12]
    cov_akp_apik = pcov[8][13]
    cov_akp_akk = pcov[8][14]
    cov_akp_apinc = pcov[8][15]
    cov_akp_apiinc = pcov[8][16]
    cov_akp_akinc = pcov[8][17]
    cov_akp_alphap = pcov[8][18]
    cov_akp_alphak = pcov[8][19]


    cov_appi_apipi = pcov[9][10]
    cov_appi_akpi = pcov[9][11]
    cov_appi_apk = pcov[9][12]
    cov_appi_apik = pcov[9][13]
    cov_appi_akk = pcov[9][14]
    cov_appi_apinc = pcov[9][15]
    cov_appi_apiinc = pcov[9][16]
    cov_appi_akinc = pcov[9][17]
    cov_appi_alphap = pcov[9][18]
    cov_appi_alphak = pcov[9][19]


    cov_apipi_akpi = pcov[10][11]
    cov_apipi_apk = pcov[10][12]
    cov_apipi_apik = pcov[10][13]
    cov_apipi_akk = pcov[10][14]
    cov_apipi_apinc = pcov[10][15]
    cov_apipi_apiinc = pcov[10][16]
    cov_apipi_akinc = pcov[10][17]
    cov_apipi_alphap = pcov[10][18]
    cov_apipi_alphak = pcov[10][19]


    cov_akpi_apk = pcov[11][12]
    cov_akpi_apik = pcov[11][13]
    cov_akpi_akk = pcov[11][14]
    cov_akpi_apinc = pcov[11][15]
    cov_akpi_apiinc = pcov[11][16]
    cov_akpi_akinc = pcov[11][17]
    cov_akpi_alphap = pcov[11][18]
    cov_akpi_alphak = pcov[11][19]


    cov_apk_apik = pcov[12][13]
    cov_apk_akk = pcov[12][14]
    cov_apk_apinc = pcov[12][15]
    cov_apk_apiinc = pcov[12][16]
    cov_apk_akinc = pcov[12][17]
    cov_apk_alphap = pcov[12][18]
    cov_apk_alphak = pcov[12][19]


    cov_apik_akk = pcov[13][14]
    cov_apik_apinc = pcov[13][15]
    cov_apik_apiinc = pcov[13][16]
    cov_apik_akinc = pcov[13][17]
    cov_apik_alphap = pcov[13][18]
    cov_apik_alphak = pcov[13][19]


    cov_akk_apinc = pcov[14][15]
    cov_akk_apiinc = pcov[14][16]
    cov_akk_akinc = pcov[14][17]
    cov_akk_alphap = pcov[14][18]
    cov_akk_alphak = pcov[14][19]


    cov_apinc_apiinc = pcov[15][16]
    cov_apinc_akinc = pcov[15][17]
    cov_apinc_alphap = pcov[15][18]
    cov_apinc_alphak = pcov[15][19]


    cov_apiinc_akinc = pcov[16][17]
    cov_apiinc_alphap = pcov[16][18]
    cov_apiinc_alphak = pcov[16][19]


    cov_akinc_alphap = pcov[17][18]
    cov_akinc_alphak = pcov[17][19]


    cov_alphap_alphak = pcov[18][19]



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
        partial_F_apinc**2*cov_apinc +
        partial_F_apiinc**2*cov_apiinc +
        partial_F_akinc**2*cov_akinc +
        partial_F_alphap**2*cov_alphap +
        partial_F_alphak**2*cov_alphak +
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
        2*partial_F_mup*partial_F_apinc*cov_mup_apinc +
        2*partial_F_mup*partial_F_apiinc*cov_mup_apiinc +
        2*partial_F_mup*partial_F_akinc*cov_mup_akinc +
        2*partial_F_mup*partial_F_alphap*cov_mup_alphap +
        2*partial_F_mup*partial_F_alphak*cov_mup_alphak +
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
        2*partial_F_mupi*partial_F_apinc*cov_mupi_apinc +
        2*partial_F_mupi*partial_F_apiinc*cov_mupi_apiinc +
        2*partial_F_mupi*partial_F_akinc*cov_mupi_akinc +
        2*partial_F_mupi*partial_F_alphap*cov_mupi_alphap +
        2*partial_F_mupi*partial_F_alphak*cov_mupi_alphak +
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
        2*partial_F_muk*partial_F_apinc*cov_muk_apinc +
        2*partial_F_muk*partial_F_apiinc*cov_muk_apiinc +
        2*partial_F_muk*partial_F_akinc*cov_muk_akinc +
        2*partial_F_muk*partial_F_alphap*cov_muk_alphap +
        2*partial_F_muk*partial_F_alphak*cov_muk_alphak +
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
        2*partial_F_sigp*partial_F_apinc*cov_sigp_apinc +
        2*partial_F_sigp*partial_F_apiinc*cov_sigp_apiinc +
        2*partial_F_sigp*partial_F_akinc*cov_sigp_akinc +
        2*partial_F_sigp*partial_F_alphap*cov_sigp_alphap +
        2*partial_F_sigp*partial_F_alphak*cov_sigp_alphak +
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
        2*partial_F_sigpi*partial_F_apinc*cov_sigpi_apinc +
        2*partial_F_sigpi*partial_F_apiinc*cov_sigpi_apiinc +
        2*partial_F_sigpi*partial_F_akinc*cov_sigpi_akinc +
        2*partial_F_sigpi*partial_F_alphap*cov_sigpi_alphap +
        2*partial_F_sigpi*partial_F_alphak*cov_sigpi_alphak +
        2*partial_F_sigk*partial_F_app*cov_sigk_app +
        2*partial_F_sigk*partial_F_apip*cov_sigk_apip +
        2*partial_F_sigk*partial_F_akp*cov_sigk_akp +
        2*partial_F_sigk*partial_F_appi*cov_sigk_appi +
        2*partial_F_sigk*partial_F_apipi*cov_sigk_apipi +
        2*partial_F_sigk*partial_F_akpi*cov_sigk_akpi +
        2*partial_F_sigk*partial_F_apk*cov_sigk_apk +
        2*partial_F_sigk*partial_F_apik*cov_sigk_apik +
        2*partial_F_sigk*partial_F_akk*cov_sigk_akk +
        2*partial_F_sigk*partial_F_apinc*cov_sigk_apinc +
        2*partial_F_sigk*partial_F_apiinc*cov_sigk_apiinc +
        2*partial_F_sigk*partial_F_akinc*cov_sigk_akinc +
        2*partial_F_sigk*partial_F_alphap*cov_sigk_alphap +
        2*partial_F_sigk*partial_F_alphak*cov_sigk_alphak +
        2*partial_F_app*partial_F_apip*cov_app_apip +
        2*partial_F_app*partial_F_akp*cov_app_akp +
        2*partial_F_app*partial_F_appi*cov_app_appi +
        2*partial_F_app*partial_F_apipi*cov_app_apipi +
        2*partial_F_app*partial_F_akpi*cov_app_akpi +
        2*partial_F_app*partial_F_apk*cov_app_apk +
        2*partial_F_app*partial_F_apik*cov_app_apik +
        2*partial_F_app*partial_F_akk*cov_app_akk +
        2*partial_F_app*partial_F_apinc*cov_app_apinc +
        2*partial_F_app*partial_F_apiinc*cov_app_apiinc +
        2*partial_F_app*partial_F_akinc*cov_app_akinc +
        2*partial_F_app*partial_F_alphap*cov_app_alphap +
        2*partial_F_app*partial_F_alphak*cov_app_alphak +
        2*partial_F_apip*partial_F_akp*cov_apip_akp +
        2*partial_F_apip*partial_F_appi*cov_apip_appi +
        2*partial_F_apip*partial_F_apipi*cov_apip_apipi +
        2*partial_F_apip*partial_F_akpi*cov_apip_akpi +
        2*partial_F_apip*partial_F_apk*cov_apip_apk +
        2*partial_F_apip*partial_F_apik*cov_apip_apik +
        2*partial_F_apip*partial_F_akk*cov_apip_akk +
        2*partial_F_apip*partial_F_apinc*cov_apip_apinc +
        2*partial_F_apip*partial_F_apiinc*cov_apip_apiinc +
        2*partial_F_apip*partial_F_akinc*cov_apip_akinc +
        2*partial_F_apip*partial_F_alphap*cov_apip_alphap +
        2*partial_F_apip*partial_F_alphak*cov_apip_alphak +
        2*partial_F_akp*partial_F_appi*cov_akp_appi +
        2*partial_F_akp*partial_F_apipi*cov_akp_apipi +
        2*partial_F_akp*partial_F_akpi*cov_akp_akpi +
        2*partial_F_akp*partial_F_apk*cov_akp_apk +
        2*partial_F_akp*partial_F_apik*cov_akp_apik +
        2*partial_F_akp*partial_F_akk*cov_akp_akk +
        2*partial_F_akp*partial_F_apinc*cov_akp_apinc +
        2*partial_F_akp*partial_F_apiinc*cov_akp_apiinc +
        2*partial_F_akp*partial_F_akinc*cov_akp_akinc +
        2*partial_F_akp*partial_F_alphap*cov_akp_alphap +
        2*partial_F_akp*partial_F_alphak*cov_akp_alphak +
        2*partial_F_appi*partial_F_apipi*cov_appi_apipi +
        2*partial_F_appi*partial_F_akpi*cov_appi_akpi +
        2*partial_F_appi*partial_F_apk*cov_appi_apk +
        2*partial_F_appi*partial_F_apik*cov_appi_apik +
        2*partial_F_appi*partial_F_akk*cov_appi_akk +
        2*partial_F_appi*partial_F_apinc*cov_appi_apinc +
        2*partial_F_appi*partial_F_apiinc*cov_appi_apiinc +
        2*partial_F_appi*partial_F_akinc*cov_appi_akinc +
        2*partial_F_appi*partial_F_alphap*cov_appi_alphap +
        2*partial_F_appi*partial_F_alphak*cov_appi_alphak +
        2*partial_F_apipi*partial_F_akpi*cov_apipi_akpi +
        2*partial_F_apipi*partial_F_apk*cov_apipi_apk +
        2*partial_F_apipi*partial_F_apik*cov_apipi_apik +
        2*partial_F_apipi*partial_F_akk*cov_apipi_akk +
        2*partial_F_apipi*partial_F_apinc*cov_apipi_apinc +
        2*partial_F_apipi*partial_F_apiinc*cov_apipi_apiinc +
        2*partial_F_apipi*partial_F_akinc*cov_apipi_akinc +
        2*partial_F_apipi*partial_F_alphap*cov_apipi_alphap +
        2*partial_F_apipi*partial_F_alphak*cov_apipi_alphak +
        2*partial_F_akpi*partial_F_apk*cov_akpi_apk +
        2*partial_F_akpi*partial_F_apik*cov_akpi_apik +
        2*partial_F_akpi*partial_F_akk*cov_akpi_akk +
        2*partial_F_akpi*partial_F_apinc*cov_akpi_apinc +
        2*partial_F_akpi*partial_F_apiinc*cov_akpi_apiinc +
        2*partial_F_akpi*partial_F_akinc*cov_akpi_akinc +
        2*partial_F_akpi*partial_F_alphap*cov_akpi_alphap +
        2*partial_F_akpi*partial_F_alphak*cov_akpi_alphak +
        2*partial_F_apk*partial_F_apik*cov_apk_apik +
        2*partial_F_apk*partial_F_akk*cov_apk_akk +
        2*partial_F_apk*partial_F_apinc*cov_apk_apinc +
        2*partial_F_apk*partial_F_apiinc*cov_apk_apiinc +
        2*partial_F_apk*partial_F_akinc*cov_apk_akinc +
        2*partial_F_apk*partial_F_alphap*cov_apk_alphap +
        2*partial_F_apk*partial_F_alphak*cov_apk_alphak +
        2*partial_F_apik*partial_F_akk*cov_apik_akk + 
        2*partial_F_apik*partial_F_apinc*cov_apik_apinc +
        2*partial_F_apik*partial_F_apiinc*cov_apik_apiinc +
        2*partial_F_apik*partial_F_akinc*cov_apik_akinc +
        2*partial_F_apik*partial_F_alphap*cov_apik_alphap +
        2*partial_F_apik*partial_F_alphak*cov_apik_alphak +
        2*partial_F_akk*partial_F_apinc*cov_akk_apinc +
        2*partial_F_akk*partial_F_apiinc*cov_akk_apiinc +
        2*partial_F_akk*partial_F_akinc*cov_akk_akinc +
        2*partial_F_akk*partial_F_alphap*cov_akk_alphap +
        2*partial_F_akk*partial_F_alphak*cov_akk_alphak +
        2*partial_F_apinc*partial_F_apiinc*cov_apinc_apiinc +
        2*partial_F_apinc*partial_F_akinc*cov_apinc_akinc +
        2*partial_F_apinc*partial_F_alphap*cov_apinc_alphap +
        2*partial_F_apinc*partial_F_alphak*cov_apinc_alphak +
        2*partial_F_apiinc*partial_F_akinc*cov_apiinc_akinc +  
        2*partial_F_apiinc*partial_F_alphap*cov_apiinc_alphap +
        2*partial_F_apiinc*partial_F_alphak*cov_apiinc_alphak +
        2*partial_F_akinc*partial_F_alphap*cov_akinc_alphap +
        2*partial_F_akinc*partial_F_alphak*cov_akinc_alphak +
        2*partial_F_alphap*partial_F_alphak*cov_alphap_alphak
    )

    return uncertainty_F
