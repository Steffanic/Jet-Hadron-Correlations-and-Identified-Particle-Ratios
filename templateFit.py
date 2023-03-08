import numpy as np
import matplotlib.pyplot as plt
def gaussian(x, A, mu, sig):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
class templateFit:
    def __init__(self, pTassoc) -> None:
        if pTassoc>2:
            self.initial_amplitude_guesses = [0.01, 0.00000001, 0.000001, 0.000001]
            self.initial_mean_guesses = [0,-5, -5, 8] # pi, K, p, e
            self.initial_sigma_guess = [3.5, 3.5, 3.5, 4.9]
            self.bounds = [[0,0,0,0,-0.2,-8,-8,3,2.5, 2.5, 2.5, 4], [2,0.5,0.5,0.5,0.2,-3,-3,10,6, 4.5, 4.5, 6]]
        else:
            self.initial_amplitude_guesses = [0.01, 0.0000001, 0.0000001, 0.0000001]
            self.initial_mean_guesses = [0,5, 5, 8] # pi, K, p, e
            self.initial_sigma_guess = [3.5, 8.4, 8.4, 4.9]
            self.bounds = [[0,0,0,0,-1.0,0,0,0,1,1,3,3], [2,0.5,0.5,0.5,1.0,20,20,20,6,9,9,6]]
        self.gaussians = lambda x:[gaussian(x, A, mu, sig) for A, mu, sig in zip(self.initial_amplitude_guesses, self.initial_mean_guesses, self.initial_sigma_guess)]


    def plot_gaussians(self):
        x_vals = np.linspace(-10,10)
        [plt.plot(x_vals, self.gaussians(x_vals)[i]) for i in range(4)]
        plt.plot(x_vals, self.sum_of_gaussians()(x_vals, *self.initial_amplitude_guesses, *self.initial_mean_guesses, *self.initial_sigma_guess))
        plt.savefig("Gaussians.png")
        plt.close()

    def sum_of_gaussians(self):
        def sum_o_gauss(x, A1, A2, A3, A4, mu1, mu2, mu3, mu4, sig1, sig2, sig3, sig4):
            gauss1 = gaussian(x, A1, mu1, sig1)
            gauss2 = gaussian(x, A2, mu2, sig2)
            gauss3 = gaussian(x, A3, mu3, sig3)
            gauss4 = gaussian(x, A4, mu4, sig4)
            return np.sum([gauss1, gauss2, gauss3, gauss4], axis=0)
        return sum_o_gauss

    def sum_of_gaussians_error(self):
        def sum_o_gauss_err(x, A1, A2, A3, A4, mu1, mu2, mu3, mu4, sig1, sig2, sig3, sig4):
            dgaussdA1 = gaussian(x, A1, mu1, sig1)/A1
            dgaussdA2 = gaussian(x, A2, mu2, sig2)/A2
            dgaussdA3 = gaussian(x, A3, mu3, sig3)/A3
            dgaussdA4 = gaussian(x, A4, mu4, sig4)/A4
            dgaussdmu1 = -gaussian(x, A1, mu1, sig1)*(x-mu1)/sig1**2
            dgaussdmu2 = -gaussian(x, A2, mu2, sig2)*(x-mu2)/sig2**2
            dgaussdmu3 = -gaussian(x, A3, mu3, sig3)*(x-mu3)/sig3**2
            dgaussdmu4 = -gaussian(x, A4, mu4, sig4)*(x-mu4)/sig4**2
            dgaussdsig1 = -gaussian(x, A1, mu1, sig1)*(x-mu1)**2/sig1**3
            dgaussdsig2 = -gaussian(x, A2, mu2, sig2)*(x-mu2)**2/sig2**3
            dgaussdsig3 = -gaussian(x, A3, mu3, sig3)*(x-mu3)**2/sig3**3
            dgaussdsig4 = -gaussian(x, A4, mu4, sig4)*(x-mu4)**2/sig4**3
            return [dgaussdA1, dgaussdA2, dgaussdA3, dgaussdA4, dgaussdmu1, dgaussdmu2, dgaussdmu3, dgaussdmu4, dgaussdsig1, dgaussdsig2, dgaussdsig3, dgaussdsig4]
        return sum_o_gauss_err

    def gauss_error(self):
        def gauss_err(x, A, mu, sig):
            dgaussdA = gaussian(x, A, mu, sig)/A
            dgaussdmu = -gaussian(x, A, mu, sig)*(x-mu)/sig**2
            dgaussdsig = -gaussian(x, A, mu, sig)*(x-mu)**2/sig**3
            return np.array([dgaussdA, dgaussdmu, dgaussdsig])
        return gauss_err

    def fit_sum_of_gaussians(self, xdata, ydata, yerr, title):
        if len(ydata)==0:
            return
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(self.sum_of_gaussians(), xdata, ydata, sigma=yerr**0.5, p0=[*self.initial_amplitude_guesses, *self.initial_mean_guesses, *self.initial_sigma_guess], bounds=self.bounds, maxfev=100000)
        NDF = len(ydata) - len(popt) - 1
        chi2 = np.sqrt(np.sum(ydata - self.sum_of_gaussians()(xdata, *popt))**2)/(yerr)
        chi2 = np.array(np.sum(chi2))
        self.chi2OverNDF = chi2/NDF

        self.popt = popt
        self.pcov = pcov
        errors = self.sum_of_gaussians_error()(xdata, *popt)
        diag_err = np.sum([errors[i]**2*np.diag(self.pcov)[i] for i in range(len(errors))], axis=0)
        off_diag_err = np.sum([errors[i]*errors[j]*self.pcov[i][j] for i in range(len(errors)) for j in range(len(errors)) if i!=j], axis=0)
        tot_err = np.sqrt(diag_err + off_diag_err)
        print(f"{chi2=}, {NDF=}")
        plt.plot(xdata, [np.sqrt(np.sum((ydata[i]-self.sum_of_gaussians()(xdata[i], *popt))**2))/(yerr[i] or 1) for i in range(len(xdata))])
        plt.savefig("residuals.png")
        plt.close()
        plt.errorbar(xdata, ydata, yerr, fmt="r*")
        bot, top = plt.ylim()
        plt.plot(xdata, self.sum_of_gaussians()(xdata, *popt))
        plt.fill_between(xdata, self.sum_of_gaussians()(xdata, *popt)-tot_err, self.sum_of_gaussians()(xdata, *popt)+tot_err, alpha=0.5)
        plt.ylim(bot, top)
        labels = ["$\\pi$", "K", "p", "e"]
        [plt.plot(xdata, gaussian(xdata, popt[i], popt[i+4], popt[i+8]), label=labels[i]) for i in range(4)]
        gauss_errors = [self.gauss_error()(xdata, popt[i], popt[i+4], popt[i+8]) for i in range(4)]
        [plt.fill_between(xdata, gaussian(xdata, popt[i], popt[i+4], popt[i+8])-np.sqrt(np.sum([self.gauss_error()(xdata, popt[i], popt[i+4], popt[i+8])[j]**2*np.diag(self.pcov)[i::4][j] for j in range(3)], axis=0)), gaussian(xdata, popt[i], popt[i+4], popt[i+8])+np.sqrt(np.sum([self.gauss_error()(xdata, popt[i], popt[i+4], popt[i+8])[j]**2*np.diag(self.pcov)[i::4][j] for j in range(3)], axis=0)), alpha=0.5) for i in range(4)]
        plt.legend()
        plt.title(f"{title}, Chi2/NDF={chi2/NDF}")
        plt.savefig(f"{title[17:]}.png")
        plt.close()
        print(list(zip(("A1", "A2", "A3", "A4", "mu1", "mu2", "mu3", "mu4", "sig1", "sig2", "sig3", "sig4"),popt, np.sqrt(np.diag(pcov)))))
        print("*"*100)
        print("Pion percentage: ", (np.sum(gaussian(xdata, popt[0], popt[4], popt[8]), axis=0)/np.sum(self.sum_of_gaussians()(xdata, *popt), axis=0))*100)
        print("Pion error: ", np.sum((np.sqrt(np.sum([gauss_errors[0][j]**2*np.diag(self.pcov)[j] for j in range(3)], axis=0))/np.sum(self.sum_of_gaussians()(xdata, *popt), axis=0)))*100)
        print("*"*100)
        return popt, pcov, chi2/NDF