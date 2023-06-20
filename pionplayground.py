import ROOT 
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os
# load the pp file
#f = ROOT.TFile("/mnt/d/pp/17p.root")
f = ROOT.TFile("/mnt/d/18q/295754.root")
# get the tree
#emcalList = f.Get("AliAnalysisTaskJetH_tracks_caloClusters_biased")
emcalList = f.Get("AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2SemiCentralq")
# get the histogram
JH = emcalList.FindObject("fhnJH")
# JH is a ThNsparse with 10 dimensions. Let's get axis 8 and setrangeuser to +-2 

pTtrigBinEdges = [
            # 10,
            20,
            40,
            60,
        ]
pTassocBinEdges = [
            # 0.15,
            # 0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            10.0,
        ]
for pt_trig_bin in range(len(pTtrigBinEdges)-1):
    for pt_assoc_bin in range(len(pTassocBinEdges)-1):
        if not os.path.exists(f"./plots/{pt_trig_bin}_{pt_assoc_bin}"):
            os.makedirs(f"./plots/{pt_trig_bin}_{pt_assoc_bin}")
        JH.GetAxis(1).SetRangeUser(pTtrigBinEdges[pt_trig_bin],pTtrigBinEdges[pt_trig_bin+1])
        JH.GetAxis(2).SetRangeUser(pTassocBinEdges[pt_assoc_bin],pTassocBinEdges[pt_assoc_bin+1])
        # get the axis
        JH.GetAxis(9).SetRangeUser(-2,2)
        # then project the histogram to get a TH3D with 3 dimensions    and axes 3,4,6
        th3d_proton = JH.Projection(3,4,7).Clone()
        JH.GetAxis(9).SetRangeUser(-100,100)
        JH.GetAxis(8).SetRangeUser(-2,2)
        th3d_pion = JH.Projection(3,4,7).Clone()
        JH.GetAxis(8).SetRangeUser(-100,100)
        JH.GetAxis(10).SetRangeUser(-2,2)
        th3d_kaon = JH.Projection(3,4,7).Clone()
        JH.GetAxis(10).SetRangeUser(-100,100)
        # lets make a z projection
        th1d1_proton = th3d_proton.ProjectionZ("proton")
        th1d1_pion = th3d_pion.ProjectionZ("pion")
        th1d1_kaon = th3d_kaon.ProjectionZ("kaon")
        nbins_proton = th1d1_proton.GetNbinsX()
        nbins_pion = th1d1_pion.GetNbinsX()
        nbins_kaon = th1d1_kaon.GetNbinsX()
        # now make a plot
        plt.errorbar(np.linspace(-2,2,nbins_proton),[th1d1_proton.GetBinContent(k) for k in range(1,nbins_proton+1)], yerr=[th1d1_proton.GetBinError(k) for k in range(1,nbins_proton+1)],fmt='o')
        plt.xlabel("Pion NSigma")
        plt.ylabel("Counts")
        plt.title("Proton-Enhanced")
        plt.savefig(f"./plots/{pt_trig_bin}_{pt_assoc_bin}/plot_proton.png")
        plt.clf()

        plt.errorbar(np.linspace(-2,2,nbins_pion),[th1d1_pion.GetBinContent(k) for k in range(1,nbins_pion+1)], yerr=[th1d1_pion.GetBinError(k) for k in range(1,nbins_pion+1)],fmt='o')
        plt.xlabel("Pion NSigma")
        plt.ylabel("Counts")
        plt.title("Pion-Enhanced")
        plt.savefig(f"./plots/{pt_trig_bin}_{pt_assoc_bin}/plot_pion.png")
        plt.clf()

        plt.errorbar(np.linspace(-2,2,nbins_kaon),[th1d1_kaon.GetBinContent(k) for k in range(1,nbins_kaon+1)], yerr=[th1d1_kaon.GetBinError(k) for k in range(1,nbins_kaon+1)],fmt='o')
        plt.xlabel("Pion NSigma")
        plt.ylabel("Counts")
        plt.title("Kaon-Enhanced")
        plt.savefig(f"./plots/{pt_trig_bin}_{pt_assoc_bin}/plot_kaon.png")
        plt.clf()

        # now lets fit the peaks with gaussians 
        # first define the functions
        def gauss(x,mean,sigma,A):
            return A*np.exp(-(x-mean)**2/(2*sigma**2))


        # now lets fit the peaks
        # using scipy curve fit

        from scipy.optimize import curve_fit
        # first define the fit function
        def fitfunc(x,mean1,sigma1,A1,mean2,sigma2,A2, mean3,sigma3,A3):
            return gauss(x,mean1,sigma1,A1)+gauss(x,mean2,sigma2,A2)+gauss(x,mean3,sigma3,A3)

        bounds = [(-0.2,0.2),(-2,2),(0,100000),(-2,-0.05),(-2,2),(0,100000), (-2,-0.05),(-2,2),(0,100000)]
        # now unzip them
        bounds = list(zip(*bounds))
        # now fit the data
        # first get the data
        xdata = np.linspace(-2,2,nbins_proton)
        ydata = [th1d1_proton.GetBinContent(k) for k in range(1,nbins_proton+1)]
        # now fit the data
        popt_proton,pcov_proton = curve_fit(fitfunc,xdata,ydata,p0=[0,0.2,100,-0.5,0.2,100, -0.2, 0.2, 100], bounds=bounds)
        # now plot the data and the fit
        plt.errorbar(xdata,ydata,yerr=[th1d1_proton.GetBinError(k) for k in range(1,nbins_proton+1)],fmt='o')
        plt.plot(xdata,fitfunc(xdata,*popt_proton))
        plt.plot(xdata,gauss(xdata,popt_proton[0],popt_proton[1],popt_proton[2]), label=f"Mean:{popt_proton[0]:.2f} SIgma:{popt_proton[1]:.2f} Area:{popt_proton[2]:.2f} Percentage:{popt_proton[2]/(popt_proton[2]+popt_proton[5]+popt_proton[8])*100:.2f}%")
        plt.plot(xdata,gauss(xdata,popt_proton[3],popt_proton[4],popt_proton[5]), label=f"Mean:{popt_proton[3]:.2f} Sigma:{popt_proton[4]:.2f} Area:{popt_proton[5]:.2f} Percentage:{popt_proton[5]/(popt_proton[2]+popt_proton[5]+popt_proton[8])*100:.2f}%")
        plt.plot(xdata,gauss(xdata,popt_proton[6],popt_proton[7],popt_proton[8]), label=f"Mean:{popt_proton[6]:.2f} Sigma:{popt_proton[7]:.2f} Area:{popt_proton[8]:.2f} Percentage:{popt_proton[8]/(popt_proton[2]+popt_proton[5]+popt_proton[8])*100:.2f}%")
        plt.xlabel("Pion NSigma")
        plt.ylabel("Counts")
        plt.title("Proton-Enhanced")
        plt.legend()
        plt.savefig(f"./plots/{pt_trig_bin}_{pt_assoc_bin}/fit_proton.png")
        plt.clf()

        # now lets fit the pion
        # now lets fit the data
        # first get the data
        xdata = np.linspace(-2,2,nbins_pion)
        ydata = [th1d1_pion.GetBinContent(k) for k in range(1,nbins_pion+1)]
        # now fit the data
        popt_pion,pcov_pion = curve_fit(fitfunc,xdata,ydata,p0=[0,0.2,100,-0.5,0.2,100, -0.2, 0.2, 100], bounds=bounds)
        # now plot the data and the fit
        plt.errorbar(xdata,ydata,yerr=[th1d1_pion.GetBinError(k) for k in range(1,nbins_pion+1)],fmt='o')
        plt.plot(xdata,fitfunc(xdata,*popt_pion))
        plt.plot(xdata,gauss(xdata,popt_pion[0],popt_pion[1],popt_pion[2]), label=f"Mean:{popt_pion[0]:.2f} SIgma:{popt_pion[1]:.2f} Area:{popt_pion[2]:.2f} Percentage:{popt_pion[2]/(popt_pion[2]+popt_pion[5]+popt_pion[8])*100:.2f}%")
        plt.plot(xdata,gauss(xdata,popt_pion[3],popt_pion[4],popt_pion[5]), label=f"Mean:{popt_pion[3]:.2f} Sigma:{popt_pion[4]:.2f} Area:{popt_pion[5]:.2f} Percentage:{popt_pion[5]/(popt_pion[2]+popt_pion[5]+popt_pion[8])*100:.2f}%")
        plt.plot(xdata,gauss(xdata,popt_pion[6],popt_pion[7],popt_pion[8]), label=f"Mean:{popt_pion[6]:.2f} Sigma:{popt_pion[7]:.2f} Area:{popt_pion[8]:.2f} Percentage:{popt_pion[8]/(popt_pion[2]+popt_pion[5]+popt_pion[8])*100:.2f}%")
        plt.xlabel("Pion NSigma")
        plt.ylabel("Counts")
        plt.title("Pion-Enhanced")
        plt.legend()
        plt.savefig(f"./plots/{pt_trig_bin}_{pt_assoc_bin}/fit_pion.png")
        plt.clf()

        # now lets fit the kaon
        # now lets fit the data
        # first get the data
        xdata = np.linspace(-2,2,nbins_kaon)
        ydata = [th1d1_kaon.GetBinContent(k) for k in range(1,nbins_kaon+1)]
        # now fit the data
        popt_kaon,pcov_kaon = curve_fit(fitfunc,xdata,ydata,p0=[0,0.2,100,-0.5,0.2,100, -0.2, 0.2, 100], bounds=bounds)
        # now plot the data and the fit
        plt.errorbar(xdata,ydata,yerr=[th1d1_kaon.GetBinError(k) for k in range(1,nbins_kaon+1)],fmt='o')
        plt.plot(xdata,fitfunc(xdata,*popt_kaon))
        plt.plot(xdata,gauss(xdata,popt_kaon[0],popt_kaon[1],popt_kaon[2]), label=f"Mean:{popt_kaon[0]:.2f} SIgma:{popt_kaon[1]:.2f} Area:{popt_kaon[2]:.2f} Percentage:{popt_kaon[2]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])*100:.2f}%")
        plt.plot(xdata,gauss(xdata,popt_kaon[3],popt_kaon[4],popt_kaon[5]), label=f"Mean:{popt_kaon[3]:.2f} Sigma:{popt_kaon[4]:.2f} Area:{popt_kaon[5]:.2f} Percentage:{popt_kaon[5]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])*100:.2f}%")
        plt.plot(xdata,gauss(xdata,popt_kaon[6],popt_kaon[7],popt_kaon[8]),  label=f"Mean:{popt_kaon[6]:.2f} Sigma:{popt_kaon[7]:.2f} Area:{popt_kaon[8]:.2f} Percentage:{popt_kaon[8]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])*100:.2f}%")
        plt.xlabel("Pion NSigma")
        plt.ylabel("Counts")
        plt.title("Kaon-Enhanced")
        plt.legend()
        plt.savefig(f"./plots/{pt_trig_bin}_{pt_assoc_bin}/fit_kaon.png")
        plt.clf()  


        print("The mean of the first proton peak is: ", popt_proton[0], " +- ", pcov_proton[0][0]**0.5)
        print("The mean of the second proton peak is: ", popt_proton[3], " +- ", pcov_proton[3][3]**0.5)
        print("The mean of the third proton peak is: ", popt_proton[6], " +- ", pcov_proton[6][6]**0.5)

        print("The sigma of the first proton peak is: ", popt_proton[1], )
        print("The sigma of the second proton peak is: ", popt_proton[4])
        print("The sigma of the third proton peak is: ", popt_proton[7])

        print("The area of the first proton peak is: ", popt_proton[2], " +- ", pcov_proton[2][2]**0.5)
        print("The area of the second proton peak is: ", popt_proton[5], " +- ", pcov_proton[5][5]**0.5)
        print("The area of the third proton peak is: ", popt_proton[8], " +- ", pcov_proton[8][8]**0.5)

        print("The percent of the first proton peak is: ", popt_proton[2]/(popt_proton[2]+popt_proton[5]+popt_proton[8]), " +- ", (pcov_proton[2][2]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2+pcov_proton[5][5]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2+pcov_proton[8][8]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2)**0.5)
        print("The percent of the second proton peak is: ", popt_proton[5]/(popt_proton[2]+popt_proton[5]+popt_proton[8]), " +- ", (pcov_proton[2][2]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2+pcov_proton[5][5]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2+pcov_proton[8][8]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2)**0.5)
        print("The percent of the third proton peak is: ", popt_proton[8]/(popt_proton[2]+popt_proton[5]+popt_proton[8]), " +- ", (pcov_proton[2][2]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2+pcov_proton[5][5]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2+pcov_proton[8][8]/(popt_proton[2]+popt_proton[5]+popt_proton[8])**2)**0.5)

        print("The mean of the first pion peak is: ", popt_pion[0], " +- ", pcov_pion[0][0]**0.5)
        print("The mean of the second pion peak is: ", popt_pion[3], " +- ", pcov_pion[3][3]**0.5)
        print("The mean of the third pion peak is: ", popt_pion[6], " +- ", pcov_pion[6][6]**0.5)

        print("The sigma of the first pion peak is: ", popt_pion[1], )
        print("The sigma of the second pion peak is: ", popt_pion[4])
        print("The sigma of the third pion peak is: ", popt_pion[7])

        print("The area of the first pion peak is: ", popt_pion[2], " +- ", pcov_pion[2][2]**0.5)
        print("The area of the second pion peak is: ", popt_pion[5], " +- ", pcov_pion[5][5]**0.5)
        print("The area of the third pion peak is: ", popt_pion[8], " +- ", pcov_pion[8][8]**0.5)

        print("The percent of the first pion peak is: ", popt_pion[2]/(popt_pion[2]+popt_pion[5]+popt_pion[8]), " +- ", (pcov_pion[2][2]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2+pcov_pion[5][5]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2+pcov_pion[8][8]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2)**0.5)
        print("The percent of the second pion peak is: ", popt_pion[5]/(popt_pion[2]+popt_pion[5]+popt_pion[8]), " +- ", (pcov_pion[2][2]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2+pcov_pion[5][5]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2+pcov_pion[8][8]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2)**0.5)
        print("The percent of the third pion peak is: ", popt_pion[8]/(popt_pion[2]+popt_pion[5]+popt_pion[8]), " +- ", (pcov_pion[2][2]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2+pcov_pion[5][5]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2+pcov_pion[8][8]/(popt_pion[2]+popt_pion[5]+popt_pion[8])**2)**0.5)

        print("The mean of the first kaon peak is: ", popt_kaon[0], " +- ", pcov_kaon[0][0]**0.5)
        print("The mean of the second kaon peak is: ", popt_kaon[3], " +- ", pcov_kaon[3][3]**0.5)
        print("The mean of the third kaon peak is: ", popt_kaon[6], " +- ", pcov_kaon[6][6]**0.5)

        print("The sigma of the first kaon peak is: ", popt_kaon[1], )
        print("The sigma of the second kaon peak is: ", popt_kaon[4])
        print("The sigma of the third kaon peak is: ", popt_kaon[7])

        print("The area of the first kaon peak is: ", popt_kaon[2], " +- ", pcov_kaon[2][2]**0.5)
        print("The area of the second kaon peak is: ", popt_kaon[5], " +- ", pcov_kaon[5][5]**0.5)
        print("The area of the third kaon peak is: ", popt_kaon[8], " +- ", pcov_kaon[8][8]**0.5)

        print("The percent of the first kaon peak is: ", popt_kaon[2]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8]), " +- ", (pcov_kaon[2][2]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2+pcov_kaon[5][5]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2+pcov_kaon[8][8]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2)**0.5)
        print("The percent of the second kaon peak is: ", popt_kaon[5]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8]), " +- ", (pcov_kaon[2][2]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2+pcov_kaon[5][5]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2+pcov_kaon[8][8]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2)**0.5)
        print("The percent of the third kaon peak is: ", popt_kaon[8]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8]), " +- ", (pcov_kaon[2][2]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2+pcov_kaon[5][5]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2+pcov_kaon[8][8]/(popt_kaon[2]+popt_kaon[5]+popt_kaon[8])**2)**0.5)


        # now lets fit them simultaneously, they should have the same mean and sigma but different areas 
        # we will use the same initial guesses as before

        # define the function to fit
        def three_gaussians(x, mup, mupi, muk, sigp, sigpi, sigk, ap, api, ak):
            return api*np.exp(-0.5*((x-mupi)/sigpi)**2)+ap*np.exp(-0.5*((x-mup)/sigp)**2)+ak*np.exp(-0.5*((x-muk)/sigk)**2)
        
        def fitfunc(x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk):
            return np.hstack([three_gaussians(x, mup, mupi, muk, sigp, sigpi, sigk, appi, apipi, akpi),three_gaussians(x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp),three_gaussians(x, mup, mupi, muk, sigp, sigpi, sigk, apk, apik, akk)])
        
        # define the initial guesses
        if pt_assoc_bin==0:
            p0 = [.5, 0, .5, 0.1, 0.1, 0.1, 10000,10000, 0.11, 10,10000, 10, 0.11,10000, 10000]
            bounds = [[-2, -0.1, -2, 1e-4,1e-4,1e-4, 10.0,0.1,0.1, 0.1,1.0,0.1, 0.1,0.1,10.0], [2, 0.1, 2, 10.0, 10.0, 10.0, 100000,100000,10,100000,100000,100000,10,100000,100000]]
        else:
            p0 = [-.5, 0, -.3, 0.1, 0.1, 0.1, 10000,10000, 0.11, 10,10000, 10, 0.11,10000, 10000]
            bounds = [[-2, -0.1, -2, 1e-4,1e-4,1e-4, 10.0,0.1,0.1, 0.1,10.0,0.1, 0.1,0.1,10.0], [0, 0.1, 0, 10.0, 10.0, 10.0, 100000,100000,10,100000,100000,100000,10,100000,100000]]

        # fit the data
        xdata = np.linspace(-2,2,nbins_pion)
        ydata = [[th1d1_pion.GetBinContent(k) for k in range(1,nbins_pion+1)], [th1d1_proton.GetBinContent(k) for k in range(1,nbins_proton+1)], [th1d1_kaon.GetBinContent(k) for k in range(1,nbins_kaon+1)]]
        sigma = [[th1d1_pion.GetBinError(k) for k in range(1,nbins_pion+1)], [th1d1_proton.GetBinError(k) for k in range(1,nbins_proton+1)], [th1d1_kaon.GetBinError(k) for k in range(1,nbins_kaon+1)]]

        popt_all, pcov_all = curve_fit(fitfunc, xdata, np.hstack(ydata), p0=p0, bounds=bounds, maxfev=1000000)

        # plot the data and the fit
        plt.figure(figsize=(10,10))
        plt.subplot(3,1,1)
        plt.plot(xdata, ydata[0], 'bo', label='pion')
        plt.subplot(3,1,2)
        plt.plot(xdata, ydata[1], 'ro', label='proton')
        plt.subplot(3,1,3)
        plt.plot(xdata, ydata[2], 'go', label='kaon')
        plt.subplot(3,1,1)
        plt.plot(xdata, fitfunc(xdata, *popt_all)[0:nbins_pion], 'b-', label='pion fit')
        plt.plot(xdata, gauss(xdata, popt_all[1], popt_all[4], popt_all[10]), 'b--', label='pion gaussians')
        plt.plot(xdata, gauss(xdata, popt_all[0], popt_all[3], popt_all[9]), 'r--', label='proton gaussians')
        plt.plot(xdata, gauss(xdata, popt_all[2], popt_all[5], popt_all[11]), 'g--', label='kaon gaussians')
        plt.subplot(3,1,2)
        plt.plot(xdata, fitfunc(xdata, *popt_all)[nbins_pion:2*nbins_pion], 'r-', label='proton fit')
        plt.plot(xdata, gauss(xdata, popt_all[1], popt_all[4], popt_all[7]), 'b--', label='pion gaussians')
        plt.plot(xdata, gauss(xdata, popt_all[0], popt_all[3], popt_all[6]), 'r--', label='proton gaussians')
        plt.plot(xdata, gauss(xdata, popt_all[2], popt_all[5], popt_all[8]), 'g--', label='kaon gaussians')
        plt.subplot(3,1,3)
        plt.plot(xdata, fitfunc(xdata, *popt_all)[nbins_pion*2:3*nbins_pion], 'g-', label='kaon fit')
        plt.plot(xdata, gauss(xdata, popt_all[1], popt_all[4], popt_all[13]), 'b--', label='pion gaussians')
        plt.plot(xdata, gauss(xdata, popt_all[0], popt_all[3], popt_all[12]), 'r--', label='proton gaussians')
        plt.plot(xdata, gauss(xdata, popt_all[2], popt_all[5], popt_all[14]), 'g--', label='kaon gaussians')
        plt.xlabel('x')

        plt.legend()
        plt.savefig(f"./plots/{pt_trig_bin}_{pt_assoc_bin}_fit_all.png")