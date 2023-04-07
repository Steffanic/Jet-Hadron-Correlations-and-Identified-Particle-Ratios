
import numpy as np
import matplotlib.pyplot as plt
class RPF:
    def __init__(self, p0=[1, 0.005, 0.02, 0.005, 0.02, 0.05, 0.03,]):
        print(f"Intializing RPF method with parameters: {list(zip(['B', 'v1', 'v2', 'v3', 'v4', 'va2', 'va4'], p0))}")
        self.in_plane_func = self.background(0, np.pi/6)
        self.mid_plane_func = self.background(np.pi/4, np.pi/6)
        self.out_of_plane_func = self.background(np.pi/2, np.pi/6)
        self.simultaneous_fit = lambda dPhi, B, v1, v2, v3, v4, va2, va4: np.hstack([self.in_plane_func(dPhi, B, v1, v2, v3, v4, va2, va4), self.mid_plane_func(dPhi, B, v1, v2, v3, v4, va2, va4), self.out_of_plane_func(dPhi, B, v1, v2, v3, v4, va2, va4)])
        self.in_plane_err = self.background_err(0, np.pi/6)
        self.mid_plane_err = self.background_err(np.pi/4, np.pi/6)
        self.out_of_plane_err = self.background_err(np.pi/2, np.pi/6)
        self.simultaneous_fit_err = lambda dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4: np.hstack([self.in_plane_err(dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4), self.mid_plane_err(dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4), self.out_of_plane_err(dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4)])
        # 9 params in fit
        self.p0 = p0 # B, v1, v2, v3, v4, va2, va4
        self.bounds = [[0.0, -0.0001, -0.2, -0.2, -0.1, 0.0005, 0.0], [1e6, 0.0001, 0.2, 0.1, 0.10, 0.40, 0.10]] # B, v1, v2, v3, v4, va2, va4
        self.popt = None
        self.pcov = np.zeros((7,7))
        self.chi2OverNDF = None
        self.R2 = 0.5713
        self.R4 = 0.2972
        self.R6 = 0.1315 # random guess
        self.R8 = 0.0500 # random guess
        

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['simultaneous_fit']
        del state['in_plane_func']
        del state['mid_plane_func']
        del state['out_of_plane_func']
        del state['simultaneous_fit_err']
        del state['in_plane_err']
        del state['mid_plane_err']
        del state['out_of_plane_err']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.in_plane_func = self.background(0, np.pi/6)
        self.mid_plane_func = self.background(np.pi/4, np.pi/6)
        self.out_of_plane_func = self.background(np.pi/2, np.pi/6)
        self.simultaneous_fit = lambda dPhi, B, v1, v2, v3, v4, va2, va4: np.hstack([self.in_plane_func(dPhi, B, v1, v2, v3, v4, va2, va4), self.mid_plane_func(dPhi, B, v1, v2, v3, v4, va2, va4), self.out_of_plane_func(dPhi, B, v1, v2, v3, v4, va2, va4)])
        self.in_plane_err = self.background_err(0, np.pi/6)
        self.mid_plane_err = self.background_err(np.pi/4, np.pi/6)
        self.out_of_plane_err = self.background_err(np.pi/2, np.pi/6)
        self.simultaneous_fit_err = lambda dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4: np.hstack([self.in_plane_err(dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4), self.mid_plane_err(dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4), self.out_of_plane_err(dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4)])

    
    def vtilde(self, n, phis, c):
        def vtilde_(v1, v2, v3, v4):
            denom = (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)
            if n<=0 or n>=5: return 0
            if n==1:
                return v1 
            if n==2:
                return (v2 + np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + (v4)*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + (v2)*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4 + (v4)*np.cos(6*phis)*np.sin(6*c)/(6*c)*self.R6) / denom
            if n==3:
                return v3 
            if n==4:
                return (v4 + np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4 + (v2)*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + (v2)*np.cos(6*phis)*np.sin(6*c)/(6*c)*self.R6 + (v4)*np.cos(8*phis)*np.sin(8*c)/(8*c)*self.R8) / denom
            else:
                return 0
        return vtilde_

    def btilde(self, phis, c):
        def btilde_(B, v2, v4):
            return B*(1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)
        return btilde_

    def background(self,phis, c):
        def background_(dPhi, B, v1, v2, v3, v4,  va2, va4):
            btilde_func = self.btilde(phis, c)
            return btilde_func(B, v2, v4)*(1+2*self.vtilde(3, phis, c)(v1, v2, v3, v4)*np.cos(1*dPhi) +2*self.vtilde(2, phis, c)(v1, v2, v3, v4)*va2*np.cos(2*dPhi) +2*self.vtilde(3, phis, c)(v1, v2, v3, v4)*np.cos(3*dPhi) + 2*self.vtilde(4, phis, c)(v1, v2, v3, v4)*va4*np.cos(4*dPhi))
        return background_

    def background_err(self, phis, c):
        
        def background_err_(dPhi, dPhiErr, B, v1, v2, v3, v4, va2, va4):
            background = self.background(phis, c)
            btilde_func = self.btilde(phis, c)

            #dbddPhi = np.pi*btilde_func(B, v2, v4)*(-2*v1*np.sin(1*dPhi) -4*self.vtilde(2, phis, c)(v1, v2, v3, v4)*va2*np.sin(2*dPhi) -6*v3*np.sin(3*dPhi) - 8*self.vtilde(4, phis, c)(v2, v4)*va4*np.sin(4*dPhi))

            dbdB = background(dPhi, B, v1, v2, v3, v4, va2, va4)/B

            dbdv1 = btilde_func(B, v2, v4)*2*self.vtilde(1, phis, c)(v1, v2, v3, v4)*np.cos(1*dPhi)/v1


            v2_first_term = (-(2*self.R2*va2*np.cos(2*dPhi)*np.cos(2*phis)*np.sin(2*c)*(v2+self.R2*np.cos(2*phis)*np.sin(2*c)/(2*c) + self.R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(c*(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/2*c +2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2))
            v2_second_term = ((2*va2*np.cos(2*dPhi)*(1+self.R4*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
            v2_third_term = ((2*va4*np.cos(4*dPhi)*(self.R2*np.cos(2*phis)*np.sin(2*c)/(2*c) + self.R6*np.cos(6*phis)*np.sin(6*c)/(6*c)))/(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
            v2_fourth_term = -(2*self.R2*va4*np.cos(4*dPhi)*np.cos(2*phis)*np.sin(2*c)*(v4+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + self.R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+self.R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+self.R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c)))/(c*(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2)
            v2_bonus_term = 1/c*B*self.R2*np.cos(2*phis)*np.sin(2*c)*(1+2*v1*np.cos(dPhi)+2*v3*np.cos(3*dPhi) + (2*va2*np.cos(2*dPhi)*(v2+self.R2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(1+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))+2*va4*np.cos(4*dPhi)*(v4+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+self.R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+self.R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c))/(1+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
            
            dbdv2 = btilde_func(B, v2, v4)*(v2_first_term + v2_second_term + v2_third_term + v2_fourth_term) + v2_bonus_term
            #dbdv2 = btilde_func(B, v2, v4)*2*(va2*np.cos(2*dPhi)*(
            #    (1 + np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)
            #    /
            #    (1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)
            #            -(v2 + np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + (v4)*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + (v2)*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4 )*(2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2))
            #                /
            #                (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)**2 
            #     - va4*np.cos(4*dPhi)*((v4 + np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4  )*(2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2))
            #     /
            #     (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)**2) + np.pi*(B*c*2/np.pi*2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2)*(1+2*v1*np.cos(1*dPhi) +2*self.vtilde(2, phis, c)(v1, v2, v3, v4)*va2*np.cos(2*dPhi) +2*v3*np.cos(3*dPhi) + 2*self.vtilde(4, phis, c)(v1, v2, v3, v4)*va4*np.cos(4*dPhi))

            dbdv3 = btilde_func(B, v2, v4)*self.vtilde(3, phis, c)(v1, v2, v3, v4)*2*np.cos(3*dPhi)

            v4_first_term = (-(2*self.R4*va2*np.cos(2*dPhi)*np.cos(4*phis)*np.sin(4*c)*(v2+self.R2*np.cos(2*phis)*np.sin(2*c)/(2*c) + self.R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(c*(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/2*c +2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2))
            v4_second_term = ((self.R2*va2*np.cos(2*dPhi)*np.cos(2*phis)*np.sin(2*c)/(2*c))/(c*(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))))
            v4_third_term = ((2*va4*np.cos(4*dPhi)*(1+self.R8*np.cos(8*phis)*np.sin(8*c)/(8*c)))/(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
            v4_fourth_term = -(self.R4*va4*np.cos(4*dPhi)*np.cos(4*phis)*np.sin(4*c)*(v4+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + self.R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+self.R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+self.R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c)))/(c*(1 + 2*self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2)
            v4_bonus_term = 1/(2*c)*B*self.R4*np.cos(4*phis)*np.sin(4*c)*(1+2*v1*np.cos(dPhi)+2*v3*np.cos(3*dPhi) + (2*va2*np.cos(2*dPhi)*(v2+self.R2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(1+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))+2*va4*np.cos(4*dPhi)*(v4+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+self.R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+self.R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c))/(1+self.R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+self.R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
            
            dbdv4 = btilde_func(B, v2, v4)*(v4_first_term + v4_second_term + v4_third_term + v4_fourth_term) + v4_bonus_term

            #dbdv4 = np.pi*btilde_func(B, v2, v4)*2*(va2*np.cos(2*dPhi)*(
            #    (np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2)*(1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4) 
            #    - (v2 +np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 +v4*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 +v2*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)*(2*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4))
            #    /
            #    (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)**2 
            #    + va4*np.cos(4*dPhi)*((1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)-(v4+np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)*(2*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4))
            #    /
            #    (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*self.R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)**2) + np.pi*(B*c*2/np.pi*2*np.cos(4*phis)*np.sin(4*c)/(4*c)*self.R4)*(1+2*v1*np.cos(1*dPhi) +2*self.vtilde(2, phis, c)(v1, v2, v3, v4)*va2*np.cos(2*dPhi) +2*v3*np.cos(3*dPhi) + 2*self.vtilde(4, phis, c)(v1, v2, v3, v4)*va4*np.cos(4*dPhi))

            dbdva2 = np.pi*btilde_func(B, v2, v4)*2*self.vtilde(2, phis, c)(v1, v2, v3, v4)*np.cos(2*dPhi)

            dbdva4 = np.pi*btilde_func(B, v2, v4)*2*self.vtilde(4, phis, c)(v1, v2, v3, v4)*np.cos(4*dPhi)

            err = np.sqrt( (dbdB**2 * self.pcov[0,0]) + (dbdv1**2 * self.pcov[1,1]) + (dbdv2**2 * self.pcov[2,2]) + (dbdv3**2 * self.pcov[3,3]) + (dbdv4**2 * self.pcov[4,4]) + (dbdva2**2 * self.pcov[5,5]) + (dbdva4**2 * self.pcov[6,6]) +  \
                +2*(dbdB*dbdv1 * self.pcov[0,1]) + 2*(dbdB*dbdv2 * self.pcov[0,2]) + 2*(dbdB*dbdv3 * self.pcov[0,3])+2*(dbdB*dbdv4 * self.pcov[0,4])+2*(dbdB*dbdva2 * self.pcov[0,5])+2*(dbdB*dbdva4 * self.pcov[0,6])\
                    + 2*(dbdv1*dbdv2 * self.pcov[1,2]) + 2*(dbdv1*dbdv3 * self.pcov[1,3])+2*(dbdv1*dbdv4 * self.pcov[1,4])+2*(dbdv1*dbdva2 * self.pcov[1,5])+2*(dbdv1*dbdva4 * self.pcov[1,6])\
                        + 2*(dbdv2*dbdv3 * self.pcov[2,3])+2*(dbdv2*dbdv4 * self.pcov[2,4])+2*(dbdv2*dbdva2 * self.pcov[2,5])+2*(dbdv2*dbdva4 * self.pcov[2,6])
                        + 2*(dbdv3*dbdv4 * self.pcov[3,4])+2*(dbdv3*dbdva2 * self.pcov[3,5])+2*(dbdv3*dbdva4 * self.pcov[3,6])
                        + 2*(dbdv4*dbdva2 * self.pcov[4,5])+2*(dbdv4*dbdva4 * self.pcov[4,6])
                        + 2*(dbdva2*dbdva4 * self.pcov[5,6]))
                       
            return err
        return background_err_

    def fit(self, x, y, yerr):
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(self.simultaneous_fit, x, y, sigma=[yerr_i or 1 for yerr_i in yerr], p0=self.p0, bounds=self.bounds,absolute_sigma=True, maxfev=1000000)


        NDF = len(y) - len(popt) - 1
        chi2 = []
        for i in range(len(x)):
            chi2.append(np.sum((np.array([y[i], y[i+len(x)], y[i+2*len(x)]]) - self.simultaneous_fit(x[i], *popt))**2)/(yerr[i] or 1)**2)
        chi2 = np.array(np.sum(chi2))
        self.chi2OverNDF = chi2/NDF
        self.popt = popt
        self.pcov = pcov
        print(f"{chi2=}, {NDF=}")
        plt.plot(x, [np.sqrt(np.sum((y[i]-self.simultaneous_fit(x[i], *popt))**2))/(yerr[i] or 1) for i in range(len(x))])
        plt.savefig("residuals.png")
        return popt, pcov, chi2/NDF

    