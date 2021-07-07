import sys
sys.path.append("/home/cvv5110/Desktop/APUS/fiml/")
sys.path.append('/home/cvv5110/Desktop/APUS/fiml/pyaeroutils/')
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from hypate.FluidSolver import SolverDummy2D
from Miscellaneous.miscellaneous import addBW
from RANS.getConditions import getConditions
from Solvers.solvers import solver
from pyaeroutils.supersonicFlow_hyp import analHypLoad
from scipy.interpolate import CubicSpline


class SolverGPR(SolverDummy2D):
    def __init__(self, **kwargs):
        opts = {
            'tvi_dir'  : '/home/cvv5110/Desktop/APUS/GPRs/pflux/D2D4T7T8/BHBCBW/80-datasize/TVI/matern525252-1e-5/',  # Directory containing TVI-GPR models
            'atvi_dir' : '/home/cvv5110/Desktop/APUS/GPRs/pflux/D2D4T7T8/BHBCBW/80-datasize/ATVI/matern525252-1e-3/',
            'level'    : -1,                # Level of fidelity
            'Tini'     : None,              # Initial wall temperature
            'x'        : None               # X-coordinates associated with the output
        }
        
        for key in opts:
            setattr(self, key, opts[key])
        for key in kwargs:
                setattr(self, key, kwargs[key])
        
        # Prepare GPR models
        self.tvi_bh_gpr = pickle.load(open(self.tvi_dir+'BH-GPR.p', 'rb'))
        self.tvi_bc_gpr = pickle.load(open(self.tvi_dir+'BC-GPR.p', 'rb'))
        self.tvi_bw_gpr = pickle.load(open(self.tvi_dir+'BW-GPR.p', 'rb'))
        self.atvi_bh_gpr = pickle.load(open(self.atvi_dir+'BH-GPR.p', 'rb'))
        self.atvi_bc_gpr = pickle.load(open(self.atvi_dir+'BC-GPR.p', 'rb'))
        self.atvi_bw_gpr = pickle.load(open(self.atvi_dir+'BW-GPR.p', 'rb'))
        #self.tvi_scaler = pickle.load(open(self.tvi_dir+'scaler.p', 'rb'))
        #self.atvi_scaler = pickle.load(open(self.atvi_dir+'scaler.p', 'rb'))
        
        self.atvi_solver = solver(c1type='BH', c2type='BC', numDom=2, node_mode='gpr', Minf=self.Minf, Pinf=self.Pinf, Tinf=self.Tinf, Mfront=100, Mpanel=100)
        self.atvi_solver.set_case('GPR')
        self.panel_index = np.where(self.atvi_solver.X == self.atvi_solver.x1)[0][0]
        self.x = self.atvi_solver.X
        self.xHyp = self.x[self.panel_index:]
        self.N = len(self.x)
        self.atvi_solver.ahl = None
        self.atvi_solver.setInit(self.del0, self.Me0)
        
    def initSolution(self, **kwargs):
        self.atvi_solver.ahl = analHypLoad(mach=self.Minf, P=self.Pinf, T=self.Tinf)
        yw = self.hyp2atvi(np.zeros_like(self.xHyp))
        Tw = self.hyp2atvi(273.0*np.ones_like(self.xHyp))
        self.atvi_solver.set_thermoelastic(self.x, yw, Tw, self.panel_index)
        Xo = self.solve_atvi(np.ones((self.N,)), np.ones((self.N,)))
        Xkp1, pw = self.fpi(Xo)
        pstd = self.Pinf*pw[self.panel_index:]
        self.Pcav = pstd
        self.p = np.zeros_like(self.Pcav)
        self.q = Xkp1[self.panel_index:,12]
        # x_transformed 
        
    def timeStepping(self, **kwargs):
        yw, Tw = self.hyp2atvi(self.w, self.TW)
        self.atvi_solver.set_thermoelastic(self.x, yw, Tw, self.panel_index)
        Xo = self.solve_atvi(np.ones((self.N,)), np.ones((self.N,)))
        Xkp1, pw = self.fpi(Xo)
        dyw, pstd = Xkp1[self.panel_index:,4], self.Pinf*pw[self.panel_index:]
        self.p = self.Pcav - pstd # + self.unsteady_corrector(pstd, dyw)
        self.q = Xkp1[self.panel_index:,12]
        print()
        
    def hyp2atvi(self, X):
        M = int(self.N - X.shape[0])
        X_add = X[0]*np.ones((M,))
        X = np.concatenate((X_add, X))
        return X
    
    def getCoord(self):
        x_sub = self.x[self.panel_index:]
        y = x_sub[0] + 0.500
        x_transformed = x_sub - y
        _coor = np.vstack([x_transformed, np.zeros_like(x_transformed)]).T
        return _coor # where self.x is the transformed x
        
    def fpi(self, Xo):
        tvi_bh, tvi_bc, tvi_bw = self.pred_tvi_beta(self.tvi_scaler.specific_transform(Xo, 0, 13))
        tvi_bh, tvi_bc, tvi_bw = self.tvi_scaler.inverse_transform(tvi_bh,-3), self.tvi_scaler.inverse_transform(tvi_bc,-2), self.tvi_scaler.inverse_transform(tvi_bw,-1)
        Xk = self.solve_atvi(tvi_bh, tvi_bc)
        atvi_bh, atvi_bc, atvi_bw = self.pred_atvi_beta(self.atvi_scaler.specific_transform(Xk, 0, 13))
        atvi_bh, atvi_bc, atvi_bw = self.atvi_scaler.inverse_transform(atvi_bh, -3), self.atvi_scaler.inverse_transform(atvi_bc, -2), self.atvi_scaler.inverse_transform(atvi_bw, -1)
        Xkp1 = self.solve_atvi(atvi_bh, atvi_bc)
        Xkp1 = np.hstack((Xkp1, atvi_bw.reshape(-1,1)))
        return Xkp1, atvi_bw*Xkp1[:,2]
        
    def solve_atvi(self, bh, bc): 
        self.atvi_solver.setGPRDomain('front')
        self.atvi_solver.setInit(self.del0, self.Me0)
        self.atvi_solver.defDistribution(C1=bh[0:self.atvi_solver.Mfront], C2=bc[0:self.atvi_solver.Mfront])
        fsol = self.atvi_solver.solve_dae('front')
        fD = self.atvi_solver.getVariables(fsol)
        fye, fMe = self.atvi_solver._getYeMe(fsol)
        self.atvi_solver.setGPRDomain('panel')
        self.atvi_solver.setInit(fye[-1], fMe[-1])
        self.atvi_solver.defDistribution(C1=bh[self.atvi_solver.Mfront:], C2=bc[self.atvi_solver.Mfront:])
        psol = self.atvi_solver.solve_dae('panel')
        pD = self.atvi_solver.getVariables(psol)
        X = np.vstack((fD,pD))
        return X
    
    def pred_tvi_beta(self, X):
        tvi_bh = self.tvi_bh_gpr.predict(X)
        tvi_bc = self.tvi_bc_gpr.predict(X)
        tvi_bw = self.tvi_bw_gpr.predict(X)
        return tvi_bh, tvi_bc, tvi_bw
    
    def pred_atvi_beta(self, X):
        atvi_bh = self.atvi_bh_gpr.predict(X)
        atvi_bc = self.atvi_bc_gpr.predict(X)
        atvi_bw = self.atvi_bw_gpr.predict(X)
        return atvi_bh, atvi_bc, atvi_bw
    
    def setPcav(self, x_cfd, pcav_cfd):
        self.Pcavfit = interp1d(x_cfd, pcav_cfd, kind='cubic', fill_value=(pcav_cfd[0], pcav_cfd[-1]), bounds_error=False)
        print()
        
    def decoupled_solve(self, yw_cfd, Tw_cfd, wt_cfd):
        self.wt = wt_cfd
        yw = self.hyp2atvi(yw_cfd)
        Tw = self.hyp2atvi(Tw_cfd)
        self.wt = self.hyp2atvi(wt_cfd)
        self.atvi_solver.set_thermoelastic(self.x, yw, Tw, self.panel_index)
        Xo = self.solve_atvi(np.ones((self.N,)), np.ones(self.N,))
        Xkp1, pw = self.fpi(Xo)
        dyw, pstd = Xkp1[:,4], self.Pinf*pw - self.Pinf
        puns = self.unsteady_corrector(dyw)
        Pcav = self.hyp2atvi(self.Pcavfit(self.xHyp))
        p = Pcav - (pstd+puns)
        q = Xkp1[:,12]
        return p, q, Xkp1[:,-3], Xkp1[:,-2], Xkp1[:,-1]
    
    def unsteady_corrector(self, dyw):
        gm = self.atvi_solver.gamma
        ainf = self.atvi_solver.Cinf
        Mn2 = self.Minf*dyw + self.wt/ainf
        Mn1 = self.Minf*dyw 
        p1_uns = self.Pinf*(1 + 0.5*(gm-1)*Mn1)**((2*gm)/(gm-1))
        p2_uns = self.Pinf*(1 + 0.5*(gm-1)*Mn2)**((2*gm)/(gm-1))
        corr = p2_uns - p1_uns
        return corr
        
    def match_grid(self, x, yw, Tw, wt):
        wtFit = CubicSpline(x, wt) 
        ywFit = CubicSpline(x, yw)
        TwFit = CubicSpline(x, Tw)
        _wt = wtFit(self.xHyp)
        _yw = ywFit(self.xHyp)
        _Tw = TwFit(self.xHyp)
        return _yw, _Tw, _wt