import sys
sys.path.append("/home/cvv5110/Desktop/APUS/fiml/")
sys.path.append("/home/cvv5110/Desktop/APUS/fiml/GPRs/")
sys.path.append('/home/cvv5110/Desktop/APUS/fiml/pyaeroutils/')
from scipy.interpolate import interp1d
import numpy as np
import pickle
import sys
from hypate.FluidSolver import SolverDummy2D
from pyaeroutils.supersonicFlow_hyp import analHypLoad
from scipy.interpolate import CubicSpline
from Solvers.piromi_solver import COUPLED_PIROMI
import time


class PIROMI(SolverDummy2D):
    
    def __init__(self, **kwargs):
        opts = {
            'level'          :  -1,                      # Level of fidelity
            'Tini'           : None,                     # Initial wall temperature
            'x'              : None,                     # X-coordinates associated with the output
            'switch_param'   : 0.10*0.01,                # If doing window-GPR, switch sampling window when deformation is past 10% of ref. structural amplitude (0.01m).
            'w0_bool'        : False,
            'w1_bool'        : False,
            'w2_bool'        : False,
            'fpi_iterations' : None                       # Number of FPI iterations in PIROMI solver
        }
        for key in opts:
            setattr(self, key, opts[key])
        for key in kwargs:
                setattr(self, key, kwargs[key])
        
        # Initialize internal PIROMi solver
        self.piromi_solver = COUPLED_PIROMI(**kwargs, node_mode='coupled', domain='panel', noise='nonoise')
        self.piromi_solver.load_low_window()
                
        # Initialize reference
        self.Tref = 380.0
        self.use_flags = {}
        self._set_window_use_flags()
        # Initialize coupled mode
        self.piromi_solver.set_mode('coupled')
        # Specify transformation parameters to go from HYPATE coordinates to ATVI coordinates
        self._set_domain_transformations()
        
    def _set_domain_transformations(self):
        self.panel_index = np.where(self.piromi_solver.X == self.piromi_solver.x1)[0][0] + 1
        self.x = self.piromi_solver.xpanel
        self.xHyp = self.piromi_solver.xpanel
        self.N = len(self.x)
        self.piromi_solver.ahl = None
        self.piromi_solver.setInit(self.del0, self.Me0)
            
    def initSolution(self, **kwargs):
        self.piromi_solver.ahl = analHypLoad(mach=self.Minf, P=self.Pinf, T=self.Tinf)
        yw        = self.hyp2atvi(np.zeros_like(self.xHyp))
        Tw        = self.hyp2atvi(self.Tref*np.ones_like(self.xHyp))
        self.piromi_solver.set_coupled_thermoelastic(self.x, yw, Tw, self.panel_index)
        X         = self.piromi_solver.fpi(iterations=self.fpi_iterations)
        pe, BW    = X[self.panel_index-1:,2], X[self.panel_index-1:,-1]
        pw        = pe*BW
        qw        = X[self.panel_index-1:,12]
        pstd      = self.Pinf*pw - self.Pinf
        self.Pcav = pstd
        self.p    = np.zeros_like(self.Pcav)
        self.q    = qw
        
    def timeStepping(self, **kwargs):
        yw     = self.hyp2atvi(self.w)
        Tw     = self.hyp2atvi(self.TW)
        self._check_window(yw)
        self.piromi_solver.set_coupled_thermoelastic(self.x, yw, Tw, self.panel_index)
        X      = self.piromi_solver.fpi(iterations=self.fpi_iterations)
        pe, BW = X[self.panel_index-1:,2], X[self.panel_index-1:,-1]
        pw     = pe*BW
        qw     = X[self.panel_index-1:,12]
        dyw    = X[self.panel_index-1:,4]
        puns   = self.unsteady_corrector(dyw)
        pstd   = pw + puns
        self.p = self.Pcav - (self.Pinf*pstd - self.Pinf)
        self.q = qw
        
    def _check_window(self, yw):
        yw_min, yw_max = np.min(yw), np.max(yw)
        print("\n")
        print("----------------------------------------------------------------------------")
        print("Switch Parameter: {0}, Min yw: {1}, Max yw: {2}".format(self.switch_param, yw_min, yw_max))
        print("----------------------------------------------------------------------------")
        print("\n")
        if yw_min < -self.switch_param:
            self.piromi_solver.load_high_window()
        elif yw_max > self.switch_param:
            self.piromi_solver.load_high_window()
                
    def unsteady_corrector(self, dyw):
        gm = self.atvi_solver.gamma
        ainf = self.atvi_solver.Cinf
        Mn2 = self.Minf*dyw + self.wt/ainf
        Mn1 = self.Minf*dyw 
        p1_uns = (1 + 0.5*(gm-1)*Mn1)**((2*gm)/(gm-1))
        p2_uns = (1 + 0.5*(gm-1)*Mn2)**((2*gm)/(gm-1))
        corr = (p2_uns - p1_uns)
        return corr
    
    def getCoord(self):
        x_sub = self.x[self.panel_index:]
        y = x_sub[0] + 0.500
        x_transformed = x_sub - y
        _coor = np.vstack([x_transformed, np.zeros_like(x_transformed)]).T
        return _coor # where self.x is the transformed x
        
    def match_grid(self, x, yw, Tw, wt):
        wtFit = CubicSpline(x, wt) 
        ywFit = CubicSpline(x, yw)
        TwFit = CubicSpline(x, Tw)
        _wt = wtFit(self.xHyp)
        _yw = ywFit(self.xHyp)
        _Tw = TwFit(self.xHyp)
        return _yw, _Tw, _wt
    
    def hyp2atvi(self, X):
        M = int(self.N - X.shape[0])
        X_add = X[0]*np.ones((M,))
        X = np.concatenate((X_add, X))
        return X
    
    def wt_hyp2atvi(self, X, wt):
        M = int(100-wt.shape[0])
        wt_front = np.zeros((M,))
        xfront = np.linspace(0.0, 1.0, wt_front.shape[0])
        xpanel = np.linspace(1.0, 2.0, wt.shape[0])
        wt_front_fit = lambda x: 0.0*x
        wt_panel_fit = CubicSpline(xpanel, wt)
        wt_front = wt_front_fit(self.atvi_solver.xfront)
        wt_panel = wt_panel_fit(self.atvi_solver.xpanel)[1:]
        return np.concatenate((wt_front, wt_panel))