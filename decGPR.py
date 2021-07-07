import sys
sys.path.append("/home/cvv5110/Desktop/APUS/fiml/")
sys.path.append('/home/cvv5110/Desktop/APUS/fiml/pyaeroutils/')
sys.path.append("/home/cvv5110/Packages/jenny/")
import numpy as np
import pickle
from RANS.getConditions import getConditions
from scipy.interpolate import interp1d, CubicSpline, InterpolatedUnivariateSpline
from Solvers.solvers import solver
from pyaeroutils.supersonicFlow_hyp import analHypLoad
import os
import matplotlib.pyplot as plt
import pickle
from python.jnPostproc import postproc
from solverGPR import SolverGPR

class decoupledSolve():
    """ Plot, extract and save ATE solutions at specific time stamps to solve steady solutions on them """
    
    def __init__(self, **kwargs):
        
       for _k, _v in kwargs.items():
           setattr(self, _k, _v) 
           
       self.solver = SolverGPR(Minf=self.Minf, Pinf=self.Pinf, Tinf=self.Tinf, del0=self.del0, Me0=self.Me0, level=-1, Tini=Tw)
       self.colors = ['k', 'grey', 'brown', 'orange', 'blue', 'lawngreen', 'red', 'purple', 'y', 'cyan']
       self.cfd_dir = "/home/cvv5110/Packages/simg/test/hypate/ate_res/"
       self.cfd = postproc(fileName=self.cfd_dir+"ate2d_l3_F.hdf")
       self.cfd.probeAt(xlim=[-0.5,0.5], ylim=0.01)
       self.cfd_data = self.cfd.prbData
       self.cfd_disps = self.cfd_data[:,2,:]
       self.cfd_temps = self.cfd_data[:,5,:]
       self.cfd_velos = self.cfd_data[:,3,:]
       self.cfd_coor = self.cfd.coor[0:49,0]+1.50
       print()
       
    def solve_decoupled(self, times):
        self.solver.initSolution()
       # Set common cavity pressure
        cfd_pcav = -self.cfd_data[:,0,0]
        self.solver.setPcav(self.cfd_coor, cfd_pcav)
        self.press, self.heat, self.betas = [],[],[[],[],[]]
        for time in times:
            disp = self.cfd_disps[:,time]
            temp = self.cfd_temps[:,time]
            velo = self.cfd_velos[:,time]
            yw, Tw, wt = self.solver.match_grid(self.cfd_coor, disp, temp, velo)
            p, q, BH, BC, BW = self.solver.decoupled_solve(yw, Tw, wt)
            self.press.append(p); self.heat.append(q); self.betas[0].append(BH); self.betas[1].append(BC); self.betas[2].append(BW);
        
    def extract(self, time):
       # Create save name
        self.name = 'M{0}{1}{2}-{3}'.format(self.Minf, self.BC1, self.BC2, int(time))
        if not os.path.exists(self.dumpdir+self.name+'/'):
            os.mkdir(self.dumpdir+self.name+'/')
        disp = self.cfd_disps[:,time]
        temp = self.cfd_temps[:,time]
        if disp[0] > 1e-8 or disp[-1] > 1e-8:
            print("Leading or trailing edge displacements are not zero")
            disp = disp - disp[0]
        ywCoefs, ywFit, dywFit = self.get_equation(disp)
        TwCoefs, TwFit, dTwFit = self.get_equation(temp)
        self.save_coefficients(ywCoefs, TwCoefs)
        self.test_fit(ywFit, disp)
        self.test_fit(TwFit, temp)
        self.saveConditions()
        self.save_thermoelastic(disp, temp, ywFit, TwFit)
        self.setDatabase(ywFit, TwFit)
        
    def plot_thermoelastic(self, times):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        for _i in range(len(times)):
            time = times[_i]
            disp = self.cfd_disps[:,time]
            temp = self.cfd_temps[:,time]
            ax1.plot(self.cfd_coor, disp, color=self.colors[_i], label='t={0}'.format(time))
            ax2.plot(self.cfd_coor, temp, color=self.colors[_i], label='t={0}'.format(time))
        ax1.set_xlabel('x, meters', size=13)
        ax2.set_xlabel('x, meters', size=13)
        ax1.set_ylabel('Displacement', size=13)
        ax2.set_ylabel('Temperature', size=13)
        ax1.legend()
        ax2.legend()
        fig.set_size_inches(10,5)
        plt.show()
        
    def get_equation(self, value):
        degree = 9
        coefs, res, _, _, _ = np.polyfit(self.cfd_coor-1.50, value, degree, full=True)
        # [coefs[0]*x**9, coefs[1]*x**8, coefs[2]*x**7, coefs[3]*x**6, coefs[4]*x**5, coefs[5]*x**4, coefs[6]*x**3, coefs[7]*x**2, coefs[8]*x**1, coefs[9]]
        fit = np.poly1d(coefs)
        dcoefs = np.array([9*coefs[0], 8*coefs[1], 7*coefs[2], 6*coefs[3], 5*coefs[4], 4*coefs[5], 3*coefs[6], 2*coefs[7], 1*coefs[8]])
        dfit = np.poly1d(dcoefs)
        return coefs, fit, dfit
    
    def save_coefficients(self, ywCoefs, TwCoefs):
        np.savetxt(self.dumpdir+self.name+'/ywCoefs.out', ywCoefs)
        np.savetxt(self.dumpdir+self.name+'/TwCoefs.out', TwCoefs)
        
    def save_thermoelastic(self, yw, Tw, ywFit, TwFit):
        pickle.dump(ywFit, open(self.dumpdir+self.name+'/ywFit.p', 'wb'))
        pickle.dump(TwFit, open(self.dumpdir+self.name+'/TwFit.p', 'wb'))
        data = np.hstack((self.cfd_coor.reshape(-1,1), yw.reshape(-1,1), Tw.reshape(-1,1)))
        np.savetxt(self.dumpdir+self.name+'/thermoelastic.out', data)
        
    def saveConditions(self):
        if not os.path.exists(self.dumpdir+self.name+'/conditions.txt'):
            f = open(self.dumpdir+self.name+'/conditions.txt', 'w')
            f.write('Minf = '+str(self.Minf)+'\n')
            f.write('Pinf = '+str(self.Pinf)+'\n')
            f.write('Tinf = '+str(self.Tinf)+'\n')
            f.write('del0 = '+str(self.del0)+'\n')
            f.write('Me0  = '+str(self.Me0)+'\n')
            f.close()
    
    def setDatabase(self, ywFit, TwFit):
        if self.name not in self.ATEdatabase:
            self.ATEdatabase[self.name] = {}
            self.ATEdatabase[self.name]['Minf']  = self.Minf
            self.ATEdatabase[self.name]['Pinf']  = self.Pinf
            self.ATEdatabase[self.name]['Tinf']  = self.Tinf
            self.ATEdatabase[self.name]['def']   = ywFit
            self.ATEdatabase[self.name]['temp']  = TwFit
        with open('/home/cvv5110/Packages/interface/ATEdatabase.p', 'wb') as f:
            pickle.dump(self.ATEdatabase, f)
        f.close()
            
    def test_fit(self, fit, data):
        X = np.linspace(self.cfd_coor[0], self.cfd_coor[-1], data.shape[0])
        fitted = fit(X)
        error = np.linalg.norm(fitted - data)
        if error > 1e-7:
            print("Fitted error above threshold")

    def plotBetas(self, times):
        labels = ['$\\beta_H$', '$\\beta_C$', '\\beta_W']
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        for _i in range(len(times)):
            ax1.plot(self.solver.atvi_solver.X, self.betas[0][_i], color=self.colors[_i], label='t={0}'.format(format(times[_i]*0.001, '0.1f')))
            ax2.plot(self.solver.atvi_solver.X, self.betas[1][_i], color=self.colors[_i], label='t={0}'.format(format(times[_i]*0.001, '0.1f')))
            ax3.plot(self.solver.atvi_solver.X, self.betas[2][_i], color=self.colors[_i], label='t={0}'.format(format(times[_i]*0.001, '0.1f')))
        plt.legend()
        ax1.set_title('$\\beta_H$')
        ax2.set_title('$\\beta_c$')
        ax3.set_title('$\\beta_W$')
        plt.tight_layout()
        fig.set_size_inches(10,5)
        
    def plotPressures(self, times):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for _i in range(len(times)):
            time = times[_i]
            p_cfd = -self.cfd_data[:,0,time]
            ax.plot(self.cfd_coor, p_cfd, linestyle='-', color=self.colors[_i], label='t={0}'.format(format(times[_i]*0.001, '0.1f')))
            ax.plot(self.solver.atvi_solver.X, -self.press[_i,:], linestyle='--', color=self.colors[_i], label='t={0}'.format(format(times[_i]*0.001, '0.1f')))
        ax.set_ylabel('Pressure, Pa', size=14)
        ax.set_xlabel('x, meters', size=14)        
            
    def plotHeat(self, times):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for _i in range(len(times)):
            time = times[_i]
            ax.plot(self.coor, self.cfd_data[:,1,time], linestyle='-', color=self.colors[_i], label='t={0}'.format(format(times[_i]*0.001, '0.1f')))
            ax.plot(self.solver.atvi_solver.X, self.heat[_i,:], linestyle='--', color=self.colors[_i], label='t={0}'.format(format(times[_i]*0.001, '0.1f')))
        ax.set_ylabel('Heat flux, $W/m^2$', size=14)
        ax.set_xlabel('x, meters', size=14)
        
def open_database():
    if os.path.exists('./ATEdatabase.p'):
        database = pickle.load(open('/home/cvv5110/Packages/interface/ATEdatabase.p', 'rb'))
    else:
        database = {'init': 0}
        pickle.dump(database, open('/home/cvv5110/Packages/interface/ATEdatabase.p', 'wb'))
    return database
            
dumpdir = '/home/cvv5110/Desktop/APUS/HYPATE DATA/ate/'
    
database = open_database()
Minf = 7.523
Pinf = 3759.678
Tinf = 466.2
del0 = 0.0009095675977560256
Me0  = 7.228500244963227
Tw   = 380.0
#times = [100, 200, 300, 400, 500, 600, 700, 800, 900]
times = [50, 100, 150, 200, 250, 300, 350]

decSolver = decoupledSolve(Minf=Minf, Pinf=Pinf, Tinf=Tinf, del0=del0, Me0=Me0, Tw=Tw, BC1='CC', BC2='SS', ATEdatabase=database, dumpdir=dumpdir)
#decSolver.solve_decoupled(times)
decSolver.extract(0)
decSolver.plotBetas(times)
decSolver.plot_thermoelastic(times)
plt.show()
    
print()
