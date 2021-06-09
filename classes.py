import random_funcs as rand
import numpy as np
import scipy.integrate as scint


G = 6.67*(10**(-11))
Msol = 2.0*(10**30)
Rsol = 700000000
Mterre = 6.0*(10**24)
Rterre = 6378137

class Planet:
    def __init__(self):
        self.real = False
        self.M = None
        self.R = None
        self.P = None
        self.proba = None
        self.e = None
        self.a = None
        self.insolation = None
        self.SNR = None


class TESS_PlanetarySystem:
    
    def __init__(self, TOI = [], orientation = np.pi/2, TESSdetectionlimit = 6, side = '<', generate = True):

        self.TOI = TOI
        self.Mstar = 0.4
        self.Rstar = 0.1
        self.orientation = None
        self.Nmin = 0
        
        if self.TOI:
            self.Nmin = self.TOI.n_planets
            self.Rstar = self.TOI.star.Rs
            self.Mstar = self.TOI.star.Ms
            
        self.N = None
        self.sigma = None
        self.orientation = orientation
        self.Planets = []
        
        self.TESSdetectionlimit = TESSdetectionlimit
        self.side = side
        self.SNR = None
        
        if self.TOI:
            S = 0
            for planet in self.TOI.Planets:
                r = planet.R
                p = planet.P
                s = planet.SNR
                S += s*np.sqrt(p)/(r**2)
            self.SNR = S / self.TOI.n_planets
            
        if generate:
            self.generate_planets()
            self.characterize_planets()
        
        
            
    def generate_planets(self):
        """
        Draws main characteristics of the system : number of planets and
        mutual inclination
        """
        self.N = self.Nmin - 1
        while self.N < self.Nmin:
            self.N = int(round(rand.random_N(), 0))
        self.sigma = rand.random_S()
        for n in range(self.N):
            self.Planets.append(Planet())
    
    def characterize_planets(self):
        self.step_1()
        self.step_2()
        return
    
    def step_1(self):
        """
        Takes real candidates into account
        Draws R, m, e for fictive planets
        """
        if self.Planets is None:
            return
        if self.TOI:
            for i in range(self.TOI.n_planets):
                self.Planets[i].R = self.TOI.Planets[i].R
                self.Planets[i].P = self.TOI.Planets[i].P
                self.Planets[i].SNR = self.TOI.Planets[i].SNR
                self.Planets[i].proba = 1
                self.Planets[i].real = True
                a = (((self.Planets[i].P*86400)**2)*G*self.Mstar*Msol/(4*np.pi**2))**(1/3)
                self.Planets[i].a = a
                self.Planets[i].e = 0
                m = rand.OtegiMass(self.Planets[i].R)
                self.Planets[i].M = m
                self.Planets[i].insolation = self.TOI.Planets[i].insolation
                
        for planet in self.Planets:
            if not planet.real:
                R = rand.random_radius()
                planet.R = R
                
                mass = rand.OtegiMass(R)
                e = rand.random_e(self.N)
                planet.M = mass
                planet.e = e
                
    
    def step_2(self):
        """
        Computes P, insolation, SNR
        """
        for planet in self.Planets:
            if not planet.real:
                P = rand.random_period(self.Planets, planet, self.Mstar)
                if P == False:
                  self.reinitialize()
                  self.caracterize_planets()
                planet.P = P
                a = (((P*86400)**2)*G*self.Mstar*Msol/(4*np.pi**2))**(1/3)
                planet.a = a
        if self.TOI:
            for planet in self.Planets:
                if not planet.real:
                    planet.insolation = self.TOI.Planets[0].insolation * (planet.P/self.TOI.Planets[0].P)**(-4/3)
                    planet.SNR = self.SNR * planet.R **2 / np.sqrt(planet.P)
        self.sort_by_period()
        


    def reinitialize(self):
        self.Planets = []
        for i in range(self.N):
            self.Planets.append(Planet())
    
    def sort_by_period(self):
        L = []
        i = 0
        for planet in self.Planets:
            j = 0
            while j < i and planet.P > L[j].P:
                j += 1
            L.insert(j, planet)
            i += 1
        self.Planets = L
            
            
    def compute_probability(self, counts, bins):
        for planet in self.Planets:
            if not planet.real:
                planet.proba = self.transit_probability(planet, counts, bins)
                
      
    def transit_probability(self, planet, counts, bins):
        """
        Computes the transit probability of a planet
        given a distribution of the global orientation of the system
        """
        a = planet.a
        e = planet.e
        theta = np.arcsin(self.Rstar*Rsol/a * 1/(1-e**2))
        
        def gauss(x, mu, sigma):
            return(1/sigma*1/np.sqrt(2*np.pi)*np.exp(-.5*((x-mu)/sigma)**2))
        
        n = len(counts)
        
        if n == 0:
            n = 1
            bins = [self.orientation - 1, self.orientation]
            counts = [1]
        L = []
        
        for j in range(n):
            i = bins[j+1]
            integ = scint.quad(lambda x: gauss(x, i, self.sigma * np.pi/180), np.pi/2-theta, np.pi/2+theta)[0]
            L.append((bins[j+1]-bins[j])*counts[j]*integ)
        
        coef = 1 if self.TOI.n_planets > 1 else 0.4
        return(sum(L) * coef)
    
    def asTable(self):
        """
        Converts the system from this class formalism into an array.
        columns = R, P, M, e, a, insolation, SNR, transit_proba , real
        """
        table = np.zeros((self.N, 9))
        for i in range(self.N):
            p = self.Planets[i]
            table[i] = [p.R, p.P, p.M, p.e, p.a, p.insolation, p.SNR, p.proba, p.real]
        return(table)
        
    def fromTable(self, table, TOI):
        """
        Converts table (array of planetary characteristics) into a planetary
        system in this class formalism
        """
        self.N = len(table)
        self.Planets = []
        self.TOI = TOI
        for planet in table:
            pl = Planet()
            pl.R = planet[0]
            pl.P = planet[1]
            pl.M = planet[2]
            pl.e = planet[3]
            pl.a = planet[4]
            pl.insolation = planet[5]
            pl.SNR = planet[6]
            pl.proba = planet[7]
            pl.real = bool(planet[8])
            self.Planets.append(pl)
        return