import random as rd
import numpy as np
import scipy.integrate as scint
import operator

G = 6.67*(10**(-11))
Msol = 2.0*(10**30)
Rsol = 700000000
Mterre = 6.0*(10**24)
Rterre = 6378137

def random_N():
    """
    Number of planets in the system
    Ballard & Johnson 2016
    """
    x = rd.uniform(0, 1.2)
    if x <= 0.2:
        N = rd.gauss(5.18, 0.56)
    else:
        N = rd.gauss(7.3, 0.85)
    return(N)
    
def random_S():
    """
    Mutual inclination of the system
    Ballard & Johnson 2016
    """
    S = -1
    while S < 0:
        S = rd.gauss(1.91, 1)
    return(S)


def Mass(R):
    """
    Planetary mass drawn from the radius
    Rocky : Zeng & Jacobsen 2017
    Gaseous : Wolfgang et al. 2016
    """
    if R < 1.5:
        return(R**4)
    else:
        a, b, c = 0.0975, 0.4938, 0.7932
        MpureFe = np.exp((-b+np.sqrt(b**2-4*a*(c-R)))/(2*a))
        C, gm, sg, bt = 2.6, 1.3, 2.1, 1.5
        M = -1
        while not (0 < M < MpureFe):
            M = rd.gauss(C*(R**gm), np.sqrt(sg**2+bt*R))
        return(M)

def OtegiMass(R):
    """
    Planetary mass drawn from the radius
    Otegi 2020
    """
    x = rd.uniform(0, 1)
    if R < 2:
        return(0.9 * R**3.45)
    elif R > 3:
        return(1.74 * R**1.58)
    elif x < 0.5:
        return(0.9 * R**3.45)
    else:
        return(1.74 * R**1.58)
        


f8 = lambda x: -79.86*(x**2)+12.809*x-0.0026
f56 = lambda x: -23.54*(x**2)+6.8885*x-0.0003
f4 = lambda x: -14.371*(x**2)+5.4279*x-0.0027
f3 = lambda x: -9.8023*(x**2)+4.449*x-0.0009
f2 = lambda x: -4.7075*(x**2)+3.0739*x+0.0005
f1 = lambda x: -4.5084*(x**2)+2.7578*x+0.0093
functions = [f1, f2, f3, f4, f56, f56, f8, f8]
limits = [0.615, 0.653, 0.453, 0.377, 0.2925, 0.2925, 0.16, 0.16]


def random_e(N):
    """
    Orbit eccentricity. Depends on the total number of planets
    in the system.
    Limbach & Turner 2014
    """
    key = False
    if N > 8:
      N = 8
    f = functions[N-1]
    limit = limits[N-1]
    
    C = scint.quad(f, 0, limit)[0]
    while not key:
        x = rd.uniform(0, limit)
        u = rd.uniform(0, C)
        if u <= f(x):
            key = True
    return(x)


grid_DC_2015 = [
    0.4, 1.5, 4.4, 5.5, 10, 12, 11, 0, 0, 0,
    0.46, 1.4, 3.5, 5.7, 10, 13, 16, 6.4, 10, 19,
    0.061, 0.27, 1.2, 2.5, 6.7, 13, 14, 12, 8.3, 10,
    0.002, 0.009, 0.42, 1.8, 6.4, 9.3, 10, 12, 09.6, 4.5,
    0, 0.004, 0.23, 0.96, 2.7, 3.8, 4.6, 5.8, 4.2, 1.1,
    0, 0.006, 0.17, 0.42, 1.1, 1.4, 0.81, 1.6, 1.7, 0.16,
    0, 0.008, 0.18, 0.18, 0.36, 0.51, 0.32, 0.21, 0.42, 0.08
    ]
renorm = 285.35

radii = [0.5, 1., 1.5, 2., 2.5, 3., 3.5]
logx0 = np.log10(0.5)
logx1 = np.log10(200)
logx = np.r_[logx0:logx1:11j]
periods = 10**logx
periods = periods[:10]

DC_array = np.array([
        grid_DC_2015[:10],
        grid_DC_2015[10:20],
        grid_DC_2015[20:30],
        grid_DC_2015[30:40],
        grid_DC_2015[40:50],
        grid_DC_2015[50:60],
        grid_DC_2015[60:]])

    
            
def random_radius():
    """
    Radius drawn from the occurrence rates in the period-radius grid
    Dressing & Charbonneau 2015
    """
    key = False
    while not key:
        R = rd.uniform(0.5, 4)
        x = rd.uniform(0., renorm)
        ri = np.searchsorted(radii, R)
        ri -= 1
        if sum(DC_array[ri]) >= x:
            key = True    
    return(R)

def random_period(Planets, planet, Mstar):
    """
    Period drawn in the D&C grid, restricted to regions authorized by
    the Hill criterion
    Dressing & Charbonneau 2015
    """
    forbidden_intervals = []
    M = planet.M
    exc = planet.e
    
    for pla in Planets:
        if pla.P != None:
            a = pla.a
            e = pla.e
            m = pla.M
            c=np.sqrt(3)*((M+m)*Mterre/(3*Mstar*Msol))**(1/3)
            borne_sup = a*(1+e)*(1+c)/(1-c)/(1-exc)
            borne_inf = a*(1-e)*(1-c)/(1+c)/(1+exc)
            borne_sup = 2*np.pi*np.sqrt(borne_sup**3/(G*(Mstar*Msol))) / 86400
            borne_inf = 2*np.pi*np.sqrt(borne_inf**3/(G*(Mstar*Msol))) / 86400
            borne_inf = max(borne_inf, 0.5)
            borne_sup = min(borne_sup, 200)
            forbidden_intervals.append([borne_inf, borne_sup])
            
    F = np.array(sorted(forbidden_intervals, key = operator.itemgetter(0)))
    return(random_interval(0.5, 200, F, planet.R))
            


def random_interval(a, b, L, r):
    length = 0
    Lengths = []
    I = []
    i = a
    n = len(L)
    for w in range(n):
        x, y = L[w][0], L[w][1]
        if x < i:
            x = i
        j = x
        length += (j-i)
        I.append((i, j))
        Lengths.append(j-i)
        i = y
    length+= (b-i)
    Lengths.append(b-i)
    I.append((i, b))
    ri = np.searchsorted(radii, r)
    ri -= 1
    renorm2 = sum(DC_array[ri])
    
    probas = []
    for (a, b) in I:
        
        a1, b1 = np.searchsorted(periods, a), np.searchsorted(periods, b)
        a1 -= 1
        proba = 0
        
        if a1 == b1-1:
            if a1 == 9:  
                probas.append(DC_array[ri, a1]*(b-a)/(200-periods[a1]))
            else:
                probas.append(DC_array[ri, a1]*(b-a)/(periods[b1]-periods[a1]))
        else:
            for i in range(a1, b1):
                
                if i == a1:
                    if a1 == 9:
                        proba += DC_array[ri, i] * (b-a)/(b-periods[i])
                    else:
                        proba += DC_array[ri, i] * (periods[i+1]-a)/(periods[i+1]-periods[i])
                elif i == b1-1:
                    if b1 == 10:
                        proba += DC_array[ri, i] * (b-periods[i])/(200-periods[i])
                    else:
                        proba += DC_array[ri, i] * (b-periods[i])/(periods[i+1]-periods[i])
                    
                
                
                else:
                    proba += DC_array[ri, i]
            probas.append(proba)
            
    pb2 = np.cumsum(probas)
    key = False
    essais2 = 0
    while not key:
        x = rd.uniform(0, sum(probas))
        index = np.searchsorted(pb2, x)
        try:
            u, v = I[index]
        except:
            print("sum(probas) = " +str(sum(probas)))
            print("I = " + str(I))
            print("Index = " + str(index))
        key2 = False
        essais2 += 1
        if essais2 > 1000:
          return(False)
        
        essais = 0
        while not key2:
            P = rd.uniform(u, v)
            x = rd.uniform(0, renorm2)
            pi = np.searchsorted(periods, P)
            pi -= 1
            if DC_array[ri, pi] >= x:
                key2 = True
                key = True
            essais += 1
            if essais > 100:
                key2 = True
    return(P)