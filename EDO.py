# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:53:05 2016

@author: florencemontpetit
"""
import matplotlib.pyplot as plt
import numpy as np



# FONCTION CALCULANT LE COTE DROIT DU SYSTEME DE DEUX EDO
def G(t0,u,I):
    gk = 36 ; gna = 120 ; gl = 12.9
    vk = -12; vna = 115 ; vl = 10.6
    an = (0.1-0.01*u[0])/(np.exp(1-0.1*u[0])-1)   #alphas
    am = (2.5-0.1*u[0])/(np.exp(2.5-0.1*u[0])-1)   
    ah = 0.07*np.exp(-u[0]/20)
    bn = 0.125*np.exp(-u[0]/80)                  #betas
    bm = 4*np.exp(-u[0]/18)
    bh = 1/(np.exp(3-0.1*u[0])+1)
   
    gV = I-gk*(u[1]**4)*(u[0]-vk)-gna*(u[2]**3)*u[3]*(u[0]-vna)-gl*(u[0]-vl)  #eqs diff
    gN = an*(1-u[1])-bn*u[1]
    gM = am*(1-u[2])-bm*u[2]
    gH = ah*(1-u[3])-bh*u[3]
    
    eval=np.array([gV,gN,gM,gH]) # vecteur RHS (pentes)
    return eval
# END FONCTION G
# FONCTION CALCULANT UN SEUL PAS DE RUNGE-KUTTA D’ORDRE 4
def rk(h,t0,uu,I):
    g1=G(t0,uu,I) # Eq (1.15)
    g2=G(t0+h/2.,uu+h*g1/2.,I) # Eq (1.16)
    g3=G(t0+h/2.,uu+h*g2/2.,I) # Eq (1.17)
    g4=G(t0+h,uu+h*g3,I) # Eq (1.18)
    unew=uu+h/6.*(g1+2.*g2+2.*g3+g4) # Eq (1.19)
    return unew
# END FONCTION RK
# OSCILLATIONS NONLINEAIRES: 2 EDOS NONLINEAIRES COUPLEES
nMax=2000 # nombre maximal de pas de temps
eps =1.e-5 # tolerance
tfin=100. # duree d’integration
t=np.zeros(nMax) # tableau temps
u=np.zeros([nMax,4]) # tableau solution
i=np.zeros(nMax) # tableau courant appliqué

u[0,0]= 0 # condition initiale sur V

an0 = (0.1-0.01*u[0,0])/(np.exp(1-0.1*u[0,0])-1)   #alphas initiaux
am0 = (2.5-0.1*u[0,0])/(np.exp(2.5-0.1*u[0,0])-1)   
ah0 = 0.07*np.exp(-u[0,0]/20)
bn0 = 0.125*np.exp(-u[0,0]/80)                  #betas initiaux
bm0 = 4*np.exp(-u[0,0]/18)
bh0 = 1/(np.exp(3-0.1*u[0,0])+1)
neq = an0/(an0+bn0) ; meq = am0/(am0+bm0) ; heq = ah0/(ah0+bh0)   # valeurs initiales

u[0,1]=neq ; u[0,2]=meq ; u[0,3]=heq


nn=0 # compteur iterations temporelles
h=0.01 # pas initial

while (t[nn] < tfin) and (nn < nMax): # boucle temporelle
    if t[nn] > 1:  
            i[nn]=0     # on arrête le pulse à un temps x
    else:
        i[nn]=35
    u1 =rk(h, t[nn],u[nn,:],i[nn]) # pas pleine longueur
    u2a=rk(h/2.,t[nn],u[nn,:],i[nn]) # premier demi-pas
    u2 =rk(h/2.,t[nn],u2a[:],i[nn]) # second demi-pas
    delta=max(abs(u2[0]-u1[0]),abs(u2[1]-u1[1]),abs(u2[2]-u1[2]),abs(u2[3]-u1[3])) # Eq (1.42)
    if delta > eps: # on rejette
        h/=1.5 # reduction du pas
    else: # on accepte le pas
        nn=nn+1 # compteur des pas de temps
        t[nn]=t[nn-1]+h # le nouveau pas de temps
        if t[nn] > 1:  
            i[nn]=0     # on arrête le pulse à un temps x
        u[nn,:]=u2[:] # la solution a ce pas
        if delta <= eps/2.: h*=1.5 # on augmente le pas
    #print("{0}, t {1}, V {2}, n {3}, m {4}, h {5}.".format(nn,t[nn],u[nn,0],u[nn,1],u[nn,2],u[nn,3]))
    #print("pulse {0}".format(i[nn]))
# fin boucle temporelle
# END
    
    
plt.rcParams.update({'font.size': 20})
plt.plot(t[0:nn],u[0:nn,0],'r')
plt.xlabel('t [ms]')
plt.ylabel('V [mV]')
plt.show()
plt.close()



plt.show()
plt.close()