# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:21:24 2016

@author: florencemontpetit
"""

import numpy as np
import matplotlib.pyplot as plt

# fonctions mérite
def diffrac5d(x):
    
    return (np.sin(x[0,:])/x[0,:]) * (np.sin(x[1,:])/x[1,:]) * (np.sin(x[2,:])/x[2,:]) * (np.sin(x[3,:])/x[3,:]) * (np.sin(x[4,:])/x[4,:])

def diffrac5dtop(x):
    
    return (np.sin(x[0])/x[0]) * (np.sin(x[1])/x[1]) * (np.sin(x[2])/x[2]) * (np.sin(x[3])/x[3]) * (np.sin(x[4])/x[4])

G = 10000
N = 10
pc = 0.7
x=np.zeros([5,N])
xn=np.zeros([5,N])
xtop=np.zeros([5,G])
gg = np.arange(0,G,1)
epsmin = np.zeros(15)
lastn = np.zeros(15)
L=5.*np.pi      # etendue du domaine en [x,y]

for a in range(0,15):   # boucle pour les moyennes de simulation
    pm = 0.1
    s = 0.01
    eps=np.zeros(G)
    diff=np.zeros(G)
    
    for i in range(0,N):    # initialisation des solutions initiales
        for j in range(0,5):
            x[j,i]=np.random.uniform(-L,L)
                 
    for n in range(0,G):   
        f =diffrac5d(x)     # classement des solutions avec fct mérite
        rang = np.argsort(f)
        if n > 0 :      
            for i in range(0,5):
                x[i,rang[0]]=xtop[i,n-1]  # meilleur parent remplace pire rejeton                    
            f[rang[0]] = ftop    # rang du parent remplace rejeton
            rang = np.argsort(f)  # nouveau classement avec bon parent
        for i in range(0,5):    
            xtop[i,n]=np.copy(x[i,rang[-1]]) # sauvegarde du meilleur futur parent 
        ftop = f[rang[-1]]      # fonction mérite du meilleur parent
        if n > 200 and pm < 1:       
            if eps[n-1]-eps[n-200] == 0:    
                pm = pm + 0.1       # augmentation de prob. mutation
            else:
                pm= pm
        if pm > 1:
            pm = 1            
        if pm == 1:
            if eps[n-1]-eps[n-200] == 0:    
                s = s + 0.01        # augmentation de la variance
            else:
                s = s
        if s > 10:            # convergence arrêtée, fin simulation
            break
                    
        for j in range(0,N,2):       # création des rejetons    
            i1=rang[np.random.random_integers(0.8*N,N)-1] # dans les premiers 20%
            i2=i1
            while (i2==i1):
                i2= np.random.random_integers(0,N-1) # n’importe qui sauf i1                    
            if np.random.uniform() < pc:
                for i in range(0,5):  
                    r = np.random.uniform()     
                    xn[i,j]= r*x[i,i1] + (1-r)*x[i,i2]    # reproduction             
                    xn[i,j+1]= r*x[i,i2] + (1-r)*x[i,i1]
            else:
                 for i in range(0,5): 
                    xn[i,j]= x[i,i1]        # pas de reproduction    
                    xn[i,j+1]= x[i,i2] 
            for i in range(0,5):
                if np.random.uniform() < pm:
                    g = np.random.normal(0,s)
                    xn[i,j]= xn[i,j] + g      # mutation
                if np.random.uniform() < pm:
                    g = np.random.normal(0,s)       
                    xn[i,j+1]= xn[i,j+1] + g
                                
        for i in range(0,5):       
            x[i,:]=xn[i,:]          # remplacement generationnel
                              
        eps[n]= np.abs(1-diffrac5dtop(xtop[:,n]))   # erreur sur la solution
        
    epsmin[a] = np.abs(1-diffrac5dtop(xtop[:,n]))
    lastn[a] = n       
    
somme = 0
for i in range(0,len(epsmin)):
    somme += epsmin[i]
    
moy = somme/len(epsmin)
somm=0
for i in range(0,len(epsmin)):
    somm += np.abs(epsmin[i]-moy)
    
sigma = somm/len(epsmin)

print('moyeps=',moy)
print('sigmepsa=',sigma)

somme = 0
for i in range(0,len(epsmin)):
    somme += lastn[i]
    
moy = somme/len(epsmin)
somm=0
for i in range(0,len(epsmin)):
    somm += np.abs(lastn[i]-moy)
    
sigma = somm/len(epsmin)

print('moyn=',moy)
print('sigman=',sigma)
 

#plt.rcParams.update({'font.size': 16})
#plt.xlabel('Génération')
#plt.ylabel('Erreur')
#plt.semilogy(gg,eps,marker='.')
            
            
        
        