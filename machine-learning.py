# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:26:15 2016

@author: florencemontpetit
"""


#======================================================================
import numpy as np
import matplotlib.pyplot as plt

ni,nh,nh2,no =13,8,5,1 # nombre d’unites d’entree, interne, interne2 et de sortie
wih =np.zeros([ni,nh]) # poids des connexions entree vers interne
who =np.zeros([nh,nh2]) # poids des connexions interne vers interne 2
who2 =np.zeros([nh2,no]) # poids des connexions interne 2 vers sortie
ivec =np.zeros(ni) # signal en entree
sh =np.zeros(nh) # signal des neurones interne
sh2 =np.zeros(nh2) # signal des neurones interne 2
so =np.zeros(no) # signal des neurones de sortie
err =np.zeros(no) # signal d’erreur des neurones de sortie
deltao=np.zeros(no) # gradient d’erreur des neurones de sortie
deltah=np.zeros(nh) # gradient d’erreur des neurones internes
deltah2=np.zeros(nh2)  # gradient d’erreur des neurones internes2
eta =0.1 # parametre d’apprentissage
#----------------------------------------------------------------------
# Fonction d’activation sigmoidale
def actv(a):
    return 1./(1.+np.exp(-a)) # Eq. (6.5)
#----------------------------------------------------------------------
# Derivee de la fonction d’activation sigmoidale
def dactv(s):
    return s*(1.-s) # Eq. (6.19)
#----------------------------------------------------------------------
# fonction reseau feed-forward, calcul signal de sortie
def ffnn(ivec):
    for ih in range(0,nh): # couche d’entree a couche interne
        sum=0.
        for ii in range(0,ni):
            sum+=wih[ii,ih]*ivec[ii] # Eq. (6.1)
        sh[ih]=actv(sum) # Eq. (6.2)
        
    for ih2 in range(0,nh2): # couche interne a couche interne2
        sum=0.
        for ih in range(0,nh):
            sum+=who[ih,ih2]*sh[ih] # Eq. (6.3) avec b_1=0
        sh2[ih2]=actv(sum) # Eq. (6.4)
        
    for io in range(0,no): # couche interne 2 a couche sortie
        sum=0.
        for ih2 in range(0,nh2):
            sum+=who2[ih2,io]*sh2[ih2] # Eq. (6.3) avec b_1=0
        so[io]=actv(sum) # Eq. (6.4)
    return
# END fonction ffnn
#----------------------------------------------------------------------    
# retropropagation du signal d’erreur et ajustement des poids du reseau
def backprop(err):
    for io in range(0,no): # couche de sortie a couche interne2
        deltao[io]=err[io]* dactv(so[io]) # Eq. (6.20)
        for ih2 in range(0,nh2):
            who2[ih2,io]+=eta*deltao[io]*sh2[ih2] # Eq. (6.17) pour les wHO

    for ih2 in range(0,nh2): # couche interne 2 a couche de sortie
        sum=0.
        for io in range(0,no):
            sum+=deltao[io]*who2[ih2,io]
        deltah2[ih2]=dactv(sh2[ih2])*sum # Eq. (6.21)
        for ih in range(0,nh):
            who[ih,ih2]+=eta*deltah2[ih2]*sh[ih] # Eq. (6.17) pour les wIH
            
    for ih in range(0,nh): # couche interne a couche interne 2
        sum=0.
        for ih2 in range(0,nh2):
            sum+=deltah2[ih2]*who[ih,ih2]
        deltah[ih]=dactv(sh[ih])*sum # Eq. (6.21)
        for ii in range(0,ni):
            wih[ii,ih]+=eta*deltah[ih]*ivec[ii] # Eq. (6.17) pour les wIH
    return
# END fonction backprop    
#----------------------------------------------------------------------
# fonction de melange
def randomize(n):
    dumvec=np.zeros(n)
    for k in range(0,n):
        dumvec[k]=np.random.uniform() # tableau de nombre aleatoires
    return np.argsort(dumvec) # retourne le tableau de rang
# END
#======================================================================
# MAIN: Entrainement d’un reseau par retropropagation
data = np.loadtxt('training.txt')    #données fournies
exam = np.loadtxt('exam.txt')
nset =800# nombre de membres dans ensemble d’entrainement
ntest = 800 # nb membres dans ensemble de test
niter =5000 # nombre d’iterations d’entrainement
oset =np.zeros([nset,no]) # sortie pour l’ensemble d’entrainement
tset =np.zeros([nset,ni]) # vecteurs-entree l’ensemble d’entrainement
otest =np.zeros([ntest,no]) # sortie pour l’ensemble test
ttest =np.zeros([ntest,ni])  # vecteurs entrée pour ensemble test
maxx = np.zeros(ni)
minn = np.zeros(ni)
rmserr=np.zeros(niter) # erreur rms d’entrainement
n = np.arange(0,niter,1)
nn = np.arange(0,niter,100) 
sortie = np.zeros(nset)
rmserrtest = np.zeros(int(niter/100))
reponse = np.zeros([1000,niter])

# lecture/initialisation de l’ensemble d’entrainement
for i in range(0,nset):
    tset[i,:] = data[i,:-1]   
    oset[i] = data[i,13]
    if oset[i] == 1:            # évite l'explosion
        oset[i] = 0.9
    else :
        oset[i] = 0.1

for i in range(nset,ntest+nset):   # même chose pour ensemble test
    ttest[i-nset,:] = data[i,:-1]
    otest[i-nset] = data[i,13]
    if otest[i-nset] == 1:
        otest[i-nset] = 0.9
    else :
        otest[i-nset] = 0.1
                
for j in range(0,ni):           # normalisation de tous les ensembles
    maxx[j] = np.max(ttest[:,j])
    minn[j] = np.min(ttest[:,j])        
    ttest[:,j]= (ttest[:,j]-minn[j])/(maxx[j]-minn[j])     
for j in range(0,ni):
    maxx[j] = np.max(exam[:,j])
    minn[j] = np.min(exam[:,j])        
    exam[:,j]= (exam[:,j]-minn[j])/(maxx[j]-minn[j])    
for j in range(0,ni):
    maxx[j] = np.max(tset[:,j])
    minn[j] = np.min(tset[:,j])        
    tset[:,j]= (tset[:,j]-minn[j])/(maxx[j]-minn[j])
        
# initialisation aleatoire des poids
for ii in range(0,ni): # poids entree-interne
    for ih in range(0,nh):
        wih[ii,ih]=np.random.uniform(-0.5,0.5)
for ih in range(0,nh): # poids interne-interne2
    for ih2 in range(0,nh2):
        who[ih,ih2]=np.random.uniform(-0.5,0.5)
for ih2 in range(0,nh2): # poids interne2-sortie
    for io in range(0,no):
        who2[ih2,io]=np.random.uniform(-0.5,0.5)
nt = 0
for iter in range(0,niter): # boucle sur les iteration d’entrainement
    sum=0.   
    rvec=randomize(nset) # melange des membres
    for itrain in range(0,nset): # boucle sur l’ensemble d’entrainement
        itt=rvec[itrain] # le membre choisi
        ivec=tset[itt,:]
        ffnn(tset[itt,:]) # calcule signal de sortie    
        for io in range(0,no): # signaux d’erreur sur neurones de sortie
            err[io]=oset[itt,io]-so[io]
            sum+=err[io]**2 # cumul pour calcul de l’erreur rms
        backprop(err) # retropropagation
# END boucle sur ensemble        
    rmserr[iter]=np.sqrt(sum/nset/no) # erreur rms a cette iteration
        
    if iter%int(niter/100) == 0:   # test à cette itération
        sum=0.
        rvec=randomize(ntest) # melange des membres    
        for itrain in range(0,ntest): # boucle sur l’ensemble de test
            itt=rvec[itrain] # le membre choisi
            ivec=ttest[itt,:]
            ffnn(ttest[itt,:]) # calcule signal de sortie
            if so[io] < 0.5:  # calcul erreur  classification
                St = 0.1
            else:
                St = 0.9
            for io in range(0,no): # signaux d’erreur sur neurones de sortie            
                err[io]=np.abs(St-otest[itt,io])
                sum+=err[io]# cumul pour calcul de l’erreur test
        if nt < 50 :
            rmserrtest[nt]=1/ntest * sum # erreur test à cette itération
            nt += 1
            
     
    for itrain in range(0,1000): # boucle sur l’ensemble examen
        itt=itrain # le membre choisi
        ivec=exam[itt,:]
        ffnn(exam[itt,:]) # calcule signal de sortie 
        if so[io] < 0.5:
            S = 0
        else:
            S = 1        
        reponse[itrain,iter]=S     # boson ou pas !
            
            
            
        
# Maintenant la phase de test irait ci-dessous...
# END MAIN

plt.semilogx(n,rmserr)
plt.semilogx(nn,rmserrtest,'o')
plt.rcParams.update({'font.size': 16})
plt.xlabel('Itérations entraînement')
plt.ylabel('Erreur')

plt.show()