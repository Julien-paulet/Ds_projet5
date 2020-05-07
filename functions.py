#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def isnan(data, aff_glob=True, aff_col=False):
    taille = len(data)
    nulle = []
    nbr_moy = []
    for i in data.columns:
        if any(pd.isna(data[i])):
            nbr = len(data[i])-len(data[i].dropna())
            if aff_glob == True:
                print("La colonne '{}'".format(i), "comporte", nbr, "/ {}".format(taille), "valeur(s) manquante(s)")
                print("Ce qui représente {}%".format(nbr/taille*100))
            if aff_col == True:
                print("La colonne '{}' comporte des valeurs manquantes".format(i))
            col = data[i]
            nulle.append(col)
            nbr_moy.append(nbr)
            
    nbr_moy = pd.DataFrame(nbr_moy, columns=['Nbr']).mean()
    print(len(nulle), "colonnes au total possèdent des valeurs manquantes")
    print(nbr_moy[0], "/ {}".format(taille), "valeurs sont manquantes en moyenne dans chaque colonne") 

