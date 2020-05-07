#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from math import pi
from sklearn import preprocessing, decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
import warnings


# # Classe préparation des données

# In[2]:


class prepData():
    def centrageReduction(data):
        X = data.values
        std_scale = preprocessing.StandardScaler().fit(X)
        X_scaled = std_scale.transform(X)
        return X_scaled
        
    def acp():
        pass


# # Classe Clustering

# In[8]:


class clustering():
    
    def elbowMethod(data, X_scaled, min_clust, max_clust, random_s=42, acp=False, *acp_data):
        
        "Fonction qui calcul la distortion pour chaque nombre de clusters"
        distortions = []
        index_ = []
        K = range(1,30)
        for k in K:
            index_.append(i)
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(X_scaled)
            distortions.append(kmeanModel.inertia_)

        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
        distortions = pd.DataFrame(silouhette, index=index_, columns=['coef'])
    
    def silhouette(data, X_scaled, min_clust, max_clust, random_s=42, acp=False, *acp_data):
        
        """Fonction qui calcul le coefficient de silouhette pour différents nombres de
        Clusters. Renvoie un dataframe avec le coef pour chaque nombre de Clusters \n
        Le df est classé de la plus grande valeur du coef à la plus petite"""
        
        #On fixe le random 
        np.random.seed(random_s)

        #ACP si paramètre à True
        if acp == True:
            X_scaled = acp_data
            #Partie à faire, il faut penser à remettre les données transfo par la suite
            #Pour la compréhension des Clusters

        #Init silouhette list
        silouhette = []
        #init index list
        index_ = []
        for i in range(min_clust,max_clust+1):
            index_.append(i)
            #Clustering :
            kmeans = KMeans(n_clusters=i, random_state=random_s).fit(X_scaled)
            #Calcul coef de silouhette
            silhouette_avg = silhouette_score(data, kmeans.labels_)
            silouhette.append(silhouette_avg) 
            
        #Création d'un DataFrame
        silouhette_coef = pd.DataFrame(silouhette, index=index_, columns=['coef'])
        silouhette_coef = silouhette_coef.sort_values(by='coef', ascending=False).reset_index() 
        
        #Plot les coef en fonction des clusters
        ax = plt.figure().gca()
        ax.scatter(silouhette_coef['index'], silouhette_coef['coef'])
        plt.xlabel('Number of Clusters')
        plt.ylabel('Value')
        plt.title('Silhouette coefficient')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.show()
        
        return silouhette_coef
    
    
    def makeClustering(data, X_scaled, coef, random_s=42, algo='kmeans', epsilon=0.5):

        """Fonction de clustering, selon l'algo voulu par l'utilisateur. \n
        Disponible : Kmeans, Hierarchique, db_scan"""
        
        """Ici coef représente le nombre de clusters pour kmeans 
        ou le df généré par la fonction silhouette ; \n
        Mais aussi le min_sample pour Db_scan. \n
        Un seul algorithme peut être utilisé à la fois."""
        
        #On fixe le random
        np.random.seed(random_s)
        
        #On check si la variable algo est bien dans la liste disponible
        algos = ['kmeans', 'hierarchical', 'db_scan']
        if algo not in algos:
            raise ValueError("Tu n'as pas rentré le bon algorithme. Voici la liste dispo: %s" % algos)
        
        #Si l'algorithme choisi est kmeans :
        if algo == 'kmeans':
            try : 
                clust = coef.iloc[0,0] #Si le nbr clust vient de la func précédente
            except:
                clust = coef #Si le nombre de cluster vient de l'utilisateur
                
            data_clust = clustering.kmeansClustering(data, X_scaled, clust, random_s)
            
        #Si c'est du clustering hiérarchique   
        if algo == 'hierarchical':
            print('Still in build, use kmeans or db_scan for now')
        
        #Si c'est db_scan
        if algo == 'db_scan':        
            data_clust = clustering.db_scanClustering(data, X_scaled, random_s, epsilon, coef)
            
        return data_clust
    
    
    def grouping(data_clust):

        data_grouped = data_clust.groupby('Clusters').mean()
        
        return data_grouped
    
    def kmeansClustering(data, X_scaled, clust, random_s):

        np.random.seed(random_s)
            
        #Clustering avec le bon nombre de clusters
        kmeans = KMeans(n_clusters=clust, random_state=random_s, n_jobs=-1).fit(X_scaled)
        kmeans = pd.DataFrame(kmeans.labels_, index = data.index, columns=["Clusters"])

        #On merge avec nos données 
        data_clust = pd.merge(data, kmeans, left_index=True, right_index=True, how='left')
        
        return data_clust
    
    def db_scanClustering(data, X_scaled, random_s, epsilon, mini_sample, n_jobs_=-1):
        
        np.random.seed(random_s)
        
        #On entraine l'algorithme
        db = DBSCAN(eps=epsilon, min_samples=mini_sample, n_jobs=n_jobs_).fit(X_scaled)
        
        #On récupère chaque Clusters
        labels = pd.DataFrame(db.labels_, index = data.index, columns=["Clusters"])
        
        #On merge sur nos données
        data_clust = pd.merge(data, labels, left_index=True, right_index=True, how='left')
        
        return data_clust


# # Classe Plot

# In[1]:


class plotClustering():
    
    def plotPairplot(data_clust, save=False, *path):
        """Fonction permettant la création de Pairplot"""

        warnings.filterwarnings("ignore") #Ignore les messages warnings

        #Initialisation du pairplot
        sns.pairplot(data_clust, hue="Clusters") 
        
        if save == True :
            try:
                plt.savefig(path + "Pairplot.png")
            except:
                print('Missing the path for saving')
        
        plt.show()
    
    def plotBoxplot(data_clust, save=False, *path):
        """Fonction permettant la création de Boxplot"""

        
        data_ = data_clust.reset_index()

        sous_echantillon = data_.copy()
        modalites = sous_echantillon["Clusters"].unique()

        for var in data_clust.columns:
            X = "Clusters" # qualitative
            Y = var # quantitative

            groupes = []
            for m in modalites:
                groupes.append(sous_echantillon[sous_echantillon[X]==m][Y].dropna())

            medianprops = {'color':"black"}
            meanprops = {'marker':'o', 'markeredgecolor':'black',
                        'markerfacecolor':'firebrick'}

            plt.figure(figsize=[8,20])
            plt.boxplot(groupes, labels=modalites, showfliers=False, medianprops=medianprops, 
                        vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
            plt.title("Boxplot")
            plt.xlabel(var)
            plt.ylabel("Clusters")

            #Si save = True alors on enregistre
            if save == True :
                try:
                    plt.savefig(path + "boxplot" + var + ".png")
                except:
                    print('Missing the path for saving')

            #On affiche les plots
            plt.show()
    
    def plotRadarplot(data_grouped, save=False, *path):
        #On récupère le nom des features
        variables = data_grouped.columns
        
        #On récupère le range de chaque variable
        ranges = findRanges(data_grouped)
        
        #On plot chaque cluster sur un radar différent
        for i in range(0, len(data_grouped)):
            #Init la figure
            fig1 = plt.figure(figsize=(6, 6))
            #Init le radar
            radar = ComplexRadar(fig1, variables, ranges)
            
            #Mise en place des valeurs sur le radar
            radar.plot(data_grouped.loc[i,:], ranges) #Bande orange bizarre
            #Fill le radar (pour plus de clarté)
            radar.fill(data_grouped.loc[i,:], alpha=0.2)
            if save==True:
                try:
                    plt.savefig(path + "radar" + data_grouped.loc[i,:] + ".png")
                except:
                    print('Missing the path for saving')
                    
            plt.show()


# Base du code trouvées sur <a>https://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart</a>
# <br/>
# Puis modifié par mes soins pour correspondre aux données d'entrées

# In[2]:


def findRanges(data_grouped):
    ranges = []
    for i in data_grouped.columns:
        theRange = (data_grouped[i].min(), data_grouped[i].max())
        ranges.append(theRange)
    return ranges

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data_grouped, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""

    x1, x2 = ranges[0]
    d = data_grouped[0]

    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1

    sdata = [d]

    for d, (y1, y2) in zip(data_grouped[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1

        sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)

    return sdata

def set_rgrids(self, radii, labels=None, angle=None, fmt=None,
               **kwargs):
    """
    Set the radial locations and labels of the *r* grids.
    The labels will appear at radial distances *radii* at the
    given *angle* in degrees.
    *labels*, if not None, is a ``len(radii)`` list of strings of the
    labels to use at each radius.
    If *labels* is None, the built-in formatter will be used.
    Return value is a list of tuples (*line*, *label*), where
    *line* is :class:`~matplotlib.lines.Line2D` instances and the
    *label* is :class:`~matplotlib.text.Text` instances.
    kwargs are optional text properties for the labels:
    %(Text)s
    ACCEPTS: sequence of floats
    """
    # Make sure we take into account unitized data
    radii = self.convert_xunits(radii)
    radii = np.asarray(radii)
    rmin = radii.min()
    # if rmin <= 0:
    #     raise ValueError('radial grids must be strictly positive')

    self.set_yticks(radii)
    if labels is not None:
        self.set_yticklabels(labels)
    elif fmt is not None:
        self.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    if angle is None:
        angle = self.get_rlabel_position()
    self.set_rlabel_position(angle)
    for t in self.yaxis.get_ticklabels():
        t.update(kwargs)
    return self.yaxis.get_gridlines(), self.yaxis.get_ticklabels()

class ComplexRadar():
    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            # ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
            set_rgrids(ax, grid, labels=gridlabel, angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data_grouped, *args, **kw):
        sdata = _scale_data(data_grouped, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data_grouped, *args, **kw):
        sdata = _scale_data(data_grouped, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


# # Classe Clustering complet

# In[3]:


class ClustEmAll():
    def allClustering(data, min_clust, max_clust, random_s=42, algo='kmeans',
                      acp=False, save=False, *path):
        """Fonction permettant un Clustering complet avec plot des résultats"""
        
        #Initialisation des variables
        self.data = data
        self.min_clust = min_clust
        self.max_clust = max_clust
        self.random_s = random_s
        self.algo = algo
        self.acp = acp
        self.save = save
        if path:
            self.path = path
        
        #On fixe le random
        np.random.seed(random_s)
        
        #---Préparation des données
        #On centre et on réduit
        X_scaled = prepData.centrageReduction(data)
        
        #On fait une ACP si demandée
        if acp==True:
            prepData.acp(data, X_scaled) 
        else:
            pass
        
        #---Clustering
        #Calcul silhouette
        silouhette_coef = clustering.silhouette(self.data, self.min_clust, 
                                                self.max_clust, self.random_s, 
                                                self.acp)
        #Clustering
        data_clust = clustering.makeClustering(self.data, X_scaled, silouhette_coef, 
                                         self.random_s, self.algo)
        
        #Grouping
        data_grouped = clustering.grouping(data_clust)
        
        #---Plot
        #Pairplot
        plotClustering.plotPairplot(data_clust, self.save) #Penser à rajouter le path
        #Boxplot
        plotClustering.plotBoxplot(data_clust, self.save)
        #Radar
        plotClustering.plotRadarplot(data_grouped, self.save)


# In[ ]:




