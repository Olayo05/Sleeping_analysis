# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 13:07:55 2021

@author: morte
"""


# coding: utf-8


import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from pylab import *
from sklearn import metrics
import seaborn as sns

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

def agrupar_epocas(raw):
    # agrupamos en epocas de 30 muestras los datos. Por Miguel Hortelano
    #Los resumimos con algúm tipo de estadístico (en este caso energías beta, alpha y media y std)
    data = raw.get_data() * 1e6#pasamos a voltios y transfomamos raw en una matriz
    
    mediastd = np.zeros([data.shape[0]*4, int(data.shape[1]/(30*100))])
    print(mediastd.shape)
        
    for i in np.arange(0,int(mediastd.shape[1])):
                I = int(i*30)

                for j in np.arange(0,data.shape[0]):#resumimos los datos
                    mediastd[j, i] = np.mean(data[j, I:I+30])
                    mediastd[j+data.shape[0], i] = np.std(data[j, I:I+30])
                    mediastd[j+data.shape[0]*2, i] = yasa.bandpower(data[j, I:I+30], sf=100).Beta
                    mediastd[j+data.shape[0]*3, i] = yasa.bandpower(data[j, I:I+30], sf=100).Alpha
                    
    #Transponemos la matriz para que tenga las dimensiones (muestras,caracteristicas)    
    mediastd = np.array(mediastd).T
    
    return mediastd

def draw_ConfusionM(matrix,tag_list):
    
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(tag_list)
    ax.yaxis.set_ticklabels(tag_list)
    
    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
    
    
def draw_ROC(tags_true,scores,tag_list):
    #Función que dibuja las curvas roc
    #tags_true, el vector de clases verdaderas (y_train generalmente)
    #scores, el vector o matriz con las puntuaciones del modelo entrenado (scores, predict_proba, ...)
    #tag_list, la lista de etiquetas de las clases
    fig, roc = plt.subplots()
    for i in tag_list:
        fpr, tpr, thres = metrics.roc_curve(tags_true, scores[:,i],pos_label=i)
        roc.plot(fpr,tpr,"-",label = i)

    
    plt.title('Curvas ROC')
    plt.ylabel('True positives')
    plt.xlabel('False positives')
    roc.legend(title = 'Clase',bbox_to_anchor=(1,1), loc="upper left")
    plt.show()
    return(0)

def draw_silhouette(X_,cluster_labels):
    n_clusters = np.max(cluster_labels.reshape([145*145]))+1
    samples = silhouette_samples(X_,cluster_labels)
    y_lower = 10
    
    fig,ax=plt.subplots()
    silhouette_avg = silhouette_score(X_, cluster_labels)
    for i in np.arange(2,n_clusters):
        it_sample = samples[cluster_labels == i]
        it_sample.sort()
        
        size_it_sample = it_sample.shape[0]
        y_upper = y_lower + size_it_sample
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            it_sample,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,)
    
        ax.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
        
        y_lower = y_upper+10
    
    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")   
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])   
    return(0)

def extraer_caracteristicas(raw):
    #Por Nacho Arnau
    df2 = pd.DataFrame()

    for canal in raw.ch_names:

        sls = yasa.SleepStaging(raw, eeg_name = canal)
        features = sls.get_features()
        for feat in features.columns:

            name_var = feat+'_'+canal
            df2[name_var] = features[feat]
        
    return  df2

def data_loader(train_path_list = ['../data_sleep/8/8','../data_sleep/9/9'],metodo = "nada"):
    #cargamos los path a lso documentos
    train_path_list = ['../data_sleep/8/8','../data_sleep/9/9']
    out = 0
    
    #ejecutamos bucle para cada fichero
    for path in train_path_list:
        #carga y procesado inicail
        raw = mne.io.read_raw_edf(path+".edf", preload=True, verbose=False)

        raw.resample(100)
        raw.filter(0.3, 45)
        aux = raw.n_times/100
        raw.crop(tmax=(aux - 30*30), include_tmax = False)

        #raw.drop_channels(['ROC-A1', 'LOC-A2'])#eliminamos las columnas que no interesen

        #Pasamos a matriz y pasamos a mV las medidas
        data = raw.get_data() * 1e6
        
        print("50% del dataset")
        
        mediastd = np.zeros([data.shape[0]*4, int(data.shape[1]/(30*100))])
        print(mediastd.shape)

        #Colapsamos las épocas o extraemos las características según lo que queramos probar
        if metodo == "nada":
            mediastd = agrupar_epocas(raw)
        else:
            mediastd = extraer_caracteristicas(raw)
            
            
        #carga de etiquetas
        hypno = np.loadtxt(path+"_1.txt", dtype=str)[0:-30]
        hypno = tagHomo(hypno)
        
        #bloque de comprobación
        if(mediastd.shape[0] == len(hypno)):
            print("Todo correcto")
        else:
            print("Error, diferente número de muestras y etiquetas")
        
        #EN caso de cargar más de un dataset los concatenamos
        if out == 1:
            if metodo == "nada":
                out_mat = np.vstack((out_mat,mediastd))
            else:
                out_mat = pd.concat([out_mat,mediastd],axis = 0)
            
            out_tag.append(hypno)
        else:
            print("furula")

            out_mat = mediastd
            out_tag = hypno
            out = 1
        
        print("100% del dataset")
        
    return([out_mat, out_tag])

