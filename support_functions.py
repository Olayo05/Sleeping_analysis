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

def aviris_data_load():
        
    # Lectura de la imagen de fichero de Matlab .mat
    mat_file =  "datasetB1.mat"
    mat = matlab.loadmat(mat_file,squeeze_me=True) #devuelve un dictionary
    list(mat.keys()) #variables almacenadas
    
    
    # Lectura de los datos
    X = mat["X"]   #imagen (hipercubo 3D: filas x columnas x variables) 
    Xl = mat["Xl"]   #muestras etiquetadas (muestas x variables) 
    Yl = mat["Yl"]   #etiquetas de clases (muestras x 1, 0=sin clase)   
    del mat
    Yl.shape
    
    
    # Reshape del Ground Truth como una imagen
    Y = np.reshape(Yl, (X.shape[0], X.shape[1]),order="F")
    Y.shape
    
    
    # Filter background: eliminamos la clase 0 de los datos etiquetados
    Nc=Yl.max()-Yl.min()+1
    if Nc>2:
        Xl = Xl[Yl != 0,:];
        Yl = Yl[Yl != 0];
    
    return([X,Y,Xl,Yl])

def data_reducer():
    X,Y,Xl,Yl = aviris_data_load()

    n_tags = max(Yl)
    ratio = 5000/Xl.shape[0]
    #separamos según etiqueta
    tag_list = np.arange(1,n_tags+1)#lista de etiquetas
    tag_index_list = [np.where(Yl == i) for i in tag_list]#lista con las posiciones de cada etiqueta
    #subconjuntos de Yl y Xl de cada etiqueta
    X_index_sample= [Xl[indx,:] for indx in tag_index_list]
    
    
    #preparamos los vectores reducidos
    X_reduced = []
    Y_reduced = []
    
    #Llenamos los libros reducidos
    for i in range(len(tag_list)):
        
        data = X_index_sample[i][0,:,:]
        n_points_reduced = int(np.ceil(data.shape[0]*ratio))
        cluster = KMeans(n_points_reduced).fit(data)
        centers = cluster.cluster_centers_.squeeze()
        
        newdata = list()
        
        #Seleccionamos el punto más cercano al centroide para evitar distorisionar la distribución de los puntos
        for z in range(centers.shape[0]):
            d = [np.linalg.norm(data[j,:]-centers[z,:]) for j in range(data.shape[0])]
            indx = d.index(min(d))
            newdata.append(data[indx])
            
            
        X_reduced.append(newdata)
        Y_reduced.append(np.ones(n_points_reduced)*tag_list[i])
    
    X_reduced = np.concatenate(X_reduced)
    Y_reduced = np.concatenate(Y_reduced)
    
    return((X_reduced,Y_reduced))
    
def class_map(X_,image,tag = ["Kmeans",0,0],):
    a = image.shape
    cluster_labels = image.reshape([a[0]*a[1]])
    n_clusters = np.max(cluster_labels)+1
    
    #Parte donde se dibuja la imagen
    cmap = cm.get_cmap('tab20c', n_clusters) 
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    plt.imshow(image,cmap = cmap)
    plt.colorbar()
    if tag[0]=="Kmeans":
        plt.title("Kmeans "+str(n_clusters)+" clusters")
    elif tag[0]=="Jerárquico":
        plt.title("Jerárquico "+str(n_clusters)+" clusters")
    elif tag[0]=="Gaussianas":
        plt.title("Gaussianas "+str(n_clusters)+" clusters tipo "+tag[2])
    else:
        plt.title(tag[0]+" "+str(tag[1])+" e "+str(tag[2])+" n min")
    
    #Galimatías donde se dibujan los silhouette
    samples = silhouette_samples(X_,cluster_labels)
    y_lower = 10
    
    silhouette_avg = silhouette_score(X_, cluster_labels)
    for i in np.arange(0,n_clusters):
        it_sample = samples[cluster_labels == i]
        it_sample.sort()
        
        size_it_sample = it_sample.shape[0]
        y_upper = y_lower + size_it_sample
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            it_sample,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,)
    
        ax1.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
        
        y_lower = y_upper+10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")   
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    return(0)

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
        fpr, tpr, thres = metrics.roc_curve(tags_true, scores[:,i-1],pos_label=i)
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

def draw_image(image,method):
    n_clases = int(max(image))
    cmap = cm.get_cmap('tab20c', n_clases) 
    
    fig, (ax1) = plt.subplots(figsize=(5,5))
    plt.imshow(image.reshape([145,145]),cmap = cmap)
    plt.colorbar()
    plt.title(method+" "+str(n_clases)+" clases")
    ax = plt.gca()

    #hide x-axis
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    #hide y-axis

def draw_silhouette_plus(X_reshape,clstrs_Y_K9,clstrs_Y_G8,clstrs_Y_G18):
    
    sh = [0,0,0]
    
    fig, (sh1,sh2,sh3) = plt.subplots(1,3)

    samples = silhouette_samples(X_reshape,clstrs_Y_K9)
    y_lower = 10
    n_clusters = 9
    silhouette_avg = silhouette_score(X_reshape,clstrs_Y_K9)
    sh[0] = silhouette_avg
    
    for i in np.arange(0,n_clusters):
        it_sample = samples[clstrs_Y_K9 == i]
        it_sample.sort()
        
        size_it_sample = it_sample.shape[0]
        y_upper = y_lower + size_it_sample
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        sh1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            it_sample,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,)
    
        sh1.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
        
        y_lower = y_upper+10
    
    sh1.set_title("KMeans")
    #sh1.set_xlabel("The silhouette coefficient values")
    sh1.set_ylabel("Cluster label")
    sh1.axvline(x=silhouette_avg, color="red", linestyle="--")   
    sh1.set_yticks([])  # Clear the yaxis labels / ticks
    sh1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    samples = silhouette_samples(X_reshape,clstrs_Y_G8)
    y_lower = 10
    n_clusters = 8
    silhouette_avg = silhouette_score(X_reshape,clstrs_Y_G8)
    sh[1] = silhouette_avg
    
    for i in np.arange(0,n_clusters):
        it_sample = samples[clstrs_Y_G8.reshape(145*145) == i]
        it_sample.sort()
        
        size_it_sample = it_sample.shape[0]
        y_upper = y_lower + size_it_sample
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        sh2.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            it_sample,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,)
    
        sh2.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
        
        y_lower = y_upper+10
    
    sh2.set_title("Gaussian Mixture")
    sh2.set_xlabel("Silhouette coefficients")
    sh2.set_ylabel("Cluster label")
    sh2.axvline(x=silhouette_avg, color="red", linestyle="--")   
    sh2.set_yticks([])  # Clear the yaxis labels / ticks
    sh2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    n_clusters = 18
    samples = silhouette_samples(X_reshape,clstrs_Y_G18)
    y_lower = 10

    silhouette_avg = silhouette_score(X_reshape,clstrs_Y_G18)
    sh[2] = silhouette_avg
    for i in np.arange(0,n_clusters):
        it_sample = samples[clstrs_Y_G18 == i]
        it_sample.sort()
        
        size_it_sample = it_sample.shape[0]
        y_upper = y_lower + size_it_sample
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        sh3.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            it_sample,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,)
    
        sh3.text(-0.05, y_lower + 0.5 * size_it_sample, str(i))
        
        y_lower = y_upper+10
    
    sh3.set_title("Gaussian Mixture")
    #sh3.set_xlabel("The silhouette coefficient values")
    sh3.set_ylabel("Cluster label")
    sh3.axvline(x=silhouette_avg, color="red", linestyle="--")   
    sh3.set_yticks([])  # Clear the yaxis labels / ticks
    sh3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    
    objects = ('Kmeans 9',' Gaussian Mixture 8', 'Gaussian Mixture 18')
    fig, (barsh) = subplots()
    ypos = [1,2,3]
    barsh.bar(ypos, sh, align='center', alpha=0.5)
    plt.xticks(ypos, objects)
    plt.ylabel('Silhouette')
    plt.title('Coeficientes Silhouette medio')

    plt.show()