#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

def segmenta_linha_amarela(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem bgr e retornar uma máscara com os segmentos amarelos do centro da pista em branco.
        Utiliza a função cv2.morphologyEx() para limpar ruidos na imagem
    """
    mask = bgr.copy()
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV)
    
    low_H=40//2
    low_S=100
    low_V = 100
    high_H = 80//2
    high_S = 255
    high_V = 255
    
    mask = cv2.inRange(mask,(low_H, low_S, low_V), (high_H, high_S, high_V))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    
    mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel )
    mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kernel )
    return mask

def encontrar_contornos(mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca e retornar todos os contornos encontrados
    """
    contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8   
    """
    x,y = point
    x = int(x)
    y = int(y)
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)

def encontrar_centro_dos_contornos(bgr, contornos):
    """Não mude ou renomeie esta função
        deve receber uma lista de contornos e retornar, respectivamente,
        a imagem com uma cruz no centro de cada segmento e o centro de cada. 
        formato: img, x_list, y_list
    """

    img = bgr.copy()
    x_list = []
    y_list = []

    for contorno in contornos:
        try:
            M = cv2.moments(contorno)

            cX = int(M["m10"] / M["m00"])
            x_list.append(cX)

            cY = int(M["m01"] / M["m00"])
            y_list.append(cY)
        
            crosshair(img, (cX,cY), 5, (0,0,255))
        except:
            pass

    return img, x_list, y_list


def desenhar_linha_entre_pontos(bgr, X, Y, color):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e retornar uma imagem com uma linha entre os centros EM SEQUENCIA do mais proximo.
    """
    img = bgr.copy()
    distancia = 100000
    x_lista=[]
    y_lista=[]
    try:
        for i in range(len(X)):
            for j in range(len(X)):
                distancia_calculada = np.sqrt((X[j]-X[i])**2+(Y[j]-Y[i])**2)
                if distancia_calculada < distancia and distancia_calculada !=0:
                    distancia=distancia_calculada 
                    x1=X[j]
                    x2=X[i]
                    y1=Y[j]
                    y2=Y[i]
            cv2.line(img,(x1,y1),(x2,y2),color,2)
            distancia = 100000
        return img 
    except:
        return img

def regressao_por_centro(bgr, x_array, y_array):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta e os parametros da reta
        
        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """

    img = bgr.copy()
    lm_model = LinearRegression()
        
    yr = y_array.reshape(-1,) 
    xr = x_array.reshape(-1,1) 

    ransac = RANSACRegressor(lm_model)
    ransac.fit(xr, yr)
        
    reg = ransac.estimator_
    a, b = reg.coef_, reg.intercept_

    height, width = img.shape[:2]

    cv2.line(img,(int((height-b)/a),height),(int(-b/a),0),(0,255,0),2)  

    return img, (a,b)


def calcular_angulo_com_vertical(img, lm):
    """Não mude ou renomeie esta função
        deve receber uma imagem contendo uma reta, além da reggressão linear e determinar o ângulo da reta com a vertical, utilizando o metodo preferir.
    """
    angulo = np.degrees(np.arctan(1/lm[0]))
    return angulo

if __name__ == "__main__":
    print('Este script não deve ser usado diretamente')