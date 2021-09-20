#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

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

def magnitude_do_gradiente(img, showfig=False):
    
    # Filtro de Sobel para a derivada ao longo de X
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)

    # Filtro de Sobel para a derivada ao longo de Y
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    
    # Magnitude do gradiente
    mag_gradiente = (sobelx**2 + sobely**2)**0.5
    
    # Converte a imagem de ponto flutuante de 64 bits para imagem de 8 bits
    mag_gradiente = cv2.convertScaleAbs(mag_gradiente) 
    
    return mag_gradiente

def desenha_retas(image, lines):
    '''
    Desenha as retas encontradas pela transformada de Hough
    '''
    if len(image.shape) < 3:
        imout = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        imout = image.copy()

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(imout, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    
    return imout

def segmenta_linha_branca(bgr):
    """Não mude ou renomeie esta função
        deve receber uma imagem e segmentar as faixas brancas
    """
    mask = bgr.copy()
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2HSV)
    
    low_H=0
    low_S=0
    low_V = 200
    high_H = 360//2
    high_S = 100
    high_V = 255
    
    mask = cv2.inRange(mask,(low_H, low_S, low_V), (high_H, high_S, high_V))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    
    mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel )
    mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kernel )
    return mask

def estimar_linha_nas_faixas(img, mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca e retorna dois pontos que formen APENAS uma linha em cada faixa. Desenhe cada uma dessas linhas na iamgem.
         formato: [[(x1,y1),(x2,y2)], [(x1,y1),(x2,y2)]]
    """
    mag_gradiente = magnitude_do_gradiente(mask)
    threshold_value = 0
    retval, bordas = cv2.threshold(mag_gradiente, threshold_value, 100, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(bordas, 10, math.pi/180.0, threshold=875, minLineLength=150, maxLineGap=5)
    
    a,b,c = lines.shape

    hough_img_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    for i in range(a):
    # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
        cv2.line(hough_img_rgb, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 5, cv2.LINE_AA)
    return lines,hough_img_rgb

def calcular_equacao_das_retas(linhas):
    """Não mude ou renomeie esta função
        deve receber dois pontos que estejam em cada uma das faixas e retornar a equacao das duas retas. Onde y = h + m * x. Formato: [(m1,h1), (m2,h2)]
    """
    equacoes=[]
    for linha in linhas:
        m=(linha[0][1]-linha[0][3])/(linha[0][0]-linha[0][2])
        h=linha[0][1]-linha[0][0]*m
        equacoes.append([m,h])
    
    return equacoes

def calcular_ponto_de_fuga(img, equacoes):
    """Não mude ou renomeie esta função
        deve receber duas equacoes de retas e retornar o ponto de encontro entre elas. Desenhe esse ponto na imagem.
    """
    equacao1, equacao2 = equacoes

    m1,h1=equacao1
    m2,h2=equacao2

    xi=int((h2-h1)/(m1-m2))
    yi=int(m1*xi+h1)

    height, width = img.shape[:2]

    crosshair(img, (xi, yi), 15, (0, 0, 255))

    cv2.line(img,((int((height-h1)/m1) ,height)),(xi,yi),(255, 0,0),2)
    cv2.line(img,((int((height-h2)/m2) ,height)),(xi ,yi),(255, 0,0),2)
    return img, (xi,yi)


