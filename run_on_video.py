#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import biblioteca
import os


os.chdir("C:/Users/gvgia/Documents/Insper/3 Semestre/Robótica Computacional/APSs/APS 3")

print("Baixe o arquivo a seguir para funcionar: ")
print("https://github.com/Insper/robot202/raw/master/projeto/centro_massa/video.mp4")

cap = cv2.VideoCapture('yellow.mp4')

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    # frame = cv2.imread("frame0000.jpg")
    # ret = True
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")
        break
    else:
        mask = img.copy()
        new_mask = biblioteca.segmenta_linha_amarela(mask)
        contornos = biblioteca.encontrar_contornos(new_mask)
        centro_dos_contornos = biblioteca.encontrar_centro_dos_contornos(new_mask,contornos)
        mask = biblioteca.desenhar_linha_entre_pontos(mask,centro_dos_contornos[0],centro_dos_contornos[1],(255,0,0))
        mask,coeficientes= biblioteca.regressao_por_centro(mask,np.array(centro_dos_contornos[1]),np.array(centro_dos_contornos[2]))
        angulo = float(np.degrees(np.arctan(1/coeficientes[0])))
        cv2.putText(mask, f"Angulo: {angulo:.2f}", (40,40),cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0),2,cv2.LINE_AA)
        # Imagem original
        cv2.imshow('img',img)
        # Mascara
        cv2.imshow('mask',mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()