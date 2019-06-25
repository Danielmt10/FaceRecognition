#OpenCV module
import cv2
#Modulo para leer directorios y rutas de archivos
import os
import shutil

#cargar la plantilla e inicializar la webcam
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if cv2.VideoCapture.isOpened:
    cap = cv2.VideoCapture(1)
    
cap = cv2.VideoCapture(0)

print('Por favor ingrese el nombre de la persona a escanear:')
nombre=input()

#Directorio donde se encuentra la carpeta con el nombre de la persona
dir_faces = 'faces'
path = os.path.join(dir_faces, nombre)

#Tamaño para reducir a miniaturas las fotografias
size = 4

#Si no hay una carpeta con el nombre ingresado entonces se crea
if not os.path.isdir(path):
    os.mkdir(path)
    
else:
    shutil.rmtree(path)
    os.mkdir(path)

img_width, img_height = 112, 92

#Ciclo para tomar fotografias
count = 0
while count < 500:
    #leemos un frame y lo guardamos
    rval, img = cap.read()
    img = cv2.flip(img,1,0)

    #convertir la imagen a blanco y negro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #redimensionar la imagen
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    #Buscar las coordenadas de los rostros 
    faces = face_cascade.detectMultiScale(mini)    
    faces = sorted(faces, key=lambda x: x[3])
    
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (img_width, img_height))
        
        #Dibujamos un rectangulo en las coordenadas del rostro
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)   

        #El nombre de cada foto es el numero del ciclo
        #Obtenemos el nombre de la foto
        #Despues de la ultima sumamos 1 para continuar con los demas nombres
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
               if n[0]!='.' ]+[0])[-1] + 1

        #Metemos la foto en el directorio
        cv2.imwrite('%s/%s.jpg' % (path, pin), face_resize)

        #Contador del ciclo
        count += 1

    #Mostramos la imagen
    cv2.imshow('Facial Recognition ', img)

    #Si se presiona la tecla ESC se cierra el programa
    key = cv2.waitKey(10)
    if key == 27 or count==500:
        cap=cv2.VideoCapture(1)
        break
    
cap.release()
cv2.destroyAllWindows()