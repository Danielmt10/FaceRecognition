#OpenCV module
import cv2
#Modulo para leer directorios y rutas de archivos
import os
import numpy as np

print('Iniciando...')

#Directorio donde se encuentran las carpetas con las caras de entrenamiento
dir_faces = 'faces'

#Tamaño para reducir a miniaturas las fotografias
size = 4

# Crear una lista de imagenes y una lista de nombres correspondientes
(images, lables, names, id) = ([], [], {}, 0)
for (subdir, dirs, files) in os.walk(dir_faces):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dir_faces, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1

#Dimensiones de la imagen
im_width, im_height = 112,92

# Crear una matriz Numpy de las dos listas anteriores
(images, lables) = [np.array(lis) for lis in [images, lables]]
# OpenCV entrena un modelo a partir de las imagenes
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

#Utilizar el modelo entrenado en funcionamiento con la camara
#Se instancia el clasificador cascada basado en el rostro frontal
face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')

#Si la camara está encendida se apaga para evitar una excepcion 
if cv2.VideoCapture.isOpened:
    cap = cv2.VideoCapture(1)
    
cap = cv2.VideoCapture(0)

while True:
    #lee un frame y lo guarda
    rval, frame = cap.read()
    frame = cv2.flip(frame,1,0)

    #converir la imagen a blanco y negro    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #redimensionar la imagen
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    #Buscar las coordenadas de los rostros
    faces = face_cascade.detectMultiScale(mini)
    
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        #Intentado reconocer la cara
        prediction = model.predict(face_resize)
        
        #Dibujamos un rectangulo en las coordenadas del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Escribiendo el nombre de la cara reconocida
        # La variable cara tendra el nombre de la persona reconocida
        faceName = '%s' % (names[prediction[0]])

        #Si la prediccion tiene una exactitud menor a 75 se toma como prediccion valida
        if prediction[1]<75 :
            #Ponemos el nombre de la persona que se reconoció
            cv2.putText(frame,'%s -%f' % (faceName, prediction[1]),(x-10, y-10), cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,1,(0, 255, 0))

        #Si la prediccion es mayor a 100 no es un reconomiento con la exactitud suficiente
        elif prediction[1]>75 and prediction[1]<500:           
            #Si la cara es desconocida, poner desconocido
            cv2.putText(frame, '%s -%f' % ('Desconocido', prediction[1]),(x-10, y-10),cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,1,(0, 255, 0))  

        #Mostramos la imagen
        cv2.imshow('Facial Recognition', frame)

    #Si se presiona la tecla ESC se cierra el programa
    key = cv2.waitKey(10)
    if key == 27:
        if cap.isOpened():
            cap=cv2.VideoCapture(1)
        break

cap.release()
cv2.destroyAllWindows()