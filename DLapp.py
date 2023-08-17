import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import time
import numpy as np
from PIL import Image
import pandas as pd
import os
from math import acos, degrees
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D,MaxPooling1D,Dropout,Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


DEMO_VIDEO = '28_cut.mp4'
FRAME_WINDOW = st.image([])

#st.title('Aplicativo web para la predicción de patologías en la marcha humana')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>

    """,
    unsafe_allow_html=True
)

st.sidebar.title("Barra de parametros")
st.sidebar.subheader("Parametros")

@st.cache_data()
def image_resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))
    
    #redimensionar imagen
    resized = cv2.resize(image,dim,interpolation=inter)
    
    return resized

app_mode = st.sidebar.selectbox('Seleccione el modo de la App',
['Acerca de la App', 'Ejecutar en video'])

if app_mode == 'Acerca de la App':
    st.title('Aplicativo web para la predicción de patologías en la marcha humana')
    st.header('Pasos de la utilización del aplicativo web')
    st.markdown('1. Ingresa "ejecutar en video" desde la barra de parámetros.')
    st.markdown('2. Por defecto el modelo presentado es el de "coordenadas", en la barra desplegable escoger el que se quiera probar.')
    st.markdown('3. Seleccionar el modo de uso, ya sea con la webcam o subiendo un video.')
    st.markdown('Si se seleccionó subir video, puedes subir uno de hasta 200MB.')
    st.markdown('En ambos modos puedes modificar los parámetros de seguimiento y detección')

    st.markdown('---')
    
    st.header("Acerca del autor")
    st.write("Nombre: Sebastian Andreiv Jaimes Gómez")
    st.write("Estudiante de décimo semestre de la Universidad de Pamplona")
    st.markdown('---')
    st.header("Tutor")
    st.write("Msc. Luis Enrique Mendoza")
    

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>

        """,
        unsafe_allow_html=True
    )
    img = Image.open('logoupa.png')
    st.markdown('---')
    st.image(img,width=300, caption='Logo Universidad de Pamplona')
    
elif app_mode == 'Ejecutar en video':

    model_app = st.sidebar.selectbox('Seleccione el modelo de la app',
    ['Modelo de coordenadas', 'Modelo de angulos'])

    if model_app == 'Modelo de coordenadas':

        st.set_option('deprecation.showfileUploaderEncoding', False)

        use_webcam = st.sidebar.button('Usar webcam')

        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
                width: 350px
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
                width: 350px
                margin-left: -350px
            }
            </style>

            """,
            unsafe_allow_html=True
        )

        detection_confidence = st.sidebar.slider('Minima confianza de detección',min_value=0.0,max_value=1.0,value=0.5)
        tracking_confidence = st.sidebar.slider('Minima confianza de seguimiento',min_value=0.0,max_value=1.0,value=0.5)
        st.sidebar.markdown('---')

        st.markdown('## Salida')

        stframe = st.empty
        video_file_buffer = st.sidebar.file_uploader("Subir video",type=['mp4','mov','avi'])
        tfflie = tempfile.NamedTemporaryFile(delete=False)
    
        if not video_file_buffer:
            if use_webcam:
                cap = cv2.VideoCapture(1)
            else:
                cap = cv2.VideoCapture(DEMO_VIDEO)
                tfflie.name = DEMO_VIDEO
        
        else:
           tfflie.write(video_file_buffer.read())
           cap = cv2.VideoCapture(tfflie.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))

        st.sidebar.text('Video de entrada')
        st.sidebar.video(tfflie.name)
        #fps = 0
        #i = 0
        #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        kpi1, kpi3 = st.columns(2)

        with kpi1:
            st.markdown("**Movimiento**")
            kpi1_text = st.markdown("0")


        st.markdown("<hr/>", unsafe_allow_html=True)

        # cargando los datos de las carpetas recolectadas
        DATA_PATH = 'Datos_Marcha'

        # acciones a detectar
        desp_normal = np.array(['normal'])
        desp_hemiplejia = np.array(['hemiplejia'])
        desp_estepaje = np.array(['estepaje'])
        desp_balanceante = np.array(['balanceante'])
        desp_parkinson = np.array(['parkinson'])

        acciones = np.array([desp_normal, desp_hemiplejia, desp_estepaje, desp_balanceante, desp_parkinson]).flatten()

        #función de toma de puntos clave
        def extraer_puntos_clave(results):
            pose_1 = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark[11:17]]).flatten()
            pose_2 = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark[23:29]]).flatten()
            pose_3 = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark[31:33]]).flatten()
            return np.concatenate([pose_1,pose_2,pose_3])
        
        model = tf.keras.models.load_model('acciones_completasV2.h5')

        sequence = []
        sentence = []
        predicciones = []
        threshold = 0.4
        movimiento = ''


        with mp_pose.Pose(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence) as pose:
            #prevTime = 0

            while cap.isOpened():
                #i +=1
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks is not None:
                        
                        hom_izq = [int(results.pose_landmarks.landmark[11].x * width),
                                    int(results.pose_landmarks.landmark[11].y * height)]
                        
                        hom_der = [int(results.pose_landmarks.landmark[12].x * width),
                                    int(results.pose_landmarks.landmark[12].y * height)]

                        cod_izq = [int(results.pose_landmarks.landmark[13].x * width),
                                    int(results.pose_landmarks.landmark[13].y * height)]

                        cod_der = [int(results.pose_landmarks.landmark[14].x * width),
                                    int(results.pose_landmarks.landmark[14].y * height)]

                        mu_izq = [int(results.pose_landmarks.landmark[15].x * width),
                                    int(results.pose_landmarks.landmark[15].y * height)]

                        mu_der = [int(results.pose_landmarks.landmark[16].x * width),
                                    int(results.pose_landmarks.landmark[16].y * height)]

                        cad_izq = [int(results.pose_landmarks.landmark[23].x * width),
                                    int(results.pose_landmarks.landmark[23].y * height)]

                        cad_der = [int(results.pose_landmarks.landmark[24].x * width),
                                    int(results.pose_landmarks.landmark[24].y * height)]

                        rod_izq = [int(results.pose_landmarks.landmark[25].x * width),
                                    int(results.pose_landmarks.landmark[25].y * height)]

                        rod_der = [int(results.pose_landmarks.landmark[26].x * width),
                                    int(results.pose_landmarks.landmark[26].y * height)]

                        to_izq = [int(results.pose_landmarks.landmark[27].x * width),
                                    int(results.pose_landmarks.landmark[27].y * height)]

                        to_der = [int(results.pose_landmarks.landmark[28].x * width),
                                    int(results.pose_landmarks.landmark[28].y * height)]

                        pp_izq = [int(results.pose_landmarks.landmark[31].x * width),
                                    int(results.pose_landmarks.landmark[31].y * height)]

                        pp_der = [int(results.pose_landmarks.landmark[32].x * width),
                                    int(results.pose_landmarks.landmark[32].y * height)]

                        points = [hom_izq,hom_der,cod_izq,cod_der,
                                mu_izq,mu_der,cad_izq,cad_der,
                                rod_izq,rod_der,to_izq,to_der,
                                pp_izq,pp_der]
                        
                        keypoints = extraer_puntos_clave(results)

                        sequence.append(keypoints)
                        sequence = sequence[-90:]

                        if len(sequence) == 90:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            movimiento = acciones[np.argmax(res)]
                            predicciones.append(movimiento)
                            print(movimiento)

                            if (res[np.argmax(res)] > threshold): 
                                if len(sentence) > 0: 
                                    if acciones[np.argmax(res)] != sentence[-1]:
                                            sentence.append(acciones[np.argmax(res)])
                                    else:
                                        sentence.append(acciones[np.argmax(res)])

                                if len(sentence) > 4: 
                                    sentence = sentence[-4:]

                        for point in points:
                            cv2.circle(frame,(point),6,(255,255,0),4)
                            
                        torax = np.array([[hom_izq],[cad_izq],[cad_der], [hom_der]],np.int32)
                        cv2.polylines(frame,[torax],True,(0,255,255),5)
                        
                        brazo_izq = np.array([[hom_izq],[cod_izq],[mu_izq]],np.int32)
                        cv2.polylines(frame,[brazo_izq],False,(0,255,255),5)

                        brazo_der = np.array([[hom_der],[cod_der],[mu_der]],np.int32)
                        cv2.polylines(frame,[brazo_der],False,(0,255,255),5)

                        pierna_izq = np.array([[cad_izq],[rod_izq],[to_izq], [pp_izq]],np.int32)
                        cv2.polylines(frame,[pierna_izq],False,(0,255,255),5)

                        pierna_der = np.array([[cad_der],[rod_der],[to_der], [pp_der]],np.int32)
                        cv2.polylines(frame,[pierna_der],False,(0,255,255),5)


                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{movimiento}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 640)
                FRAME_WINDOW.image(frame,channels = 'BGR')

        st.text('Video Procesado')

        cap.release()
        #out.release()
    
    if model_app == 'Modelo de angulos':

        st.set_option('deprecation.showfileUploaderEncoding', False)

        use_webcam = st.sidebar.button('Usar webcam')

        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
                width: 350px
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
                width: 350px
                margin-left: -350px
            }
            </style>

            """,
            unsafe_allow_html=True
        )

        detection_confidence = st.sidebar.slider('Minima confianza de detección',min_value=0.0,max_value=1.0,value=0.5)
        tracking_confidence = st.sidebar.slider('Minima confianza de seguimiento',min_value=0.0,max_value=1.0,value=0.5)
        st.sidebar.markdown('---')

        st.markdown('## Salida')

        stframe = st.empty
        video_file_buffer = st.sidebar.file_uploader("Subir video",type=['mp4','mov','avi'])
        tfflie = tempfile.NamedTemporaryFile(delete=False)
    
        if not video_file_buffer:
            if use_webcam:
                cap = cv2.VideoCapture(1)
            else:
                cap = cv2.VideoCapture(DEMO_VIDEO)
                tfflie.name = DEMO_VIDEO
        
        else:
            tfflie.write(video_file_buffer.read())
            cap = cv2.VideoCapture(tfflie.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))

        st.sidebar.text('Video de entrada')
        st.sidebar.video(tfflie.name)
        #fps = 0
        #i = 0
        #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        kpi1, kpi3 = st.columns(2)

        with kpi1:
            st.markdown("**Movimiento**")
            kpi1_text = st.markdown("0")

        

        st.markdown("<hr/>", unsafe_allow_html=True)

        #función de toma de puntos clave
        def extraer_puntos_clave(results):
            pose_1 = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark[11:17]]).flatten()
            pose_2 = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark[23:29]]).flatten()
            pose_3 = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark[31:33]]).flatten()
            return np.concatenate([pose_1,pose_2,pose_3])

        def calcular_angulo(a,b,c):
            #global angulo
            angulo = degrees(acos((a**2 + c**2 - b**2)/(2*a*c)))
            return angulo
            
        def elemento_mas_comun(lista):
                    contador = {}
                    for elemento in lista:
                        if elemento in contador:
                            contador[elemento] += 1
                        else:
                            contador[elemento] = 1
                    
                    elemento_mas_repetido = None
                    max_repeticiones = 0
                    
                    for elemento, repeticiones in contador.items():
                        if repeticiones > max_repeticiones:
                            max_repeticiones = repeticiones
                            elemento_mas_repetido = elemento
                    
                    return elemento_mas_repetido, max_repeticiones

        sentence = []
        threshold = 0.4
        movimiento = ''
        
        lista_ang_rod_izq = []
        lista_ang_rod_der = []
        lista_ang_to_izq = []
        lista_ang_to_der = []
        lista_ang_cod_izq = []
        lista_ang_cod_der = []
        lista_dist_rod = []
        lista_dist_to = []
        lista_dist_mu = []

        df_angulos_normal = pd.DataFrame()

        model= joblib.load('modelo_angulosV4.pkl')

        with mp_pose.Pose(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence) as pose:
            #prevTime = 0

            while cap.isOpened():
                #i +=1
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks is not None:

                    # hombro izquierdo
                    hom_izq = [int(results.pose_landmarks.landmark[11].x * width),
                            int(results.pose_landmarks.landmark[11].y * height)]
                    hom_izq = np.array(hom_izq)

                        # hombro derecho
                    hom_der = [int(results.pose_landmarks.landmark[12].x * width),
                            int(results.pose_landmarks.landmark[12].y * height)]
                    hom_der = np.array(hom_der)
                    # codo izquierdo
                    cod_izq = [int(results.pose_landmarks.landmark[13].x * width),
                                int(results.pose_landmarks.landmark[13].y * height)]
                    cod_izq = np.array(cod_izq)

                    # codo derecho
                    cod_der = [int(results.pose_landmarks.landmark[14].x * width),
                                int(results.pose_landmarks.landmark[14].y * height)]
                    cod_der = np.array(cod_der)

                    # muñeca izquierda
                    mu_izq = [int(results.pose_landmarks.landmark[15].x * width),
                                int(results.pose_landmarks.landmark[15].y * height)]
                    mu_izq = np.array(mu_izq)

                    # muñeca derecha
                    mu_der = [int(results.pose_landmarks.landmark[16].x * width),
                                int(results.pose_landmarks.landmark[16].y * height)]
                    mu_der = np.array(mu_der)
                        # cadera izquierda
                    cad_izq = [int(results.pose_landmarks.landmark[23].x * width),
                        int(results.pose_landmarks.landmark[23].y * height)]
                    cad_izq = np.array(cad_izq)

                    # cadera derecha
                    cad_der = [int(results.pose_landmarks.landmark[24].x * width),
                            int(results.pose_landmarks.landmark[24].y * height)]
                    cad_der = np.array(cad_der)

                        # rodilla izquierda
                    rod_izq = [int(results.pose_landmarks.landmark[25].x * width),
                        int(results.pose_landmarks.landmark[25].y * height)]
                    rod_izq = np.array(rod_izq)

                    # rodilla derecha
                    rod_der = [int(results.pose_landmarks.landmark[26].x * width),
                        int(results.pose_landmarks.landmark[26].y * height)]
                    rod_der = np.array(rod_der)

                    # tobillo IZQ
                    to_izq = [int(results.pose_landmarks.landmark[27].x * width),
                        int(results.pose_landmarks.landmark[27].y * height)]
                    to_izq = np.array(to_izq)

                    # tobillo DERECHO
                    to_der = [int(results.pose_landmarks.landmark[28].x * width),
                        int(results.pose_landmarks.landmark[28].y * height)]
                    to_der = np.array(to_der)

                        # punta de pie IZQUIERDA
                    pp_izq = [int(results.pose_landmarks.landmark[31].x * width),
                            int(results.pose_landmarks.landmark[31].y * height)]
                    pp_izq = np.array(pp_izq)
                        # punta de pie 2
                        
                    pp_der = [int(results.pose_landmarks.landmark[32].x * width),
                            int(results.pose_landmarks.landmark[32].y * height)]
                    pp_der = np.array(pp_der)
                    #distancia entre hombro izq y codo izq
                    dist_hom_cod_izq = np.linalg.norm(cod_izq - hom_izq)

                                    #distancia entre hombro izq y codo izq
                    dist_hom_cod_der = np.linalg.norm(cod_der - hom_der)

                                    #distnacia entre codo izq y muñeca izq
                    dist_cod_mu_izq = np.linalg.norm(mu_izq - cod_izq)

                                    #distnacia entre codo izq y muñeca izq
                    dist_cod_mu_der = np.linalg.norm(mu_der - cod_der)

                                    #distancia entre hom izq y mu izq
                    dist_hom_mu_izq = np.linalg.norm(mu_izq - hom_izq)

                                    #distancia entre hom izq y mu izq
                    dist_hom_mu_der = np.linalg.norm(mu_der - hom_der)

                                    # distancia entre cadera y rodilla izq
                    dist_cad_rod_izq = np.linalg.norm(rod_izq - cad_izq)

                                    # distancia entre cadera y tobillo izq
                    dist_cad_to_izq = np.linalg.norm(to_izq - cad_izq)

                                    # distancia entre cadera y rodilla der
                    dist_cad_rod_der = np.linalg.norm(rod_der - cad_der)

                                    # distancia entre cadera y tobillo der
                    dist_cad_to_der = np.linalg.norm(to_der - cad_der)

                                    # distancia entre rodilla y tobillo izq
                    dist_rod_to_izq = np.linalg.norm(to_izq - rod_izq)

                                    # distancia entre rodilla y tobillo der
                    dist_rod_to_der = np.linalg.norm(to_der - rod_der)

                                    # distancia entre tobillo y punta de pie izq
                    dist_to_pp_izq = np.linalg.norm(pp_izq - to_izq)

                                    #distancia entre tobilla y punta de pie der
                    dist_to_pp_der = np.linalg.norm(pp_der - to_der)

                                    # distancia entre rodilla y punta de pie izq
                    dist_rod_pp_izq = np.linalg.norm(pp_izq - rod_izq)

                                    # distancia entre rodilla y punta de pie der: (xy4,xy8)
                    dist_rod_pp_der = np.linalg.norm(pp_der - rod_der)
                    #distancia rod izq rod der
                    dist_rod = np.linalg.norm(rod_izq - rod_der)
                    
                    #distancia to izq to der
                    dist_to = np.linalg.norm(to_izq - to_der)
                    
                    #distancia muñecas
                    dist_mu = np.linalg.norm(mu_izq - mu_der)


                                    # ANGULOS ENTRE PUNTOS
                    ang_rod_izq = calcular_angulo(dist_rod_to_izq,dist_cad_to_izq,dist_cad_rod_izq)

                    ang_rod_der = calcular_angulo(dist_rod_to_der,dist_cad_to_der,dist_cad_rod_der)

                    ang_to_izq = calcular_angulo(dist_to_pp_izq,dist_rod_to_izq,dist_rod_pp_izq)

                    ang_to_der = calcular_angulo(dist_to_pp_der,dist_rod_to_der,dist_rod_pp_der)

                    ang_cod_izq = calcular_angulo(dist_cod_mu_izq,dist_hom_mu_izq,dist_hom_cod_izq)

                    ang_cod_der = calcular_angulo(dist_cod_mu_der,dist_hom_mu_der,dist_hom_cod_der)

                    lista_ang_rod_izq.append(ang_rod_izq)
                    lista_ang_rod_der.append(ang_rod_der)

                    lista_ang_to_der.append(ang_to_der)
                    lista_ang_to_izq.append(ang_to_izq)

                    lista_ang_cod_izq.append(ang_cod_izq)
                    lista_ang_cod_der.append(ang_cod_der)
                    
                    lista_dist_rod.append(dist_rod)
                    lista_dist_to.append(dist_to)
                    lista_dist_mu.append(dist_mu)
                    

                    points = [hom_izq,hom_der,cod_izq,cod_der,
                                mu_izq,mu_der,cad_izq,cad_der,
                                rod_izq,rod_der,to_izq,to_der,
                                pp_izq,pp_der]

                    for point in points:
                        cv2.circle(frame,(point),6,(255,255,0),4)
                    torax = np.array([[hom_izq],[cad_izq],[cad_der], [hom_der]],np.int32)
                    cv2.polylines(frame,[torax],True,(0,255,255),5)
                    
                    brazo_izq = np.array([[hom_izq],[cod_izq],[mu_izq]],np.int32)
                    cv2.polylines(frame,[brazo_izq],False,(0,255,255),5)

                    brazo_der = np.array([[hom_der],[cod_der],[mu_der]],np.int32)
                    cv2.polylines(frame,[brazo_der],False,(0,255,255),5)

                    pierna_izq = np.array([[cad_izq],[rod_izq],[to_izq], [pp_izq]],np.int32)
                    cv2.polylines(frame,[pierna_izq],False,(0,255,255),5)

                    pierna_der = np.array([[cad_der],[rod_der],[to_der], [pp_der]],np.int32)
                    cv2.polylines(frame,[pierna_der],False,(0,255,255),5)
                    
                df_angulos_normal = pd.DataFrame({"rod_izq":lista_ang_rod_izq,"rod_der":lista_ang_rod_der,
                            "to_izq":lista_ang_to_izq,"to_der":lista_ang_to_der,
                                "cod_izq":lista_ang_cod_izq,"cod_der":lista_ang_cod_der,
                                "dist_rod":lista_dist_rod,"dist_to":lista_dist_to, "dist_mu":lista_dist_mu})
                
                scaler = StandardScaler()
                df_sin_nans = df_angulos_normal.fillna(0)
                angulos_fitted = scaler.fit(df_sin_nans)
                #angulos_escalados = scaler.transform(df_sin_nans)
                #new_predictions = model.predict(angulos_escalados)

                new_data_scaled = scaler.transform(df_angulos_normal)
                new_predictions = model.predict(new_data_scaled)

                #print("Nuevas predicciones:", new_predictions)
                
                

                elemento, repeticiones = elemento_mas_comun(new_predictions)
                
                if elemento == 0:
                    movimiento = "Marcha normal"
                elif elemento == 1:
                    movimiento = "Marcha hemiplejia"
                elif elemento == 2:
                    movimiento = "Marcha estepaje"
                elif elemento == 3:
                    movimiento = "Marcha balanceante"
                elif elemento == 4: 
                    movimiento = "Marcha parkinson"
                
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{movimiento}</h1>", unsafe_allow_html=True)
                
                frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 640)
                FRAME_WINDOW.image(frame,channels = 'BGR')

            
            

        st.text('Video Procesado')

        cap.release()
        out.release()
