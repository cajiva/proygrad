import pickle
import multiprocessing
import statistics
import time
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tensorflow
from tensorflow import keras
from PIL import Image
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tqdm import tqdm
import multiprocessing
from mtcnn.mtcnn import MTCNN
import signal
import os
import pandas as pd
from pickle import dump


def init_worker():
    ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def extract_face(frame,detector,required_size=(224, 224)):
        # create the detector, using default wei	ghts
        # detect faces in the image
        faces = detector.detect_faces(frame)
        # extract the bounding box from the first face
        face_images = []
        arr =[]
        for face in faces:
            if face["confidence"] > 0.96:
                arr.append(face["confidence"])
                # extract the bounding box from the requested face
                x1, y1, width, height = face['box']
                x2, y2 = x1 + width, y1 + height

                # extract the face
                face_boundary = frame[y1:y2, x1:x2]

                # resize pixels to the model size
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize(required_size)
                face_array = np.asarray(face_image)
                face_images.append(face_array)

        return face_images

def get_embedding(face,model):
     # convert into an array of samples
    sample = [np.asarray(face, 'float32')]
    # prepare the face for the model, e.g. center pixels
    sample = preprocess_input(sample, version=2)
    # perform prediction
    yhat = model.predict(sample)
    return yhat       


def is_match(i,j,a,b,show_faces,n_faces,embedings,thresh=0.4):
    # calculate distance between embeddings
    score = cosine(embedings[i][j], embedings[a][b])
    if show_faces==True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(n_faces[i][j])
        ax1.set_title('ID face')
        ax2.imshow(n_faces[a][b])
        ax2.set_title('Subject face')

    return score

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def funcion(row):
    #arr =[]
    # Load a video
    
    video = "resources/" + row[0]
    v_cap = cv2.VideoCapture(video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through video, taking a handful of frames to form a batch
    frames = []
    for i in tqdm(range(v_len)):

        # Load frame
        success = v_cap.grab()
        #if i % round(v_len/60) == 0:
        if i % 20 == 0:
            success, frame = v_cap.retrieve()
        else:
            continue
        if not success:
            continue

        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    # Detect faces in batch
    faces = [extract_face(x,detector) for x in frames]

    caras = [len(x) for x in faces]
    mode = statistics.mode(caras)
    n_faces = [x for x in faces if len(x) == mode]

    emb = [[get_embedding(y,model) for y in x] for x in n_faces]

    for i in range(1,len(emb)): # frame actual
        for j in range(len(emb[i])): # frame anteriorx  
            s = 0.4
            p = 0
            for k in range(len(emb[i-1])): # cara actual
                if is_match(i,j,i-1,k,False,n_faces,emb) < s:
                    s = is_match(i,j,i-1,k,False,n_faces,emb)
                    p = k
            n_faces[i][j],n_faces[i][p] = n_faces[i][p],n_faces[i][j] 
            emb[i][j], emb[i][p] = emb[i][p], emb[i][j]
    
    return (n_faces,row[1])


if __name__ == "__main__":
    df = pd.read_csv("multi_1.csv")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    start_time = time.perf_counter()
    result = pool.map(funcion,df.to_numpy())
    pool.close()
    pool.join()
    print(len(result))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result[-1])

    labels = []
    data = []

    for i in result:
        if (len(i[0][0]) > 0):
            data.append(i[0])
            labels.append(i[1])

    file = open('labelsM1','wb')

    pickle.dump(labels,file)

    file.close

    file = open('dataM1','wb')

    pickle.dump(data,file)

    file.close
    