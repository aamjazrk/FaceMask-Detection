import cv2
import numpy as np
#import face_recognition as face 
from keras.models import load_model
model=load_model("./model2-001.model")

labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the cascade xml file 
classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count=0
while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        if(labels_dict[label]== 'without mask'):
            count+=1

            # Convert into grayscale
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = classifier.detectMultiScale(gray, 1.1, 10)
            #Draw rectangle around the faces and crop the faces
            for (x, y, w, h) in faces:
                #cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
                faces = im[y:y + h, x:x + w]
                filtered_image2 = cv2.GaussianBlur(faces, (7, 7), 0) #decrease noise
                kernel = np.array([[0, -1, 0],
                                   [-1, 5,-1],
                                   [0, -1, 0]])
                image_sharp = cv2.filter2D(src=filtered_image2, ddepth=-1, kernel=kernel) #make picture more sharp
                file='/Users/hunteraek/Desktop/ComVision/face_mask_detection/ImageWarning/image'+str(count)+'.jpg'
                cv2.imwrite(file,image_sharp)

    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()