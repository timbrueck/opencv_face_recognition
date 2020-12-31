import cv2 as cv 


face_cascade = cv.CascadeClassifier('haarcascade.xml') 
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv.CascadeClassifier('haarcascade_mouth.xml')
cam = cv.VideoCapture(0)
img_counter = 0
font = cv.FONT_HERSHEY_SIMPLEX

print("Hello and welcome to face detection")
print("Press ESC to quit and SPACE for taking a screenshot")
print("Enjoy!")


try:

    while True:
        _, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        roi_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        fullbodies = face_cascade.detectMultiScale(gray, 1.1, 4)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
        mouths = eye_cascade.detectMultiScale(gray, 1.1, 4)


        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (200, 100, 0), 2)
            # cv.putText(img, 'Human', (x, y+1), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv.putText(img, 'human', (x + 6, y - 6), font, 1.0, (0, 255, 0), 2)
            print("Human detected")
    
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(img, (ex,ey), (ex+ew,ey+eh), (255, 255, 0), 2)
                print("Eye detected")
        
        # for (mx, my, mw, mh) in mouths:
        #     cv.rectangle(img, (mx, my), (mx+mw, my+mh), (200, 100, 0), 2)
        #     cv.putText(img, 'Mouth', (mx + 6, my - 6), font, 1.0, (0, 255, 0), 2)
        
        #     print("Mouth detected")
      

        cv.imshow('webcam', img)
    
        k = cv.waitKey(30) & 0xff
        if k==27:
            print("Escape key has been hit. Closing...")
            break
        elif k%256 == 32:
            print("Saving screenshot...")
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv.imwrite(img_name, img)
            print("{} written!".format(img_name))
            img_counter += 1


    cam.release()
    cv.destroyAllWindows()
except(Warning):
    print("An error has occured")