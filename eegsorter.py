import cv2
import customtkinter as ck
from PIL import Image, ImageTk
import pickle
import cvzone
from cvzone.ClassificationModule import Classifier
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture("test.mp4")
model = tf.keras.models.load_model('./model_cnn.h5')
labels = ["NULL", "INFERTIL", "FERTIL"]

width, height = 235, 180
posList = [(80,65),(80,245),(315,65),(315,245)]

def pre_processing(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def show_frame():
   success, img2 = cap.read()
   if success:
   		img = cv2.resize(img2, (640,480))
   		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   		imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
   		imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
   		imgMedian = cv2.medianBlur(imgThreshold, 3)
   		kernel = np.ones((3, 3), np.int8)
   		imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

   		machineLearning(imgDilate, img)
   		
   		frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   		image = Image.fromarray(frame)
   		photo = ImageTk.PhotoImage(image=image)
   		label.photo = photo
   		label.configure(image=photo)  
   root.after(20, show_frame)

def machineLearning(imgPro, img):
	for pos in posList:
		x,y = pos
		imgCrop = imgPro[y:y+height,x:x+width]
		imgCrop3  = img[y:y+height,x:x+width]
		imgCrop2 = np.asarray(imgCrop3)
		imgCrop2 = cv2.resize(imgCrop2, (32,32))
		imgCrop2 = pre_processing(imgCrop2)
		imgCrop2 = imgCrop2.reshape(1,32,32,3)

		#predict
		class_index = np.argmax(model.predict(imgCrop2))
		prob_val = np.amax(model.predict(imgCrop2))
		#print(class_index, prob_val)

		cv2.putText(imgCrop3, str(labels[class_index]),(10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1)

		contours, _ = cv2.findContours(imgCrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		objects_contours = []
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 1000:
				objects_contours.append(cnt)
		for cnt2 in objects_contours:
			rect = cv2.minAreaRect(cnt2)
			(a, b), (w, h), angle = rect

			if h > w:
				Rasio = (w/h)*100
				Volume = (0.5163*(h/3.708)*((w/3.708)**2))/1000
				Mass = 1.032*Volume
			if w > h:
				Rasio = (h/w)*100
				Volume = (0.5163*(h/3.708)*((w/3.708)**2))/1000
				Mass = 1.032*Volume

			if Rasio < 69:
				Note = "Lonjong"
			if 69 <= Rasio <= 77:
				Note = "Normal"
			if Rasio > 77:
				Note = "Bulat"

			cvzone.putTextRect(img, "R = {}".format(round(Rasio,2)), (x+1,y+height-3), scale=1, thickness=1, offset=2, colorR=(0,255,0), colorT=(0,0,0))
			cvzone.putTextRect(img, "M = {} g".format(round(Mass,2)), (x+1,y+height-20), scale=1, thickness=1, offset=2, colorR=(0,255,0), colorT=(0,0,0))
			#cvzone.putTextRect(img, Note, (x+1,y+height-20), scale=1, thickness=1, offset=2, colorR=(0,255,0), colorT=(0,0,0))
		
		#cv2.imshow(str(x+y), imgCrop)
		cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),(0,255,0),2)

cap = cv2.VideoCapture("test.mp4")

ck.set_appearance_mode("dark")
ck.set_default_color_theme("dark-blue")

root = ck.CTk()
root.title("Egg Sortir App")
root.iconbitmap("img/logo.png")
root.geometry("820x510")

frame1 = ck.CTkFrame(master=root, height=480, width=640)
frame1.pack(pady=0, padx=0)

button = ck.CTkButton(master=frame1, text="Upload Data")
button.grid(row=1, column=1, pady=10, padx=10)

label = ck.CTkLabel(master=frame1, text="")
label.grid(row=1, column=0, pady=5, padx=5)

show_frame()
root.mainloop()
cap.release()
