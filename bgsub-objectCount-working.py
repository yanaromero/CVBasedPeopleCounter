import numpy as np 
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np 
import csv 

# cap = cv.VideoCapture('vtest.avi')
# cap = cv.VideoCapture('http://admin:RWNjZV81MTU=@192.168.1.147:8080/stream/video/mjpeg')
cap = cv.VideoCapture('http://admin:RWNjZV81MTU=@172.55.47.162:8080/stream/video/mjpeg')
# cap = cv.VideoCapture(0)

#MOG Method
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

#MOG2 Method // very noisy
# fgbg = cv.createBackgroundSubtractorMOG2()

#GMG Method // not working??
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
# fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

#KNN method //also kinda noisy
# fgbg = cv.createBackgroundSubtractorKNN(detectShadows=False)


currentFrame = 0
while currentFrame <= 100:
	ret, frame = cap.read()
	if frame is None:
		break

	fgmask = fgbg.apply(frame)
	# fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

	cv.imshow('Frame', frame)
	cv.imshow('FG MASK Frame', fgmask)
	
	try:
	    if not os.path.exists('objectCount'):
	        os.makedirs('objectCount')
	except OSError:
	    print ('Error: Creating directory of objectCount')

	# Saves image of the current frame in jpg file
	if currentFrame%30 == 0:
		name = './objectCount/frame' + str(currentFrame) + '.jpg'
		print ('Creating...' + name)
		cv.imwrite(name, fgmask)

		# To stop duplicate images
	currentFrame += 1

	keyboard = cv.waitKey(1)
	if keyboard == 'q' or keyboard == 27:
		break

cap.release()
cv.destroyAllWindows()

Frame = 0
while Frame <= 100:
	original = cv.imread('./objectCount/frame' + str(Frame) +'.jpg', -1)

	original = cv.cvtColor(original, cv.COLOR_BGR2RGB)

	# Convert image in grayscale
	gray_im = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

	# Contrast adjusting with gamma correction y = 1.2
	gray_correct = np.array(255 * (gray_im / 255) ** 1.2 , dtype='uint8')

	# Dilatation et erosion
	kernel = np.ones((5,5), np.uint8)
	img_dilation = cv.dilate(gray_correct, kernel, iterations=1)
	img_erode = cv.erode(img_dilation,kernel, iterations=3)

	# clean all noise after dilatation and erosion
	img_erode = cv.medianBlur(img_erode, 7)

	# Labeling
	ret, labels = cv.connectedComponents(img_erode)
	label_hue = np.uint8(179 * labels / np.max(labels))
	blank_ch = 255 * np.ones_like(label_hue)
	labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
	labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
	labeled_img[label_hue == 0] = 0

	frame = "For frame" + str(Frame) + " the objects number is:"
	print(frame, ret-1)

	with open('objects.csv', 'a', newline='') as file:
	    writer = csv.writer(file)
	    writer.writerow([ret-1])

	Frame += 30