import os
import cv2
import math
import time
import mediapipe as mp
import numpy as np


class HandDetector():
	def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
		self.mode = mode
		self.maxHands = maxHands
		self.trackCon = trackCon
		self.complexity = complexity
		self.detectionCon = detectionCon

		self.overlayList = []
		self.state_circle = 0
		self.folderPath = "Real"
		self.myList = os.listdir(self.folderPath)

		# For detecting hands
		self.mpHands = mp.solutions.hands
		self.mpDraw = mp.solutions.drawing_utils
		self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)

	def findHands(self, img, draw=True):
		# Required to resize the video from an external media
		# img = cv2.resize(img, (800, 400), interpolation=cv2.INTER_AREA)
		img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(img_RGB)
		# print(self.results.multi_hand_landmarks)

		if self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

		return img

	def findPosition(self, img, handNo=0, draw=True, pos=0):
		lmlist = []
		if self.results.multi_hand_landmarks:
			myHand = self.results.multi_hand_landmarks[handNo]
			for id, lm in enumerate(myHand.landmark):
				# print(id, lm)
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				lmlist.append([id, cx, cy])

			if draw:
				cv2.circle(img, (lmlist[pos][1], lmlist[pos][2]), 10, (255, 0, 255), cv2.FILLED)

		return lmlist, img

	def join(self, img, lmlist, pos1=8, pos2=12, draw=True):
		length =math.hypot((lmlist[pos1][1] - lmlist[pos2][1]), (lmlist[pos1][2] - lmlist[pos2][2]))
		state = 0
		if length < 25:
			if draw:
				cv2.circle(img, (lmlist[pos1][1], lmlist[pos1][2]), 5, (255, 0, 0), cv2.FILLED)
			state = 1

		return img, state

	def AddImgs(self, img, num=0, init_h=0, init_w=0):
		c1, c2, r = (760, 25), (810, 25), 20
		cv2.circle(img, c1, radius=r, color=(0, 255, 0), thickness=2)
		cv2.circle(img, c2, radius=r, color=(0, 0, 255), thickness=2)

		for imPath in self.myList:
			image = cv2.imread(f"{self.folderPath}/{imPath}")
			image = cv2.resize(image, (240, 200))
			self.overlayList.append(image)

		if len(self.overlayList) > num:
			h, w, c = self.overlayList[num].shape
			img[init_h:h+init_h, init_w:w+init_w] = self.overlayList[num]
		else:
			h, w, c = self.overlayList[0].shape
			img[init_h:h + init_h, init_w:w+init_w] = self.overlayList[0]

		return img

	def feed_check(self, num):

		arr = self.overlayList
		if not (num <= len(arr)):
			num %= len(arr)

		return num

	def circle_check(self, img, arr, lmlist):
		# assert len(arr) == 3
		c1, c2, r = (760, 25), (810, 25), 20

		img, state = self.join(img, lmlist)

		if ((arr[1] - c1[0]) ** 2) + ((arr[2] - c1[1]) ** 2) < r:
			if state:
				cv2.circle(img, c1, radius=r, color=(0, 255, 0), thickness=-1)
				self.state_circle -= 1 # Previous

		if ((arr[1] - c2[0]) ** 2) + ((arr[2] - c2[1]) ** 2) < r:
			if state:
				cv2.circle(img, c2, radius=r, color=(0, 0, 255), thickness=-1)
				self.state_circle += 1 # Forward

		return img

	def prev_next(self, img, init_h=0, init_w=0):

		num = self.feed_check(self.state_circle)
		h, w, c = self.overlayList[num].shape
		img[init_h:h + init_h, init_w:w + init_w] = self.overlayList[num]

		return img

def main():
	cap = cv2.VideoCapture(0)
	cap.set(3, 924)
	cap.set(4, 800)
	cap.set(10, 130)
	pTime, pTimeL = 0, 0

	detector = HandDetector()
	img_h, img_w = 42, 16

	while True:
		cTimeL = time.time()
		res, img = cap.read()
		img = cv2.flip(img, 1)
		cTime = time.time()

		img = detector.findHands(img, draw=False)

		img = detector.AddImgs(img, init_h=img_h, init_w=img_w)
		lmlist, img = detector.findPosition(img, pos=12, draw=False)
		if lmlist:
			img, state = detector.join(img, lmlist)
			img = detector.circle_check(img, lmlist[8], lmlist)
		img = detector.prev_next(img, img_h, img_w)

		# FPS
		if (cTime - pTime):
			fps = 1 / (cTime - pTime)
			pTime = cTime

			cv2.putText(img, f"FPS: {str(int(fps))}", org=(7, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
						fontScale=1, color=(0, 0, 0), thickness=1)

		# Latency
		if (cTimeL - pTimeL):
			latency = np.round((cTimeL - pTimeL), 4)
			pTimeL = cTimeL

			cv2.putText(img, f"Latency: {str(latency)}s", org=(97, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
						fontScale=1, color=(0, 0, 0), thickness=1)

		# img = detector.prev_next(img, lmlist)
		cv2.imshow("WebCam", img)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

if __name__ == "__main__":
	main()
