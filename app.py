import cv2
import time
import numpy as np
from HandBackend import HandDetector
from flask import Flask, render_template, Response

app = Flask(__name__)

cap = cv2.VideoCapture(0)
# cap.set(3, 924)
# cap.set(4, 800)
cap.set(10, 150)

def FPS(img, fps, latency):
	cv2.putText(img, f"FPS: {str(int(fps))}", org=(7, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
				fontScale=1, color=(0, 0, 0), thickness=1)

	cv2.putText(img, f"Latency: {str(latency)}s", org=(97, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
				fontScale=1, color=(0, 0, 0), thickness=1)

	return img

def gen_frames():
	pTime, pTimeL = 0, 0
	previous = time.time()
	delta = 0
	message = ""
	a = 0
	while True:
		_, img = cap.read()
		img = cv2.flip(img, 1)

		# # FPS
		cTimeL = time.time()
		#
		# detector = HandDetector()
		# img_h, img_w = 42, 16
		#
		# img = detector.findHands(img, draw=False)
		#
		# img = detector.AddImgs(img, init_h=img_h, init_w=img_w)
		# lmlist, img = detector.findPosition(img, pos=12, draw=False)
		# if lmlist:
		# 	img, state = detector.join(img, lmlist)
		# 	img = detector.circle_check(img, lmlist[8], lmlist)
		# img = detector.prev_next(img, img_h, img_w)
		#
		cTime = time.time()
		if (cTime - pTime) != 0:
			fps = 1 / (cTime - pTime)
			latency = np.round((cTimeL - pTimeL), 4)
			pTime, pTimeL = cTime, cTimeL
			a += 1

			img = FPS(img, fps, latency)

		# Video stream
		ret, buffer = cv2.imencode('.jpg', img)
		img = buffer.tobytes()
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
	app.run(debug=True)
