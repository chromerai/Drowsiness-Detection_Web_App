from scipy.spatial import distance
import streamlit as st
import dlib
import cv2
import imutils
import numpy as np
import tempfile
import time
from PIL import Image
import tensorflow as tf
from imutils import face_utils

import winsound
frequency = 2500
duration = 1000

new_frame_time = 0;
prev_frame_time = 0;

flag = 0
counter = 0

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

img_size = 224
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

st.title('Drowsiness detection')

DEMO_IMAGE = 'sad_woman.jpg'
DEMO_VIDEO = 'demo1.mp4'

model = tf.keras.models.load_model('my_model.h5')

st.markdown(

	"""
	<style>
	[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
		width: 350px
	}

	[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
		width: 350px
		margin-left=-350px
	}
	</style>


	""",

	unsafe_allow_html = True,
	)

st.sidebar.title('Sidebar')
st.sidebar.subheader('parameters')

@st.cache()
def image_resize(image,width=None, heigth = None, inter= cv2.INTER_AREA):
	dim = None
	(h,w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = width/float(w)
		dim = (int(w*r), height)

	else:
		r = width/float(w)
		dim = (width, int(h*r))

	# resize image
	resized = cv2.resize(image,dim, interpolation = inter)

	return resized



app_mode = st.sidebar.selectbox("Choose the App mode",["About Me","Run on Image","Run on Video"])

if app_mode == "About Me":
	st.header('Prevents sleep deprivation road accidents, by alerting drowsy drivers.')
	st.image('ISHN0619_C3_pic.jpg')
	st.markdown('In accordance with the survey taken by the Times Of India, about 40 % of road'
                ' accidents are caused '
                'due to sleep deprivation & fatigued drivers. In order to address this issue, this app will'
                ' alert such drivers with the help of deep learning models and computer vision.'
                '', unsafe_allow_html=True)
	st.markdown('This app is developed as part of **FINAL YEAR PROJECT**'
				'This app is developed by Team of three students under the guidance of **Dr. Sudarsan Sahoo**.'
				'<p> </p>'
				'Students : **Kaustubh Mishra**, **Yuvraj Kumar**, **Sudhanshu Vidyarthi**'
				'<p></p>'
				'<b> Scholar ID : **1816055**, **1816052**, **1816057** </b>'
				'', unsafe_allow_html=True)
	st.markdown('''
				Kaustubh Mishra Profile \n
				[Linkedln](https://www.linkedin.com/in/kaustubh-mishra-54556917b/)\n
				For any errors or suggestions kindly maul me: [email](kaustubhmishra983@gmail.com)\n

				Yuvraj Kumar Profile \n
				[Linkedln](https://www.linkedin.com/in/yuvraj-kumar-68164117a/)\n

				Sudhanshu Vidyarthi Profile \n
				[Linkedln](https://www.linkedin.com/in/sudhanshuvidyarthi/)



				''')
	st.markdown(

	"""
	<style>
	[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
		width: 350px
	}

	[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
		width: 350px
		margin-left=-350px
	}
	</style>


	""",

	unsafe_allow_html = True,
	)
	#st.video('link')

elif app_mode == 'Run on Image':
	img_file_buffer = st.sidebar.file_uploader("Upload an image", type =['jpeg','png','jpg'])
	if img_file_buffer is not None:
		image = np.array(Image.open(img_file_buffer))

	else:
		demo_image = DEMO_IMAGE
		image = np.array(Image.open(demo_image))

	st.sidebar.text('Original Image')
	st.sidebar.image(image)

	img_size = 224
	facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	eyecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	eye = eyecascade.detectMultiScale(gray,1.1,4)
	for (x,y,w,h) in eye:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0), 2)

	st.subheader('Locating the eye region')
	st.markdown('---')
	st.image(image)

	for x,y,w,h in eye:
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eyess = eyecascade.detectMultiScale(roi_gray)
		if len(eyess)==0:
			st.text('eyes not detected')
		else:
			for (ex,ey,ew,eh) in eyess:
				eyes_roi = roi_color[ey:ey+eh,ex:ex+ew]

	st.subheader('Extracted eye region')
	st.markdown('---')
	st.image(eyes_roi)

	final_image = cv2.resize(eyes_roi,(img_size,img_size))
	final_image = np.expand_dims(final_image,axis = 0)
	final_image = final_image/255.0

	st.subheader('Prediction')
	prediction = model.predict(final_image)
	if prediction > 0:
		st.write('Open eyes')
	else:
		st.write('Closed eyes')


elif app_mode == "Run on Video":
	st.set_option('deprecation.showfileUploaderEncoding', False)

	use_webcam  = st.sidebar.button('Use webcam')

	record = st.sidebar.checkbox('Record video')

	if record:
		st.checkbox('Recording', value = True)

	st.markdown(

	"""
	<style>
	[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
		width: 350px
	}

	[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
		width: 350px
		margin-left=-350px
	}
	</style>


	""",

	unsafe_allow_html = True,
	)

	stframe = st.empty()
	video_file_buffer = st.sidebar.file_uploader('Upload a video', type = ['mp4','mov','avi','asf','m4v'])
	tffile = tempfile.NamedTemporaryFile(delete = False)

	##video here
	if not video_file_buffer:
		if use_webcam:
			vid = cv2.VideoCapture(0)
		else:
			vid = cv2.VideoCapture(DEMO_VIDEO)
			tffile.name = DEMO_VIDEO
	else:
		tffile.write(video_file_buffer.read())
		vid = cv2.videoCapture(tffile.name)

	width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps_input = int(vid.get(cv2.CAP_PROP_FPS))

	#Recording part 
	codec = cv2.VideoWriter_fourcc('M','J','P','G')
	out = cv2.VideoWriter('output.mp4',codec,fps_input, (width, height))

	st.sidebar.text('Input Video')
	st.sidebar.video(tffile.name)

	fps = 0
	i = 0
	c1,c2 = st.beta_columns(2)

	with c1:
		st.markdown('**FRAME RATE**')
		c1_text = st.markdown('0')

	with c2:
		st.markdown('**IMAGE WIDTH**')
		c2_text = st.markdown('0')

	while vid.isOpened():
		i+=1
		ret,frame = vid.read()

		if not ret:
			continue

		eyecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		eye = eyecascade.detectMultiScale(gray,1.1,4)
		for (x,y,w,h) in eye:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)

		for x,y,w,h in eye:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]
			eyess = eyecascade.detectMultiScale(roi_gray)
			if len(eyess)==0:
				cv2.putText(frame,'eye_not_detected',(45,45),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)
			else:
				for (ex,ey,ew,eh) in eyess:
					eyes_roi = roi_color[ey:ey+eh,ex:ex+ew]

		final_image = cv2.resize(eyes_roi,(img_size,img_size))
		final_image = np.expand_dims(final_image,axis = 0)
		final_image = final_image/255.0

		Predictions = model.predict(final_image)
		if(Predictions>0):
			status = 'Open Eyes'
			cv2.putText(frame,status,(150,150),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)

			x1,y1,w1,h1 = 0,0,175,75
			cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
			cv2.putText(frame,'Active',(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

			subjects = detect(gray, 0)
			for subject in subjects:
				shape = predict(gray, subject)
				shape = face_utils.shape_to_np(shape)
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
				if ear < thresh:
					flag += 1
					if flag >= frame_check:
						cv2.putText(frame, "****************ALERT!****************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						cv2.putText(frame, "****************ALERT!****************", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						winsound.Beep(2000,500)
				else:
					flag = 0
		else:
			counter = counter + 1
			status = 'Closed Eyes'
			cv2.putText(frame,status,(150,150),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
			if counter > 5:
				x1,y1,w1,h1 = 0,0,175,75
				cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,0,0),-1)
				cv2.putText(frame, "********SLEEP ALERT!**********", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				winsound.Beep(frequency,duration)
				counter = 0

		new_frame_time = time.time()

		fps = 1/(new_frame_time-prev_frame_time)
		fps = int(fps)
		prev_frame_time = new_frame_time

		if record:
			out.write(frame)

		c1_text.write(f"<h1 style = 'text-align: center; color : red;'>{int(fps)}</h1>", unsafe_allow_html = True)
		c2_text.write(f"<h1 style = 'text-align: center; color : red;'>{width}</h1>", unsafe_allow_html = True)

		frame = cv2.resize(frame,(0,0),fx = 0.8, fy = 0.8)
		frame = image_resize(image = frame, width = 648)
		stframe.image(frame, channels = 'BGR',use_column_width = True )