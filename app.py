import streamlit as st 
from PIL import Image
import tensorflow as tf 
from image_classifier import process_image, prediction_result
import time

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("วิเคราะห์ภาพใบมันสำปะหลัง")
st.write("แอปพลิเคชันสำหรับทำนายโรคที่แสดงอาการบนใบมันสำปะหลัง")
img = st.file_uploader("อัพโหลดภาพ", type=["jpeg", "jpg", "png"])

# Display Image
st.write("ภาพที่อัพโหลด")
try:
	img = Image.open(img)
	st.image(img)	# display the image
	img = process_image(img)


	# Prediction
	model = tf.keras.models.load_model(
		"models/cass_classifier.hdf5")
	prediction = prediction_result(model, img)


	# Progress Bar
	my_bar = st.progress(0)
	for percent_complete in range(100):
		time.sleep(0.05)
		my_bar.progress(percent_complete + 1)

	# Output
	st.write("# โรค : {}".format(prediction["class"]))
	st.write("โอกาส :", prediction["accuracy"],"%")
except AttributeError:
	st.write("ไม่ได้เลือกภาพใด")