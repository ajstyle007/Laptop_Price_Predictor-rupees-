import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

pipe = pickle.load(open("model.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

image = "image2.jpg"
image1 = "image1.jpg"
image2 = "workflow.png"
image3 = "workflow2.png"
image4 = "pipeline.png"
image5 = "image3.jpg"
image6 = "image4.jpg"

with st.sidebar:

    st.title(":violet[Estimate ur Laptop's Price] " ":computer:")
    #st.markdown("### :computer:")
    #st.markdown("""<h3><i class="bi bi-laptop-fill"></i> Estimate Your Laptop's Price</h3>""", unsafe_allow_html=True)
    st.image(image)
    st.image(image1)
    st.image(image5)
    st.image(image6)

st.markdown("<h1 style='text-align: center; color: yellow; font-weight: bold; font-size: 45px; '>Laptop Price Predictor</h1>",  unsafe_allow_html=True)
st.markdown("***")

st.markdown("<h3 style='text-align: center; color: #5C62D6; font-size: 15px; '>Choose the specifications below to receive an estimated laptop price</h3>",  unsafe_allow_html=True)



# Company	TypeName	Ram	Weight	Touchscreen	IPS	ppi	CPU Brand	HDD	SSD	Gpu brand	OS


col1, col2, col3 = st.columns(3)

with col1:
    company = st.selectbox("Laptop Brand", df["Company"].unique())
    cpu = st.selectbox("CPU", df["CPU Brand"].unique())
    hdd = st.selectbox('HDD(GB)',[0,128,256,512,1024,2048])
    ips = st.selectbox("IPS Panel", ["Yes", "No"])    
        

with col2:
    ram = st.selectbox("RAM(GB)", [2,4,6,8,12,16,24,32,64])
    gpu_company = st.selectbox("GPU", df["Gpu brand"].unique())
    ssd = st.selectbox('SSD(GB)',[0,8,128,256,512,1024])
    touchscreen = st.selectbox("Touch Screen", ["Yes", "No"])

with col3:
    typename = st.selectbox("Type", df["TypeName"].unique())
    os = st.selectbox('Operating System',df['OS'].unique())
    resolution = st.selectbox("Screen Resolution", ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440','3072x1920','1440x900','3840x2400','1600x900','2736x1824'])
    screen_size = st.number_input("Screen Size(Inch)", min_value=10.0, max_value=18.0, value=13.0)

weight = st.number_input("Weight(Kg)", min_value=0.5, max_value=5.0, value=2.0)


if st.button('Predict Laptop Price'):

    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
        
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,typename,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu_company,os], dtype=object)

    #array(['Company', 'TypeName', 'CPU_Frequency (GHz)', 'RAM (GB)',
    #   'GPU_Company', 'Weight (kg)', 'Touchscreen', 'IPS', 'X_res',
    #   'Y_res', 'ppi', 'CPU Brand', 'HDD', 'SSD', 'OS'], dtype=object)

    #query = query.reshape(1, -1)
    query = query.reshape(1,12)
    prediction = pipe.predict(query)
    # st.write(prediction)
    #st.header(pipe.predict(query))
    st.markdown("***")
    st.title("The predicted price: "+ str(int(np.exp(((pipe.predict(query)[0]))))) + " ₹")
    st.markdown("***")
    # st.header("In Rupees "  + str(93.2 * int(np.exp(pipe.predict(query)[0]))))
    # st.header(int(pipe.predict(query)[0]))
    # st.header(pipe.predict(query)[0])    

st.markdown("***")
st.markdown("<h1 style='text-align: center; color: green; font-weight: bold; font-size: 30px; '>Laptop Price Predictor Model WorkFlow</h1>",  unsafe_allow_html=True)
st.image(image3)

st.markdown("***")
st.markdown("<h1 style='text-align: center; color: red; font-weight: bold; font-size: 30px; '>PipeLine of Above Machine Learning Model</h1>",  unsafe_allow_html=True)
st.image(image4)
