import pandas as pd
import numpy as np
import joblib as jb
import streamlit as st

model = jb.load(open(r'C:\Users\User\Desktop\Jupyter project\model.joblib', 'rb'))

st.header('Car Price Prediction Model')

cars_data = pd.read_csv('C:/Users/User/Desktop/dataset.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip(' ')
cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994,2025)
engine = st.slider('Engine CC', 700,5000)
cylinders = st.slider('Cylinders', 1,12)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
mileage = st.slider('Car Mileage', 10,40)
#transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
doors = st.slider('No of Doors', 2,6)
drivetrain = st.slider('DriveTrain', 1,4)


if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[name,year,engine,cylinders,fuel,mileage,doors,drivetrain]],
    columns=['name','year','engine','cylinders','fuel','mileage','doors','drivetrain'])
    
    input_data_model['name'].replace(['Jeep', 'GMC', 'Dodge', 'RAM', 'Nissan', 'Ford', 'Hyundai',
       'Volkswagen', 'Chevrolet', 'Kia', 'Mazda', 'Acura', 'BMW',
       'Toyota', 'Buick', 'Audi', 'Mercedes-Benz', 'Honda', 'Lincoln',
       'Chrysler', 'Cadillac', 'INFINITI', 'Lexus', 'Subaru', 'Land',
       'Volvo'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],inplace=True)
    input_data_model['fuel'].replace(['Gasoline', 'Diesel', 'Hybrid', 'E85', 'PHEV', 'Electric'],[1,2,3,4,5,6], inplace=True)
    #input_data_model['transmission'].replace(['8-Speed', 'Automatic', '6-Speed', '10-Speed', '7-Speed',
       #'9-Speed', '8-speed', 'CVT', 'Variable', '9', 'Aisin', '6-Spd',
       #'9-speed', '6-SPEED', 'automatic', 'A/T', '(CVT)', '8'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],inplace=True)
    

    car_price = model.predict(input_data_model)

    st.markdown('Car Price is going to be '+ str(car_price[0]))

