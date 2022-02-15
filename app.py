# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:11:55 2022

@author: Lando
"""


import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from newton_rhapson_weibull_estimation import nwt, get_ci


# Page Layout -- Full Width
st.set_page_config(page_title='Determining Paramaters of a Weibull distribution using Newton Rhapson estimation of the Maximul Likelihood',
                   layout='wide')


## Title and Sub Titles
st.title("Determining Paramaters of a Weibull Distribution Using Newton Rhapson")
st.write("**Created By: [Landon Mecham](https://www.linkedin.com/in/landonme/)**")
#st.header("Classification Edition")
st.write("This app uses the Newton Rhapson algorithm built from scratch to find the maximul likelihood estimators of a weibull distribution.")
st.write("The Newton Raphson algorithm finds the approximate root of a real-valued function using the idea that a continuous differentiable function can be approximated by a straight line tangent to it.")

## Uses Sidebar
# Input your csv
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV File](https://raw.githubusercontent.com/landonme/Newton-Raphson-and-GLRT/master/melanoma%20years%20till%20death%20or%20relapse.csv)
""")

# Main Panel
st.subheader('Data')

## final
def final_func(data):
    X = data.X     
    st.markdown('The sample dataset used is the number of years till death or reocurrence after being diagnosed with melanoma')
    st.write(data.head(5))
    st.write('\n')
    #data['X'].plot(kind="hist", title="Years Till Death or Reoccurence of Melanoma")
    st.write('\n')
    fw = lambda B: math.fsum(X**B*np.log(X))/math.fsum(X**B) - 1/B - math.fsum(np.log(X))/len(X)
    dfw = lambda B: math.fsum(np.log(X)**2*X**B)/math.fsum(X**B) + 1/B**2 + 1
    B, n = nwt(fw,dfw,2 ,1e-8, 100)
    A = math.fsum(X**B/len(X))**(1/B)
    B_list, A_list = get_ci(X, B, A)
    B_list = [round(x, 2) for x in B_list]
    A_list = [round(x, 2) for x in A_list]
    st.write('\n')
    ## Print out MLEs with Confidence Intervals.
    st.info('Finding equation roots via Newton Rhapson...')
    st.write(f'A solution was found after {n} iterations.')
    st.write(f"The maximum likelihood estimate of Beta is {B_list[1]}, with the following 95% confidence interval: [{B_list[0]}, {B_list[2]}]")
    st.write(f"The maximum likelihood estimate of Theta is {A_list[1]}, with the following 95% confidence interval: [{A_list[0]}, {A_list[2]}]")
    fig, ax = plt.subplots()
    ax.hist(data.X, bins=20)
    st.pyplot(fig)


# Run Newton Rhapson based off provided dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    final_func(data=data)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        data = pd.read_csv(
            'https://raw.githubusercontent.com/landonme/Newton-Raphson-and-GLRT/master/melanoma%20years%20till%20death%20or%20relapse.csv')
        final_func(data=data)
        
        