import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
from stqdm import stqdm

def fGGM(x, a, b, c, gamma):
    return (np.exp(-c * x) * (1 + (a * (-1 + np.exp(b * x)) * gamma) / b) ** (-((1 + gamma) / gamma)) * (c * (b - a * gamma) + a * np.exp(b * x) * (b + c * gamma))) / b

def FGGM(x, a, b, c, gamma):
    return np.exp(-c * x) / (1 + a * gamma * (np.exp(b * x) - 1) / b) ** (1 / gamma)

listofvalues = [[1955.8138888888889, 113.17808219178082], [1955.8138888888889, 110.6986301369863], [1956.9583333333333, 111.84657534246575], [1956.9583333333333, 107.8958904109589], [1958.2472222222223, 109.17808219178082], [1958.2472222222223, 108.56438356164384], [1958.7944444444445, 109.11506849315069], [1958.7944444444445, 109.04109589041096], [1959.7666666666667, 110.01369863013699], [1959.7666666666667, 109.43561643835616], [1959.8972222222221, 109.56712328767124], [1959.8972222222221, 108.46575342465754], [1964.9972222222223, 113.56438356164384], [1964.9972222222223, 109.63561643835617], [1965.1805555555557, 109.81643835616438], [1965.1805555555557, 109.66301369863014], [1965.5972222222222, 110.08219178082192], [1965.5972222222222, 109.57808219178082], [1966.025, 110.00821917808219], [1966.025, 109.57260273972602], [1968.2222222222222, 111.76712328767124], [1968.2222222222222, 111.17534246575343], [1968.4583333333333, 111.41369863013699], [1968.4583333333333, 110.65205479452055], [1969.3416666666667, 111.53150684931506], [1969.3416666666667, 111.23835616438356], [1970.0277777777778, 111.92876712328767], [1970.0277777777778, 109.4958904109589], [1973.1555555555556, 112.62465753424658], [1973.1555555555556, 111.63561643835617], [1973.6305555555555, 112.10684931506849], [1973.6305555555555, 110.44109589041096], [1973.8333333333333, 110.64383561643835], [1973.8333333333333, 110.59726027397261], [1975.4166666666667, 112.17808219178082], [1975.4166666666667, 111.81917808219178], [1976.875, 113.28219178082192], [1976.875, 110.38082191780822], [1977.9194444444445, 111.42465753424658], [1977.9194444444445, 111.33424657534246], [1978.3166666666666, 111.72876712328767], [1978.3166666666666, 111.03013698630137], [1981.0583333333334, 113.77534246575343], [1981.0583333333334, 112.05479452054794], [1981.1861111111111, 112.17808219178082], [1981.1861111111111, 111.53698630136986], [1982.8666666666666, 113.22191780821917], [1982.8666666666666, 112.5068493150685], [1983.7833333333333, 113.42191780821918], [1983.7833333333333, 113.26301369863013], [1985.125, 114.6082191780822], [1985.125, 113.53972602739726], [1986.8055555555557, 115.21643835616439], [1986.8055555555557, 113.3945205479452], [1987.0861111111112, 113.67945205479452], [1987.0861111111112, 113.67123287671232], [1987.9888888888888, 114.56986301369864], [1987.9888888888888, 114.21369863013699], [1988.0277777777778, 114.25479452054795], [1988.0277777777778, 112.88767123287671], [1997.5916666666667, 122.44931506849315], [1997.5916666666667, 116.93150684931507], [1998.2916666666667, 117.63013698630137], [1998.2916666666667, 117.55890410958904], [1999.9972222222223, 119.26575342465753], [1999.9972222222223, 114.14246575342466], [2000.8361111111112, 114.98630136986301], [2000.8361111111112, 114.52328767123288], [2001.4305555555557, 115.11506849315069], [2001.4305555555557, 114.37260273972603], [2002.213888888889, 115.15342465753425], [2002.213888888889, 114.33698630136986], [2002.4083333333333, 114.53150684931506], [2002.4083333333333, 114.26849315068493], [2002.638888888889, 114.5013698630137], [2002.638888888889, 113.58356164383562], [2002.9944444444445, 113.93972602739726], [2002.9944444444445, 113.76986301369863], [2003.7416666666666, 114.51780821917808], [2003.7416666666666, 114.37260273972603], [2003.8666666666666, 114.4986301369863], [2003.8666666666666, 114.2027397260274], [2004.411111111111, 114.74520547945205], [2004.411111111111, 114.7068493150685], [2006.6555555555556, 116.95068493150686], [2006.6555555555556, 116.03287671232877], [2006.9444444444443, 116.32328767123288], [2006.9444444444443, 115.30684931506849], [2007.0638888888889, 115.42739726027398], [2007.0638888888889, 114.17260273972603], [2007.075, 114.18356164383562], [2007.075, 114.06575342465753], [2007.6166666666666, 114.6054794520548], [2007.6166666666666, 114.31506849315069], [2008.9027777777778, 115.6027397260274], [2008.9027777777778, 115.21095890410959], [2009.0027777777777, 115.31232876712329], [2009.0027777777777, 114.74246575342465], [2009.6944444444443, 115.43287671232876], [2009.6944444444443, 114.33972602739726], [2010.3361111111112, 114.97808219178083], [2010.3361111111112, 114.20547945205479], [2010.8416666666667, 114.71506849315068], [2010.8416666666667, 114.32328767123288], [2011.4722222222222, 114.95068493150686], [2011.4722222222222, 114.81917808219178], [2012.925, 116.27397260273973], [2012.925, 115.66849315068494], [2012.9611111111112, 115.7041095890411], [2012.9611111111112, 115.66301369863014], [2013.4472222222223, 116.14794520547945], [2013.4472222222223, 115.27123287671233], [2015.25, 117.07397260273973], [2015.25, 116.74246575342465], [2015.263888888889, 116.75616438356164], [2015.263888888889, 115.87123287671233], [2015.4611111111112, 116.06849315068493], [2015.4611111111112, 115.94794520547946], [2016.3638888888888, 116.85205479452055], [2016.3638888888888, 116.45205479452055], [2017.2888888888888, 117.37534246575342], [2017.2888888888888, 117.0986301369863], [2017.7055555555555, 117.51780821917808], [2017.7055555555555, 117.11506849315069], [2018.3055555555557, 117.71232876712328], [2018.3055555555557, 116.96986301369863]]

peaks = np.array(listofvalues)[::2]
timespeakslows = np.column_stack((peaks, np.array(listofvalues)[1::2, 1]))

def llh(t, y, z, alpha, KK, b, c, gamma, kappa, CC, birthage):
    def integrand(u):
        return np.exp(-kappa * u) * FGGM(u - birthage, KK * np.exp(-alpha * (t - u + birthage - 2000)), b, c, gamma)
    integral, _ = quad(integrand, z, y, epsabs=1.48e-09, epsrel=1.48e-09)
    return np.log(fGGM(y - birthage, KK * np.exp(-alpha * (t - y + birthage - 2000)), b, c, gamma)) + kappa * (t - z + birthage) + np.log(CC) - CC * np.exp(kappa * (t + birthage)) * integral

def loglikelihood(vars_vals):
    alpha, KK, b, c, gamma = vars_vals[0], vars_vals[1], vars_vals[2], vars_vals[3], vars_vals[4]
    CC = 3.249*10**(-11)
    birthage = 60
    kappa = 0.02085
    return sum([llh(timespeakslows[i, 0], timespeakslows[i, 1], timespeakslows[i, 2], alpha, KK, b, c, gamma, kappa, CC, birthage) for i in range(timespeakslows.shape[0])]) + np.log(FGGM(timespeakslows[-1, 2] - birthage, KK * np.exp(-alpha * (timespeakslows[-1, 0] - timespeakslows[-1, 2] + birthage - 2000)), b, c, gamma))

def heatmap_plot(fixed_vars=[0,1,0,1,1], fixed_values=[3.7303227240317895e-06, 3.925085806440573e-06, 0.34537434850401993], var4=[0.1,0.25], var5=[0.1,0.25], num_points=100, cutoff=-10**9):
    
    var_list=["alpha","KK","b","c","gamma"]
    var_values = [0,0,0,0,0]
    
    running_vars=[]
    running_vars_index=[]
    
    count=0
    for i in range(5):
        if fixed_vars[i]==1:
            var_values[i]=fixed_values[count]
            count+=1
        if fixed_vars[i]==0:
            running_vars.append(var_list[i])
            running_vars_index.append(i)
            
    run1_range = np.linspace(var4[0], var4[1], num_points)
    run2_range = np.linspace(var5[0], var5[1], num_points)
    
    heatmap = np.zeros((len(run1_range), len(run2_range)))
    
    for i, run1 in stqdm(enumerate(run1_range)):
        for j, run2 in enumerate(run2_range):
            
            var_values[running_vars_index[0]]=run1
            var_values[running_vars_index[1]]=run2
            
            result = loglikelihood(var_values)
            heatmap[i, j] = result
            
    heatmap[heatmap < cutoff] = cutoff
            
    heatmap_img = plt.imshow(
        heatmap, extent=[var4[0], var4[1], var5[0], var5[1]], origin="lower", cmap="viridis"
    )
    plt.colorbar(label="Log Likelihood")
    plt.xlabel(running_vars[0])
    plt.ylabel(running_vars[1])

        
    return heatmap_img


import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


# Create a Streamlit app
st.title("Choose Exactly 3 Fixed Variables")

# Define the list of available variables
variables = ["alpha","KK","b","c","gamma"]

selected_variables=[]
# Use a while loop to ensure exactly 3 variables are selected
while len(selected_variables) != 3:
    selected_variables = st.multiselect("Select 3 Variables", variables, default=["KK","c","gamma"])

# Display the selected variables
st.success("You have selected exactly 3 variables: {}".format(", ".join(selected_variables)))

fixed_variables=[0,0,0,0,0]
fixed_variables_names=[]
running_variables_names=[]

if "alpha" in selected_variables:
    fixed_variables[0]=1
    fixed_variables_names.append("alpha")
else:
    running_variables_names.append("alpha")
if "KK" in selected_variables:
    fixed_variables[1]=1
    fixed_variables_names.append("KK")
else:
    running_variables_names.append("KK")
if "b" in selected_variables:
    fixed_variables[2]=1
    fixed_variables_names.append("b")
else:
    running_variables_names.append("b")
if "c" in selected_variables:
    fixed_variables[3]=1
    fixed_variables_names.append("c")
else:
    running_variables_names.append("c")
if "gamma" in selected_variables:
    fixed_variables[4]=1
    fixed_variables_names.append("gamma")
else:
    running_variables_names.append("gamma")
    
    
st.title("Enter 3 Numerical Values for the fixed parameters")

# Create input boxes for numerical values
value1 = st.number_input(fixed_variables_names[0], value=0.000000001734633, format="%g")
value2 = st.number_input(fixed_variables_names[1], value=0.0006087902,format="%g")
value3 = st.number_input(fixed_variables_names[2], value=0.5625251,format="%g")

st.title("Enter the range for the running parameters")

# Create input boxes for numerical values
value40 = st.number_input(running_variables_names[0]+" range start", value=0.1,format="%g")
value41 = st.number_input(running_variables_names[0]+" range end", value=0.3,format="%g")
value50 = st.number_input(running_variables_names[1]+" range start", value=0.1,format="%g")
value51 = st.number_input(running_variables_names[1]+" range end", value=0.3,format="%g")

st.title("Enter number of points and cutoff")

# Create input boxes for numerical values
num_points = st.number_input("number of points", value=100,format="%d")
cutoff = st.number_input("cutoff", value=-10**(9),format="%g")

var4=[value40,value41]
var5=[value50,value51]



if st.button('Enter'):
    st.write('Generating heatmap')
    heatmap_plot(fixed_vars=fixed_variables, fixed_values=[value1, value2, value3], var4=var4, var5=var5, num_points=num_points, cutoff=cutoff)
    st.pyplot()








    
