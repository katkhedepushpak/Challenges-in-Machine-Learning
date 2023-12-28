### AUTHOR : PUSHPAK VIJAY KATKHEDE
### ASSIGNMENT 1: SWEESE CHEESE IN SPACE
### DATE : 01/21/2023

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cfhtlens.csv ')


# on preliminary checking the columns 3-7 does not have any missing values
print(df.describe())
print("\n \n ---------------------------------------------------------------------------------------")

print("Feature Name : PSF_e1")
plt.hist(df["PSF_e1"]) #drawing a histogram
plt.show()
print("No missing values") 
print("\n \n ---------------------------------------------------------------------------------------")



print("Feature Name : PSF_e2")
plt.hist(df["PSF_e2"])
plt.show() 
print("No missing values") 
print("\n \n ---------------------------------------------------------------------------------------")

print("Feature Name : scalelength")
plt.hist(df["scalelength"])
plt.show() 
print("No missing values") 
print("\n \n ---------------------------------------------------------------------------------------")

print("Feature Name : model_flux")
plt.hist(df["model_flux"])
plt.show() 
print("No missing values") 

Total_count = 0 # to count all missing values in the dataset

print("\n \n ---------------------------------------------------------------------------------------")
print("Feature Name : MAG_u")
plt.hist(df["MAG_u"])
plt.show() 
val_99  = df['MAG_u'].value_counts()[99] # counting the values that match 99 in column as there are no -99 present in dataset
print("Total missing values in MAG_u feature are :" , val_99)
Total_count = Total_count +val_99

print("\n \n ---------------------------------------------------------------------------------------")
print("Feature Name : MAG_g")
plt.hist(df["MAG_g"])
plt.show() 
val_99  = df['MAG_g'].value_counts()[99]
valn99 = df['MAG_g'].value_counts()[-99]
print("Total missing values in MAG_g feature are :" , val_99 + valn99)   # counting the values that match 99, -99
Total_count = Total_count +val_99 + valn99


print("\n \n ---------------------------------------------------------------------------------------")
print("Feature Name : MAG_r")
plt.hist(df["MAG_r"])
plt.show() 
val_99  = df['MAG_r'].value_counts()[99]
print("Total missing values in MAG_r feature are :" , val_99) # counting the values that match 99 in column as there are no -99 present in dataset
Total_count = Total_count +val_99
print("\n \n ---------------------------------------------------------------------------------------")

print("Feature Name : MAG_i")
plt.hist(df["MAG_i"])
plt.show() 
val_99  = df['MAG_i'].value_counts()[99]
valn99 = df['MAG_i'].value_counts()[-99]
print("Total missing values in MAG_i feature are :" , val_99 + valn99)  # counting the values that match 99, -99
Total_count = Total_count +val_99 + valn99
print("\n \n ---------------------------------------------------------------------------------------")

print("Feature Name : MAG_z")
plt.hist(df["MAG_z"])
plt.show() 
val_99  = df['MAG_z'].value_counts()[99]
print("Total missing values in MAG_z feature are :" , val_99) # counting the values that match 99 in column as there are no -99 present in dataset
Total_count = Total_count +val_99

# hence total missing values
print("All Missing values in whole dataset :", Total_count)




