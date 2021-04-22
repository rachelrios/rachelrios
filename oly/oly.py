# -*- coding: utf-8 -*-

### Define Libraries and other items ###
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from functools import reduce

np.seterr(divide='ignore', invalid='ignore') #supress warnings

'''
Function that defines column "Medal" based on Medal column
@params: Takes in each row as parameters
@returns: A 1 or 0 depending if they won a medal or not
'''
def label_note(row,medal):
   if row['Medal'] == medal :
      return 1
   return 0

#Data Prep part 1
print("We will load the Excel  data into Pandas dataframe") #We load the data into a df
df = pd.read_csv('/content/athlete_events.csv') #Read file and assign sheet
df = df.drop(["Team"], axis=1) #drop extra columns in pandas
df = df.dropna() #Delete bad data with NANS
index_names = df[ df['Season'] == "Winter" ].index #drop winter olympics
df.drop(index_names, inplace = True) #Drop winter folks
print("DF with columns dropped") #print new df without noise
print (df) #print df
df["Gold"] = df.apply (lambda row : label_note(row,"Gold"), axis=1) #Gold winner
df["Silver"] = df.apply (lambda row : label_note(row,"Silver"), axis=1) #Silver winner
df["Bronze"] = df.apply (lambda row : label_note(row,"Bronze"), axis=1) #Bronze winner
df["Sum"] = df["Gold"] + df["Silver"] + df["Bronze"] #Sum of all to determine if they should stay or go
index_names = df[ df['Sum'] == 0 ].index #drop non medal winners
df.drop(index_names, inplace = True) #Drop non medal winners
print(df) #Print Updated dataframe
#Split by sport
swimming = df[df['Sport'] == 'Swimming']#Swim split
gym = df[df['Sport'] == 'Rhythmic Gymnastics'] #Gym Split
tennis = df[df['Sport'] == 'Tennis'] #Tennis Split
df = swimming.append([gym, tennis]) #Drop all other sports

print("Analysis based on gender and age")
# Make a separate df for each gender

x1 = df[df['Sex'] == 'M'] #split by gender for  chart stats
x2 = df[df['Sex'] == 'F'] #split by gender for chart stats
# Assign colors for each gender and the names
colors = ['#E69F00', '#56B4E9'] #nice color definitions
names = ['Male', 'Female'] #assign legend
plt.hist([x1["Age"], x2["Age"]], bins = int(180/15), color = colors, label=names) #plot
# Plot formatting
plt.legend() #add a lengend
plt.xlabel('Age') #add age
plt.ylabel('Total # Athletes since 1992') #add Y label
plt.title('Historical Age Distribution since 1992 Swimming, Tennis, Rhythmic Gymnastics') #add title

print("Analysis based on gender and weight")
# Make a separate df for each gender
x1 = df[df['Sex'] == 'M'] #split by gender for  chart stats
x2 = df[df['Sex'] == 'F'] #split by gender for chart stats
# Assign colors for each gender and the names
colors = ['#E69F00', '#56B4E9'] #nice color definitions
names = ['Male', 'Female'] #assign legend
plt.hist([x1["Weight"], x2["Weight"]], bins = int(180/15), color = colors, label=names) #plot
# Plot formatting
plt.legend() #add a lengend
plt.xlabel('Weight') #add weight
plt.ylabel('Total # Athletes since 1992') #add Y label
plt.title('Historical Weight Distribution since 1992 Swimming, Tennis, Rhythmic Gymnastics') #add title

print("Swimming")
print("We continue with stats, this time we will split by sport and top medalist country")
print("Note: The data-frame at this point only contains medal winners therefore, a simple sum of countries will determine top medalists")
swimming_stats = swimming.drop(["ID","Silver","Gold","Bronze","Sum","Year"] ,axis=1) #drop non stat columns, we only want athletes' info
print(swimming_stats.describe()) #describe the data for swimming
s = swimming["NOC"].value_counts() #determine top swimming by counting country
s2 = s.head(10) #Extract top 10
s2 = s2.sort_values() #Sort top 10 countries
names = s2.index.tolist() #extract names
values = s2.values.tolist() #extract values
top_swimming = names #Define list for future reference in NB
plt.barh(names, values, label=names) #plot
# Plot formatting
plt.xlabel(' Total #  Medal Count') #add medal count
plt.ylabel('Country') #add country
plt.title('Top 10 Medalist Countries in Swimming Since 1992') #add title

print("Tennis")
print("We continue with stats, this time we will split by sport and top medalist country")
print("Note: The data-frame at this point only contains medal winners therefore, a simple sum of countries will determine top medalists")
tennis_stats = tennis.drop(["ID","Silver","Gold","Bronze","Sum","Year"] ,axis=1) #drop non stat columns, we only want athletes' info
print(tennis_stats.describe()) #describe the data for tennis
t = tennis["NOC"].value_counts() #determine top tennis by counting country
t2 = t.head(10) #Extract top 10
t2 = t2.sort_values() #Sort top 10 countries
names = t2.index.tolist() #extract names
values = t2.values.tolist() #extract values
top_tennis = names #Define list for future reference in NB
plt.barh(names, values, label=names) #plot
# Plot formatting
plt.xlabel(' Total #  Medal Count') #add medal count
plt.ylabel('Country') #add country
plt.title('Top 10 Medalist Countries in Tennis Since 1992') #add title

print("Rhythmic Gymnastics")
print("We continue with stats, this time we will split by sport and top medalist country")
print("Note: The data-frame at this point only contains medal winners therefore, a simple sum of countries will determine top medalists")
gym_stats = gym.drop(["ID","Silver","Gold","Bronze","Sum","Year"] ,axis=1) #drop non stat columns, we only want athletes' info
print(gym_stats.describe()) #describe the data for gym
g = gym["NOC"].value_counts() #determine top gym by counting country
g2 = g.head(10) #Extract top 10
g2 = g2.sort_values() #Sort top 10 countries
names = g2.index.tolist() #extract names
values = g2.values.tolist() #extract values
top_gym = names #Define list for future reference in NB
plt.barh(names, values, label=names) #plot
# Plot format
plt.xlabel(' Total #  Medal Count') #add medal count
plt.ylabel('Country') #add country
plt.title('Top 10 Medalist Countries in Rhythmic Gymnastics Since 1992') #add title

#We calculate more stats for all countries now
pd.set_option('display.max_columns', None)
print("Gymnastics") #info on sport
statsDF = gym  #define sport
statsDFgym = statsDF.drop(["ID","Age","Height","Weight","Year"], axis=1) #Drop non country attributes
print(statsDFgym.groupby('NOC').describe()) #describe each group / country

#We calculate more stats for all countries now
pd.set_option('display.max_columns', None)

print("Swimming") #info on sport
statsDFswim = swimming #info on sport
statsDFswim = statsDFswim.drop(["ID","Age","Height","Weight","Year"], axis=1)#Drop non country attributes
print(statsDFswim.groupby('NOC').describe())#describe each group / country

#tennis
print("Tennis") #info
pd.set_option('display.max_columns', None)

statsDFtennis = tennis #define DF
statsDFtennis = statsDFtennis.drop(["ID","Age","Height","Weight","Year"], axis=1) #drop non country attributes
print(statsDFtennis.groupby('NOC').describe()) #describe info

print("We will now run NB and predict the outcome of each sport based on historical data")

print("Probability for each  medal for each country in Rhythmic Gymnastics")
newDict = {} #Create a new dictionary top store values and parse
for i in gym['NOC']: #Iterate through each country
  try: #Try catch in case they do not get any medals
    country = i #Each country is equal to i
    data =  gym[ gym['NOC'] == i ] #Data is the item at given country
    X = data[["Gold", "Silver", "Bronze",]].values #Medal wins
    Y = data['Medal'].values #Medal values / outcome
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.5) #Split data set on 50/50
    NB_classifier = GaussianNB().fit(X_train,Y_train) #Set the classifier as Gaussian NB on train X and train Y
    prediction = NB_classifier.predict(X_test) #Predict values based on X TEST
    error_rate = np.mean(prediction!=Y_test) # Calculate Error Rate
    unique, counts = np.unique(prediction, return_counts=True) #Calculate predictions and store as dictionary
    counts = (counts/ sum(counts))*100 #Count division for %
    newDict[country] = dict(zip(unique,counts)) #Append for each country medal wins
  except Exception as x: #No medals won
    print(country,"  gets no medals") #Print no medals won
# Extracting specific keys from dictionary
for i in top_gym: #Top countries array defined couple of problems above
  print( {key: newDict[key] for key in newDict.keys() & {i}}) #Extract top ones

print("Probability for each  medal for each country in Tennis")
newDict = {} #Create a new dictionary top store values and parse
for i in tennis['NOC']: #Iterate through each country
  try: #Try catch in case they do not get any medals
    country = i #Each country is equal to i
    data =  tennis[ tennis['NOC'] == i ] #Data is the item at given country
    X = data[["Gold", "Silver", "Bronze",]].values #Medal wins
    Y = data['Medal'].values #Medal values / outcome
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.5) #Split data set on 50/50
    NB_classifier = GaussianNB().fit(X_train,Y_train) #Set the classifier as Gaussian NB on train X and train Y
    prediction = NB_classifier.predict(X_test) #Predict values based on X TEST
    error_rate = np.mean(prediction!=Y_test) # Calculate Error Rate
    unique, counts = np.unique(prediction, return_counts=True) #Calculate predictions and store as dictionary
    counts = (counts/ sum(counts))*100 #Count division for %
    newDict[country] = dict(zip(unique,counts)) #Append for each country medal wins
  except Exception as x: #No medals won
    print(country,"  gets no medals") #Print no medals won
# Extracting specific keys from dictionary
for i in top_tennis: #Top countries array defined couple of problems above
  print( {key: newDict[key] for key in newDict.keys() & {i}}) #Extract top ones

print("Probability for each  medal for each country in swimming")
newDict = {} #Create a new dictionary top store values and parse
top = []
for i in swimming['NOC']: #Iterate through each country
  try: #Try catch in case they do not get any medals
    country = i #Each country is equal to i
    data =  swimming[ swimming['NOC'] == i ] #Data is the item at given country
    X = data[["Gold", "Silver", "Bronze",]].values #Medal wins
    Y = data['Medal'].values #Medal values / outcome
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.5) #Split data set on 50/50
    NB_classifier = GaussianNB().fit(X_train,Y_train) #Set the classifier as Gaussian NB on train X and train Y
    prediction = NB_classifier.predict(X_test) #Predict values based on X TEST
    error_rate = np.mean(prediction!=Y_test) # Calculate Error Rate
    unique, counts = np.unique(prediction, return_counts=True) #Calculate predictions and store as dictionary
    counts = (counts/ sum(counts))*100 #Count division for %
    newDict[country] = dict(zip(unique,counts)) #Append for each country medal wins
  except Exception as x: #No medals won
    print(country,"  gets no medals") #Print no medals won
# Extracting specific keys from dictionary
for i in top_swimming: #Top countries array defined couple of problems above
  print( {key: newDict[key] for key in newDict.keys() & {i}}) #Extract top ones
