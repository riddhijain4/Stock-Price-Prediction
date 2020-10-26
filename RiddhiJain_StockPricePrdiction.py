import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from tkinter import *

#reading_the_dataset
a=pd.read_csv('dataset.csv')

#converting_the_dates_into_uniform_format
a['DATE']=pd.to_datetime(a['DATE']).dt.strftime('%d/%m/%y')
a['date']=a['DATE'].index
pd.set_option('display.max_rows',None,'display.max_columns',None)
#print(a)

#checking_labels
labels=a.columns.values
#print(labels)
a_new=a.rename(columns={'CLOSE ':'CLOSE'})

#taking_care_of_missing_values
def imputer(column):
    b=column.to_numpy().reshape(-1,1)
    imputer=SimpleImputer(missing_values=0,strategy='mean')
    imputer=imputer.fit(b)
    column=imputer.transform(b)
    return column
for i in a_new:
    for j in a_new[i]:
        if j==0:
            a_new[i]=imputer(a[i])
    d=a_new[i].notna()
    for k in d:
        if k==False:
            a_new[i]=imputer(a[i])

#classifying_dataset_into_independent_and_dependent_variables
x=a_new.drop(['HIGH','LOW','CLOSE','VOLUME','DATE'],axis='columns')
y=a_new.drop(['DATE','OPEN','date','VOLUME'],axis='columns')

#splitting_the_dataset_into_training_and_testing_data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#sorting_testing_data_by_index
x_test=x_test.sort_index()
y_test=y_test.sort_index()


#training_and_testing_data
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pr=lr.predict(x_test)

#checking_the_accuracy_of_the_model
accuracy=metrics.r2_score(y_test,y_pr)
print('The accuracy of this model=', accuracy)


#plotting_graph

xtest=[]
g=list(a_new['date'])
k=list(a_new['DATE'])
for i in x_test['date']:
    for j in g:
        if i==j:
            h=g.index(j)
            xtest.append(k[h])
xtest2=xtest[::75]
y_pr0=pd.DataFrame(y_pr)

def graph(i,s):
    i=int(i)
    q[i].plot(xtest,y_test[s],'r-',label='Actual',linewidth=1)
    q[i].plot(xtest,y_pr0[i],'b-', label='Predicted',linewidth=1)
    q[i].set_title(s+' PRICE VS DATE')
    q[i].set_ylabel(s+' PRICE')
    q[i].set_xlabel('DATE')
    q[i].set_xticks(xtest2)
    q[i].set_xticklabels(xtest2,rotation=30)
    q[i].legend()
    
fig,q=plt.subplots(3,figsize=(25,50))
plt.gcf().canvas.set_window_title('Actual and Predicted Prices vs date')
graph(0,'HIGH')
graph(1,'LOW')
graph(2,'CLOSE')
plt.tight_layout(pad=45.0)
plt.show()

#creating_display
def a1():
    l=int(e1.get())+6244
    m=int(e2.get())
    n=list(lr.predict([[m,l]]))
    print('High price =',n[0][0])
    print('Low price =',n[0][1])
    print('Close price =',n[0][2])
def a2():
    r.destroy()

def a2():
    r.destroy()
    
r=Tk()
r.title('STOCK PRICE PREDICTION')
r.geometry('500x200')
l=Label(text='STOCK PRICE PREDICTION')
l.pack()
l1=Label(text='For nth trading day after 20/7/2020 give n')
l1.pack()
e1=Entry()
e1.pack()
l2=Label(text='Enter open price for that day')
l2.pack()
e2=Entry()
e2.pack()
p=Button(text='Predict high,low,close price respectively', command=a1)
p.pack()
t=Button(text='Close window', command=a2)
t.pack()




