#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz,DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, ShuffleSplit,StratifiedKFold 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report, recall_score, precision_score,confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, concatenate, Flatten, Input, Dropout, LSTM

import pickle

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_df1= pd.read_csv('C:/Users/srikanta.panigrahi/Desktop/predruldemo-app/train_FD001.txt', sep=" ", header=None)


# In[3]:


test_df1 = pd.read_csv('C:/Users/srikanta.panigrahi/Desktop/predruldemo-app/test_FD001.txt', sep=" ", header=None)


# In[4]:


train_df1.head()


# In[5]:


test_df1.head()


# In[6]:


train_df1.shape


# In[7]:


test_df1.shape


# In[8]:


train_df1.describe()


# In[9]:


test_df1.describe()


# In[10]:


train_df1.drop(columns=[26,27], inplace= True)


# In[11]:


test_df1.drop(columns=[26,27], inplace= True)


# In[12]:


columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]


# In[13]:


train_df1.columns = columns


# In[14]:


test_df1.columns = columns


# In[15]:


train_df1.describe()


# In[16]:


test_df1.describe()


# In[17]:


MachineID_name = ["unit_number"]
RUL_name = ["time_in_cycles"]
OS_name = columns[2:5]
Sensor_name = columns[5:26]


# In[18]:


print (MachineID_name)
print (RUL_name)
print (OS_name) 
print (Sensor_name)


# In[19]:


train_df1.columns


# In[20]:


# Data in pandas DataFrame
MachineID_data = train_df1[MachineID_name]
RUL_data = train_df1[RUL_name]
OS_data = train_df1[OS_name]
Sensor_data = train_df1[Sensor_name]


# In[21]:


# Data in pandas Series
MachineID_series = train_df1["unit_number"]
RUL_series = train_df1["time_in_cycles"]


# In[22]:


grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["time_in_cycles"]) for i in MachineID_series.unique()])
max_cycles


# In[23]:


print("Max Life >> ",max(max_cycles))
print("Mean Life >> ",np.mean(max_cycles))
print("Min Life >> ",min(max_cycles))


# In[24]:


train_df1.info()


# In[25]:


train_df1.plot(x=RUL_name[0], y= OS_name[0], c='k')
train_df1.plot(x=RUL_name[0], y=OS_name[0], kind= "kde") 

train_df1.plot(x=RUL_name[0], y=OS_name[1], c='k')
train_df1.plot(x=RUL_name[0], y=OS_name[1], kind='kde')

train_df1.plot(x=RUL_name[0], y=OS_name[2], c='k')


# In[26]:


for name in Sensor_name:
    train_df1.plot(x=RUL_name[0], y=name, c='k')


# In[27]:


data = pd.concat([MachineID_data,RUL_data,OS_data,Sensor_data], axis=1)
data.drop(data[["TRA","T2","P2","P15",
                "epr","farB","Nf_dmd","PCNfR_dmd"]], axis=1 , inplace=True)


# In[28]:


data.describe()


# In[29]:


sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()


# In[30]:


data.corr()


# In[31]:


def prepare_train_data(data, factor = 0):
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    
    return df[df['time_in_cycles'] > factor]


# In[32]:


df = prepare_train_data(data)


# In[33]:


df.head()


# In[34]:


df["RUL"].dtype


# In[35]:


df.RUL[1]


# In[36]:


df['label'] = df['RUL'].apply(lambda x: 1 if x <= 30 else 0)


# In[37]:


df.label.value_counts()


# In[38]:


df.to_csv("C:/Users/srikanta.panigrahi/Desktop/predruldemo-app/Train_final.csv",header = True, index = None)


# In[39]:


upsamp = df.copy()

rest = df[df.label==1]
l_0 = len(df[df.label==0])
l_1 = len(df[df.label==1])
l_diff = l_0 - l_1
upsampled = resample(rest,replace=True,
                     n_samples = l_diff)
upsamp = pd.concat([upsamp,upsampled])
        
        
upsamp.label.value_counts()


# In[40]:


upsamp.head(10)


# In[41]:


upsamp.drop(columns=["RUL","unit_number"],inplace = True, axis = 1)


# In[42]:


upsamp.shape


# In[43]:


upsamp['label']=upsamp['label'].astype("category")


# In[44]:


df_target = upsamp['label']
df_data = upsamp.drop('label', axis=1)


# In[45]:


df_data.shape


# In[46]:


pca=PCA(n_components=5)
pca.fit(df_data)
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var1)


# In[47]:


Xp=pca.transform(df_data)


# ##### WITH PCA

# In[48]:


X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(Xp, df_target, test_size=0.30, random_state = 25)


# #### Without PCA

# In[49]:


X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.30, random_state = 25)


# #### Logistic:

# #### With PCA:

# In[50]:


lgreg_pca = LogisticRegression(solver='lbfgs')


# In[51]:


lgreg_pca.fit(X_train_p,y_train_p)


# In[52]:


x_pred_p = lgreg_pca.predict(X_train_p)


# In[53]:


print(accuracy_score(y_train_p,x_pred_p))
print(classification_report(y_train_p,x_pred_p))


# In[54]:


y_pred_p = lgreg_pca.predict(X_test_p)


# In[55]:


print(accuracy_score(y_test_p,y_pred_p))
print(classification_report(y_test_p,y_pred_p))


# #### Without PCA:

# In[56]:


lgreg = LogisticRegression(solver='lbfgs')


# In[57]:


lgreg.fit(X_train,y_train)


# In[58]:


x_pred = lgreg.predict(X_train)


# In[59]:


print(accuracy_score(y_train,x_pred))
print(classification_report(y_train,x_pred))


# In[60]:


y_pred = lgreg.predict(X_test)


# In[61]:


print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #### Decision Tree:

# #### With PCA: 

# In[62]:


dt_pca = DecisionTreeClassifier(max_depth= 15)


# In[63]:


dt_pca.fit(X_train_p,y_train_p)


# In[64]:


x_pred_p1 = dt_pca.predict(X_train_p)


# In[65]:


print(accuracy_score(y_train_p,x_pred_p1))
print(classification_report(y_train_p,x_pred_p1))


# In[66]:


y_pred_p1 = dt_pca.predict(X_test_p)


# In[67]:


print(accuracy_score(y_test,y_pred_p1))
print(classification_report(y_test,y_pred_p1))


# #### Without PCA:

# In[68]:


dt = DecisionTreeClassifier(max_depth= 12)


# In[69]:


dt.fit(X_train,y_train)


# In[70]:


x_pred1 = dt.predict(X_train)


# In[71]:


print(accuracy_score(y_train,x_pred1))
print(classification_report(y_train,x_pred1))


# In[72]:


y_pred1 = dt.predict(X_test)


# In[73]:


print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))


# #### Random Forest: 

# #### With PCA:

# In[74]:


rfc_pca = RandomForestClassifier(max_depth = 12, n_estimators= 14)
rfc_pca.fit(X=X_train_p, y=y_train_p)


# In[75]:


x_pred_p2 = rfc_pca.predict(X_train_p)


# In[76]:


print(accuracy_score(y_train_p,x_pred_p2))
print(classification_report(y_train_p,x_pred_p2))


# In[77]:


y_pred_p2 = rfc_pca.predict(X_test_p)


# In[78]:


print(accuracy_score(y_test_p,y_pred_p2))
print(classification_report(y_test_p,y_pred_p2))


# #### Without PCA:

# In[79]:


rfc = RandomForestClassifier(max_depth = 14, n_estimators= 15)
rfc.fit(X=X_train, y=y_train)


# In[80]:


x_pred2 = rfc.predict(X_train)


# In[81]:


print(accuracy_score(y_train,x_pred2))
print(classification_report(y_train,x_pred2))


# In[82]:


y_pred2 = rfc.predict(X_test)


# In[83]:


print(accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))


# #### XGBoost:

# #### With PCA: 

# In[84]:


param_grid = {
    'colsample_bytree': np.linspace(0.5,0.9,1),
     'n_estimators':[10,15,20],
     'max_depth': [2,3,4],
    'learning_rate' : [0.001,0.01]
}

 
CV_XGB_pca = GridSearchCV(XGBClassifier(RandomForestClassifier(max_depth=15)), param_grid=param_grid, cv= 2)


# In[85]:


get_ipython().run_line_magic('time', 'CV_XGB_pca.fit(X_train_p,y_train_p)')


# In[86]:


x_pred_p3 = CV_XGB_pca.predict(X_train_p)


# In[87]:


print(accuracy_score(y_train_p,x_pred_p3))
print(classification_report(y_train_p,x_pred_p3))


# In[88]:


y_pred_p3 = CV_XGB_pca.predict(X_test_p)


# In[89]:


print(accuracy_score(y_test_p,y_pred_p3))
print(classification_report(y_test_p,y_pred_p3))


# #### Without PCA: 

# In[90]:


param_grid = {
     'colsample_bytree': np.linspace(0.5,0.9,1),
     'n_estimators':[10,15,20,25],
     'max_depth': [15,20,25,30,35],
    'learning_rate' : [0.01,0.1]
}

 
CV_XGB = GridSearchCV(XGBClassifier(RandomForestClassifier()), param_grid=param_grid, cv= 2)


# In[91]:


get_ipython().run_line_magic('time', 'CV_XGB.fit(X_train,y_train)')


# In[92]:


x_pred3 = CV_XGB.predict(X_train)


# In[93]:


print(accuracy_score(y_train,x_pred3))
print(classification_report(y_train,x_pred3))


# In[94]:


y_pred3 = CV_XGB.predict(X_test)


# In[95]:


print(accuracy_score(y_test,y_pred3))
print(classification_report(y_test,y_pred3))


# #### Final Model: 

# #### XGB without PCA: 

# In[96]:


filename = 'finalized_model.pkl'


# In[97]:


pickle.dump(CV_XGB, open(filename, 'wb'))


# # Model Validation:

# In[98]:


test_df1 = pd.read_csv('C:/Users/srikanta.panigrahi/Desktop/predruldemo-app/test_FD001.txt', sep=" ", header=None)


# In[99]:


test_df1.head()


# In[100]:


test_df1.describe()


# In[101]:


test_df1.drop(columns=[26,27], inplace= True)


# In[102]:


columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]


# In[103]:


test_df1.columns = columns


# In[104]:


MachineID_data = test_df1[MachineID_name]
RUL_data = test_df1[RUL_name]
OS_data = test_df1[OS_name]
Sensor_data = test_df1[Sensor_name]


# In[105]:


test_df1.info()


# In[106]:


test_data = pd.concat([MachineID_data,RUL_data,OS_data,Sensor_data], axis=1)
test_data.drop(test_data[["TRA","T2","P2","P15",
                "epr","farB","Nf_dmd","PCNfR_dmd"]], axis=1 , inplace=True)


# In[107]:


test_data.describe()


# In[108]:


test = test_data.groupby('unit_number').max()
test.head()


# In[109]:


test.shape


# In[110]:


result_df = pd.read_csv('C:/Users/srikanta.panigrahi/Desktop/predruldemo-app/RUL_FD001.txt', sep=" ", header=None)


# In[111]:


result_df.head()


# In[112]:


result_df.drop(columns = [1], axis=1, inplace= True)


# In[113]:


result_df.head()


# In[114]:


col = ["label"]


# In[115]:


result_df.columns = col


# In[116]:


result_df['label'] = result_df['label'].apply(lambda x: 1 if x <= 30 else 0)


# In[117]:


result_df.label.value_counts()


# In[118]:


res_true = result_df["label"]


# #### Prediction:

# #### Logistic:

# In[119]:


res_pred = rfc.predict(test)


# In[120]:


print(accuracy_score(res_true,res_pred))
print(classification_report(res_true,res_pred))


# #### Decision Tree:

# In[121]:


res_pred_2 = dt.predict(test)


# In[122]:


print(accuracy_score(res_true,res_pred_2))
print(classification_report(res_true,res_pred_2))


# #### Random Forest: 

# In[123]:


res_pred_3 = rfc.predict(test)


# In[124]:


print(accuracy_score(res_true,res_pred_3))
print(classification_report(res_true,res_pred_3))


# #### XGB:

# In[125]:


loaded_model = pickle.load(open(filename, 'rb'))


# In[126]:


res_pred_4 = loaded_model.predict(test)


# In[127]:


print(accuracy_score(res_true,res_pred_4))
print(classification_report(res_true,res_pred_4))


# #### Function: 

# In[128]:


res = pd.DataFrame(res_pred_4)


# In[129]:


test1 = pd.DataFrame(test)


# In[130]:


test1.head()


# In[131]:


test2 = test1.reset_index()


# In[132]:


test2.head()


# In[133]:


j=0
for i, item in enumerate(res[0]):
    if(item == 1): 
        j= j+1
        print (" After 30 cycles Machine {} will breakdown".format(test2.unit_number[i]))
  
print("Number of machins about to fail:",j)

