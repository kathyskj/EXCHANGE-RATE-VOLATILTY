#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# EXPORT

# In[2]:


import os

os.chdir('..')
#C:\Users\Khadijah Jasni\Desktop\SITC_EXPORT

exp= pd.read_csv('C:/Users/Khadijah Jasni/Desktop/export_clean.csv', encoding='latin-1')


# In[3]:


exp.head()


# VISUALIZATION ON EXPORT

# In[4]:


### In Which year Malaysia exported highest in value
plt.figure(figsize=[11,7])
exp.groupby(by='YEAR')['VALUE'].sum().sort_values(ascending=True).plot.bar(color='green',edgecolor='Black')
plt.xlabel('Year')
plt.ylabel('Exports in Million RM')
ticks = np.arange(0, 4000, 1000)
#labels = ["{}RM".format(i//100) for i in ticks]
plt.yticks(ticks)
plt.title('Exports by year',fontdict={'fontsize': 20,'color':'Red'})
plt.show()


# In[5]:


### Exports trend during 2010-2018 using line plot

plt.figure(figsize=[11,7])
exp.groupby(by='YEAR')['VALUE'].sum().plot.line(color='red',marker='s',linestyle='--')
ticks = np.arange(2000, 4500, 5000)
labels = ["{}B USD".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.xlabel('Year',fontdict={'fontsize': 12,'color':'green'})
plt.ylabel('Exports in RM Million',fontdict={'fontsize': 12,'color':'green'})
plt.title('Exports in RM Million',fontdict={'fontsize': 20,'color':'Blue'})
plt.show()


# In[7]:


### To which country Malaysia exported highest?

plt.figure(figsize=[15,9])
a=exp.groupby(by=['COUNTRY'])['VALUE'].sum().sort_values(ascending=False).head(10)
a.plot(kind='bar',color='red',edgecolor='black')
plt.xlabel('country',fontdict={'fontsize': 12,'color':'green'})
plt.ylabel('Exports in RM Million',fontdict={'fontsize': 12,'color':'green'})
plt.title('Total Exports by country(2010-2020)',fontdict={'fontsize': 20,'color':'Blue'})
ticks = np.arange(0, 450000, 50000)
labels = ["{}B USD".format(i//1000) for i in ticks]
plt.yticks(ticks, labels)
plt.show()


# In[8]:


### which Malaysia commodity was exported(highest in value)
plt.figure(figsize=[15,9])
a=exp.groupby(by=['PRODUCT DESCRIPTION'])['VALUE'].sum().sort_values(ascending=False).head(10)
a.plot(kind='bar',color='cyan',edgecolor='black')
plt.xlabel('commodity',fontdict={'fontsize': 12,'color':'green'})
plt.ylabel('Exports in RM Million',fontdict={'fontsize': 12,'color':'green'})
plt.title('Total Exports by commodity(2010-2020)',fontdict={'fontsize': 20,'color':'Blue'})
ticks = np.arange(0, 450000, 50000)
labels = ["{}B USD".format(i//1000) for i in ticks]
plt.yticks(ticks, labels)
plt.show()


# IMPORT DATA

# In[9]:


imp= pd.read_csv('C:/Users/Khadijah Jasni/Desktop/import_clean.csv', encoding='latin-1')
imp.head()


# VISUALIZATION IMPORT

# In[10]:


### In Which year Malaysia imported highest in value
plt.figure(figsize=[11,7])
imp.groupby(by='YEAR')['VALUE'].sum().sort_values(ascending=False).plot.bar(color='green',edgecolor='Black')
plt.xlabel('Year')
plt.ylabel('Import in RM Miliion')
ticks = np.arange(0, 5000, 5000)
labels = ["{}B$".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.title('Imports by year',fontdict={'fontsize': 20,'color':'Red'})
plt.show()


# In[12]:


### Which commmodity has been imported highest by value?
plt.figure(figsize=[11,7])
imp_by_com=imp.groupby(by=['PRODUCT DESCRIPTION'])['VALUE'].sum().sort_values(ascending=False).head(10)
imp_by_com.plot(kind='bar',color='cyan',edgecolor='black')
plt.xlabel('Year',fontdict={'fontsize': 15,'color':'green'})
plt.ylabel('Import in RM Million',fontdict={'fontsize': 15,'color':'green'})
ticks = np.arange(0, 1600, 2000)
labels = ["{}B$".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.title('Total Imports by commodity(2010-2020)',fontdict={'fontsize': 20,'color':'Red'})
plt.show()


# In[13]:


### Imports trend during 2010-2021 using line plot
plt.figure(figsize=[11,7])
imp.groupby(by='YEAR')['VALUE'].sum().plot.line(color='red',marker='s',linestyle='--')
ticks = np.arange(3000, 6000, 5000)
labels = ["{}B USD".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.xlabel('Year',fontdict={'fontsize': 12,'color':'green'})
plt.ylabel('Imports in RM Million',fontdict={'fontsize': 12,'color':'green'})
plt.title('Imports in RM Million',fontdict={'fontsize': 20,'color':'Blue'})
plt.show()


# EXPORT-IMPORT

# In[15]:


### We can compare Import Vs. Export during 2010-2021

plt.figure(figsize=[11,7])
imp.groupby(by='YEAR')['VALUE'].sum().plot.line(color='red',marker='s',linestyle='--',label='import')
exp.groupby(by='YEAR')['VALUE'].sum().plot.line(color='blue',marker='o',linestyle='-.',label='export')
ticks = np.arange(1500, 6000, 5000)
labels = ["{}B USD".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.xlabel('Year',fontdict={'fontsize': 12,'color':'green'})
plt.ylabel('Imports in RM Million',fontdict={'fontsize': 12,'color':'green'})
plt.title('Imports Vs Export in RM Million(2010-2020)',fontdict={'fontsize': 20,'color':'Blue'})
plt.legend()
plt.show()


# In[16]:


#trade gap

x=imp.groupby(by='YEAR')['VALUE'].sum()
y=exp.groupby(by='YEAR')['VALUE'].sum()
print(min(x-y))
print(max(x-y))
print(x-y)


# In[17]:


### Grouping countries for export data
df1=exp.groupby(by='COUNTRY')['VALUE'].sum()
df1.head()


# In[18]:


### Grouping countries for import data
df2=imp.groupby(by='COUNTRY')['VALUE'].sum()
df2.head()


# In[19]:


### Merging data frame on country

df3=pd.merge(df1,df2,on='COUNTRY')
df3.head()


# In[20]:


df3.rename(columns={'VALUE_x':'export_val','VALUE_y':'import_val'},inplace=True)


# In[21]:


df3.head()


# In[22]:


### Finding correlation b/w exoprt and import value
### Finding correlation b/w exoprt and import value
value_corr=df3.corr()
sns.heatmap(value_corr,annot=True)
plt.show()


# In[24]:


plt.figure(figsize=[15,7])
plt.scatter(df3.export_val,df3.import_val,marker='o',edgecolor='black',color='red')
plt.ylabel('Import value',fontdict={'fontsize':15,'color':'blue'})
plt.xlabel('export value',fontdict={'fontsize':15,'color':'blue'})
ticks_y = np.arange(0, 6000, 1000)
ticks_x = np.arange(0, 4000, 5000)
labels_y = ["{}B".format(i//100) for i in ticks_y]
labels_x = ["{}B".format(i//100) for i in ticks_x]
plt.yticks(ticks_y, labels_y)
plt.xticks(ticks_x, labels_x)
plt.title('Export-Import of Malaysia with another countries(2010-2020) in RM Million',fontdict={'fontsize':20,'color':'green'})
plt.show()


# In[25]:


### Making a column for trade difference to find out with which country Malaysia a been net exoprter and with a net importer?

df3['trade_diff']=df3['export_val']-df3['import_val']
df3.head()


# In[26]:


df3['NE/NI']=df3['trade_diff'].apply(lambda x: 'NE' if x>0 else 'NI')

### NE means Net Exporter
### NI means Net Importer
df3.head()


# In[27]:


### With How many country Malaysia have been Net Exporter and Net Importer.


#plt.figure(figsize=[10,5])
df3['NE/NI'].value_counts(normalize=True)


# In[31]:


### lets understand With bar graph

plt.figure(figsize=[10,7])
df3['NE/NI'].value_counts().plot.bar(color='green',edgecolor='black')
plt.title('Net Exports and Net Imports',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Number of countries',fontdict={'fontsize':15,'color':'blue'})
plt.xlabel('Net Export Vs Net Import',fontdict={'fontsize':15,'color':'blue'})
plt.show()


# In[32]:


### Which are top 5 countries to which Malaysia exports more and imoprt less?

plt.figure(figsize=[11,7])
top5=df3.sort_values(by='trade_diff',ascending=False).head(5)
top5.plot(kind='bar')
ticks = np.arange(0, 6000, 500)
labels = ["{}B USD".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.title('top trade surplus country(by value)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Values in RM Million',fontdict={'fontsize':15,'color':'blue'})
plt.xlabel('Country',fontdict={'fontsize':15,'color':'blue'})
plt.show()


# In[33]:


### Which are top 5 countries to which Malaysia imports more and export less?

plt.figure(figsize=[11,7])
top5=df3.sort_values(by='trade_diff',ascending=True).head(5)
top5.plot(kind='bar')
ticks = np.arange(-4000, 5000, 1000)
labels = ["{}B USD".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.title('top trade deficit country(by value)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Values in RM Million',fontdict={'fontsize':15,'color':'blue'})
plt.xlabel('Country',fontdict={'fontsize':15,'color':'blue'})
plt.show()


# In[34]:


### Which countries are  Malaysia's top 5 trade partner?

df3['total trade']=df3['export_val']+df3['import_val']
df3.head()


# In[35]:


plt.figure(figsize=[11,7])
df3.sort_values(by='total trade',ascending=False).head(5).plot.bar()
ticks = np.arange(-40000, 7000, 1000)
labels = ["{}B USD".format(i//100) for i in ticks]
plt.yticks(ticks, labels)
plt.title('Malaysia top 5 trade partner(by value)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Values in RM Million',fontdict={'fontsize':15,'color':'blue'})
plt.xlabel('Country',fontdict={'fontsize':15,'color':'blue'})
plt.show()


# In[36]:


c=df3.corr()
c


# In[37]:


plt.figure(figsize=[11,7])
c=df3.corr()
sns.heatmap(c,annot=True)
plt.show()


# In[38]:


plt.figure(figsize=[11,7])
plt.scatter(data=df3,x='total trade',y='import_val',marker='s',color='cyan',edgecolor='black')
plt.title('total trade vs import_val',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Import Values ',fontdict={'fontsize':15,'color':'blue'})
plt.xlabel('total trade',fontdict={'fontsize':15,'color':'blue'})
plt.show()


# In[39]:


plt.figure(figsize=[11,7])
plt.scatter(data=df3,x='total trade',y='export_val',marker='s',color='yellow',edgecolor='black')
plt.title('total trade vs export_val',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Export Values ',fontdict={'fontsize':15,'color':'blue'})
plt.xlabel('total trade',fontdict={'fontsize':15,'color':'blue'})
plt.show()


# In[40]:


### Export dataset grouped by country and year

exp_grpby_yr=exp.groupby(by=['COUNTRY','YEAR'])['VALUE'].sum()
exp_grpby_yr.head()


# In[41]:


### Import dataset grouped by country and year

imp_grpby_yr=imp.groupby(by=['COUNTRY','YEAR'])['VALUE'].sum()
imp_grpby_yr.head()


# In[42]:


### Merge both dataset which grouped by country and year both

merged_grp=pd.merge(exp_grpby_yr,imp_grpby_yr,on=['COUNTRY','YEAR'])
merged_grp.head()


# In[43]:


###Reseting index
merged_grp.reset_index(inplace=True)


# In[44]:


### rename column names

merged_grp.rename(columns={'VALUE_x':'val_exp','VALUE_y':'val_imp'},inplace=True)
merged_grp.head()


# In[45]:


### Adding columnn net_val
merged_grp['net_trade']=merged_grp['val_exp']-merged_grp['val_imp']
merged_grp.head()


# In[46]:


###For CHINA

CHINA=merged_grp[merged_grp['COUNTRY']=='CHINA']
CHINA


# In[47]:


### lINEPLOT FOR MALAYSIA'S TRADE WITH CHINA BETWEEN 2010-2021


plt.figure(figsize=[11,7])
# CHINA = merged_grp.query("COUNTRY == 'CHINA P RP'")
sns.lineplot(data=CHINA, x="YEAR", y="val_exp",label='exp',marker='o',color='red',linestyle='-.')
sns.lineplot(data=CHINA, x="YEAR", y="val_imp",label='imp',marker='s',color='blue',linestyle=':')
sns.lineplot(data=CHINA, x="YEAR", y="net_trade",label='net_trade',marker='s',color='magenta',linestyle='-')
plt.title('MALAYSIA trade with CHINA(2010-2018)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.legend()
plt.show()


# In[48]:


### Heatmap for Malaysia trade with CHINA 
plt.figure(figsize=[11,7])
CHINA_corr=CHINA.corr()
sns.heatmap(CHINA_corr,annot=True)
plt.title('MALAYSIA with CHINA',color='Blue')
plt.show()


# In[49]:


### For USA
USA=merged_grp[merged_grp['COUNTRY']=='UNITED STATES']
USA


# In[50]:


### lINEPLOT FOR MALAYSIA'S TRADE WITH USA BETWEEN 2010-2021

plt.figure(figsize=[11,7])
sns.lineplot(data=USA, x="YEAR", y="val_exp",label='exp',marker='o',color='red',linestyle='-.')
sns.lineplot(data=USA, x="YEAR", y="val_imp",label='imp',marker='s',color='blue',linestyle=':')
sns.lineplot(data=USA, x="YEAR", y="net_trade",label='net_trade',marker='s',color='magenta',linestyle='-')
plt.title('MALAYSIA trade with USA(2010-2021)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.legend()
plt.show()


# In[51]:


### Heatmap for Malaysia trade with USA


plt.figure(figsize=[11,7])
USA_corr=USA.corr()
sns.heatmap(USA_corr,annot=True)
plt.title('MALAYSIA with USA',color='Blue')
plt.show()


# In[52]:


### For SINGAPORE
SINGAPORE=merged_grp[merged_grp['COUNTRY']=='SINGAPORE']
SINGAPORE


# In[53]:


### lINEPLOT FOR MALAYSIA'S TRADE WITH SINGAPORE BETWEEN 2010-2021

plt.figure(figsize=[11,7])
sns.lineplot(data=USA, x="YEAR", y="val_exp",label='exp',marker='o',color='red',linestyle='-.')
sns.lineplot(data=USA, x="YEAR", y="val_imp",label='imp',marker='s',color='blue',linestyle=':')
sns.lineplot(data=USA, x="YEAR", y="net_trade",label='net_trade',marker='s',color='magenta',linestyle='-')
plt.title('MALAYSIA trade with SINGAPORE(2010-2021)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.legend()
plt.show()


# In[54]:


### For JAPAN
JAPAN=merged_grp[merged_grp['COUNTRY']=='JAPAN']
JAPAN


# In[55]:


### lINEPLOT FOR MALAYSIA'S TRADE WITH JAPAN BETWEEN 2010-2021

plt.figure(figsize=[11,7])
sns.lineplot(data=USA, x="YEAR", y="val_exp",label='exp',marker='o',color='red',linestyle='-.')
sns.lineplot(data=USA, x="YEAR", y="val_imp",label='imp',marker='s',color='blue',linestyle=':')
sns.lineplot(data=USA, x="YEAR", y="net_trade",label='net_trade',marker='s',color='magenta',linestyle='-')
plt.title('MALAYSIA trade with JAPAN(2010-2021)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.legend()
plt.show()


# In[56]:


### For THAILAND
THAILAND=merged_grp[merged_grp['COUNTRY']=='THAILAND']
THAILAND


# In[57]:


### lINEPLOT FOR MALAYSIA'S TRADE WITH THAILAND BETWEEN 2010-2021

plt.figure(figsize=[11,7])
sns.lineplot(data=USA, x="YEAR", y="val_exp",label='exp',marker='o',color='red',linestyle='-.')
sns.lineplot(data=USA, x="YEAR", y="val_imp",label='imp',marker='s',color='blue',linestyle=':')
sns.lineplot(data=USA, x="YEAR", y="net_trade",label='net_trade',marker='s',color='magenta',linestyle='-')
plt.title('MALAYSIA trade with THAILAND(2010-2021)',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.legend()
plt.show()


# In[58]:


d=merged_grp.groupby(by=['YEAR','COUNTRY'])['net_trade'].sum().sort_values(ascending=False)
d.reset_index()
d


# In[59]:


plt.figure(figsize=[11,7])
d=merged_grp.groupby(by=['YEAR','COUNTRY'])['net_trade'].sum().sort_values(ascending=False).head(5)
d.plot(kind='bar')
plt.title('TOP 5 country with highest surplus trade ',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.xlabel('country and year')
plt.show()


# In[60]:


plt.figure(figsize=[11,7])
d=merged_grp.groupby(by=['YEAR','COUNTRY'])['net_trade'].sum().sort_values(ascending=False).tail(5)
d.plot(kind='bar')
plt.title('TOP 5 country with highest trade deficit  ',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.xlabel('country and year')
plt.show()


# In[61]:


merged_grp['total trade']=merged_grp['val_imp']+merged_grp['val_exp']
merged_grp


# In[62]:


d=merged_grp.groupby(by=['YEAR','COUNTRY'])['total trade'].sum()
d


# In[63]:


plt.figure(figsize=[11,7])
d=merged_grp.groupby(by=['YEAR','COUNTRY'])['total trade'].sum().sort_values(ascending=False).head(5)
d.plot(kind='bar')
plt.title('TOP 5 country with highest total trade ',fontdict={'fontsize':20,'color':'green'})
plt.ylabel('Value in million RM')
plt.xlabel('country and year')
plt.show()


# # SECOND ANALYSIS

# In[7]:


'''Ignore deprecation and future, and user warnings.'''
import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 

'''Import basic modules.'''
import pandas as pd
import numpy as np
from scipy import stats

'''Customize visualization
Seaborn and matplotlib visualization.'''
import matplotlib.pyplot as plt
import seaborn as sns                   
sns.set_style("whitegrid") 

'''Plotly visualization .'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook

'''Display markdown formatted output like bold, italic bold etc.'''
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))


# In[8]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #statistical data visualization
import matplotlib.pyplot as plt #visualization library
from statsmodels.graphics.tsaplots import plot_acf #Auto-Correlation Plots
from statsmodels.graphics.tsaplots import plot_pacf #Partial-Auto Correlation Plots


# In[9]:


#exp= pd.read_csv('C:/Users/Khadijah Jasni/Desktop/full_export.csv', encoding='latin-1')

df_export = pd.read_csv('C:/Users/Khadijah Jasni/Desktop/export_clean/export_clean.csv', encoding='latin-1')
df_import = pd.read_csv('C:/Users/Khadijah Jasni/Desktop/import_clean/import_clean.csv', encoding='latin-1')


# In[10]:


exporting_countries=df_export[['COUNTRY']].nunique()
importing_countries=df_import[['COUNTRY']].nunique()
print("Malaysia imports from:",importing_countries,"countries")
print("Malaysia exports to:",exporting_countries,"countries")


# YEAR WISE ANALYSIS

# In[21]:


#convert data into year wise
exp_year = df_export.groupby('YEAR').agg({'VALUE': 'sum'})
exp_year = exp_year.rename(columns={'VALUE': 'Export'})
imp_year = df_import.groupby('YEAR').agg({'VALUE': 'sum'})
imp_year = imp_year.rename(columns={'VALUE': 'Import'})

#calculate growth of export and import
exp_year['Growth Rate(E)'] = exp_year.pct_change()
imp_year['Growth Rate(I)'] = imp_year.pct_change()

#calculate trade deficit
total_year = pd.concat([exp_year, imp_year], axis = 1)
total_year['Trade Deficit'] = exp_year.Export - imp_year.Import
total_year['Trade Surplus'] = imp_year.Import-exp_year.Export


display(total_year)
display(total_year.describe())


# In[22]:


total_year.to_csv(r'C:\Users\Khadijah Jasni\Desktop\try.csv')


# In[23]:


#convert data into year wise
exp_month = df_export.groupby('MONTH').agg({'VALUE': 'sum'})
exp_month = exp_month.rename(columns={'VALUE': 'Export'})
imp_month = df_import.groupby('MONTH').agg({'VALUE': 'sum'})
imp_month = imp_month.rename(columns={'VALUE': 'Import'})

#calculate growth of export and import
exp_month['Growth Rate(E)'] = exp_month.pct_change()
imp_month['Growth Rate(I)'] = imp_month.pct_change()

#calculate trade deficit
total_month = pd.concat([exp_month, imp_month], axis = 1)
total_month2 = pd.concat([exp_month, imp_month], axis = 1)
total_month['Trade Deficit'] = exp_month.Export - imp_month.Import
total_month2['Trade Surplus'] = imp_month.Import-exp_month.Export

display(total_month)
display(total_month.describe())

display(total_month2)
display(total_month2.describe())


# In[ ]:





# VISUALIZATION EXPORT AND IMPORT

# In[24]:


'''Visualization of Export and Import'''
# create trace1
trace1 = go.Bar(
                x = total_year.index,
                y = total_year.Export,
                name = "Export",
                marker = dict(color = 'rgb(55, 83, 109)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Export)
# create trace2 
trace2 = go.Bar(
                x = total_year.index,
                y = total_year.Import,
                name = "Import",
                marker = dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Import)


layout = go.Layout(hovermode= 'closest', title = 'Export and Import of Malaysia Trade from 2010 to 2020' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'RM (millions)'))
fig = go.Figure(data = [trace1, trace2], layout = layout)
fig.show()


# In[25]:


'''Visualization of Export/Import Growth Rate'''
# create trace1
trace1 = go.Scatter(
                x = total_year.index,
                y = total_year['Growth Rate(E)'],
                name = "Growth Rate(E)",
                line_color='deepskyblue',
                opacity=0.8,
                text = total_year['Growth Rate(E)'])
# create trace2 
trace2 = go.Scatter(
                x = total_year.index,
                y = total_year['Growth Rate(I)'],
                name = "Growth Rate(I)",
                line_color='dimgray',
                opacity=0.8,
                text = total_year['Growth Rate(I)'])

layout = go.Layout(hovermode= 'closest', title = 'Export and Import Growth Rate of Malaysia Trade from 2010 to 2020' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Growth Rate'))
fig = go.Figure(data = [trace1, trace2], layout = layout)
fig.show()


# In[26]:


'''Visualization of Export/Import and Trade Deficit'''
trace1 = go.Bar(
                x = total_year.index,
                y = total_year.Export,
                name = "Export",
                marker = dict(color = 'rgb(55, 83, 109)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Export)
# create trace2 
trace2 = go.Bar(
                x = total_year.index,
                y = total_year.Import,
                name = "Import",
                marker = dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Import)
# create trace3
trace3 = go.Bar(
                x = total_year.index,
                y = total_year['Trade Deficit'],
                name = "Trade Deficit",
                marker = dict(color = 'crimson',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year['Trade Deficit'])

layout = go.Layout(hovermode= 'closest', title = 'Export and Import and Trade Deficit of Malaysia Trade from 2010 to 2020' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'RM (millions)'))
fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)
fig.show()


# In[29]:


'''Visualization of Export/Import and Trade Surplus'''
trace1 = go.Bar(
                x = total_year.index,
                y = total_year.Export,
                name = "Export",
                marker = dict(color = 'rgb(55, 83, 109)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Export)
# create trace2 
trace2 = go.Bar(
                x = total_year.index,
                y = total_year.Import,
                name = "Import",
                marker = dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Import)
# create trace3
trace3 = go.Bar(
                x = total_year.index,
                y = total_year['Trade Surplus'],
                name = "Trade Surplus",
                marker = dict(color = 'crimson',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year['Trade Surplus'])

layout = go.Layout(hovermode= 'closest', title = 'Export and Import and Trade Surplus of Malaysia Trade from 2010 to 2021' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'RM (millions)'))
fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)
fig.show()


# COMMODITY-WISE ANALYSIS

# In[90]:


#Commodity export/Import count
print('Total number of Export commodity:', df_export['COMMODITY'].nunique())
print('Total number of Import commodity:', df_import['COMMODITY'].nunique())


# In[92]:


#The most importing and exporting commodities
bold('**Most Exporting Commodities(In Numbers) from 2010 to 2020**')
display(pd.DataFrame(df_export['COMMODITY'].value_counts().head(20)))
bold('**Most Importing Commodities(In Numbers) from 2010 to 2020**')
display(pd.DataFrame(df_import['COMMODITY'].value_counts().head(20)))


# In[93]:


#Coverting dataset in commodity wise'''
exp_comm = df_export.groupby('COMMODITY').agg({'VALUE':'sum'})
exp_comm = exp_comm.sort_values(by = 'VALUE', ascending = False)
exp_comm = exp_comm[:20]

imp_comm = df_import.groupby('COMMODITY').agg({'VALUE':'sum'})
imp_comm = imp_comm.sort_values(by = 'VALUE', ascending = False)
imp_comm = imp_comm[:20]


# In[95]:


#Visualization of Export/Import Commodity wise'''
def bar_plot(x,y, xlabel, ylabel, label, color):
    global ax
    font_size = 30
    title_size = 60
    plt.rcParams['figure.figsize'] = (40, 30)
    ax = sns.barplot(x, y, palette = color)
    ax.set_xlabel(xlabel = xlabel, fontsize = font_size)
    ax.set_ylabel(ylabel = ylabel, fontsize = font_size)
    ax.set_title(label = label, fontsize = title_size)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()
    
bar_plot(exp_comm.VALUE, exp_comm.index, 'RM (millions)', 'Commodities', 'Export of Malaysia (Commodity wise from 2010 to 2020)', 'gist_rainbow')
bar_plot(imp_comm.VALUE, exp_comm.index, 'RM (millions)', 'Commodities', 'Import of Malaysia (Commodity wise from 2010 to 2020)', 'rainbow')


# In[97]:


#Create pivot table of export/import (commodity wise)
exp_comm_table = pd.pivot_table(df_export, values = 'VALUE', index = 'COMMODITY', columns = 'YEAR')
imp_comm_table = pd.pivot_table(df_import, values = 'VALUE', index = 'COMMODITY', columns = 'YEAR')
bold('**Commodity Composition of Exports**')
display(exp_comm_table.sample(n=5))
bold('**Commodity Composition of Imports**')
display(imp_comm_table.sample(n=5))


# In[106]:


bold('**Trend of the Most Exporting Goods(In Values) From 2010 to 2021**')
plt.figure(figsize=(15,19))
 
plt.subplot(411)
g = exp_comm_table.loc["MACHINERY & TRANSPORT EQUIPMENT"].plot(color='green', linewidth=3)
g.set_ylabel('RM (millions)', fontsize = 15)
g.set_xlabel('Year', fontsize = 15)
g.set_title('Trend of Machinery & Transport Equipment', size = 20)

plt.subplot(412)
g1 = exp_comm_table.loc["MINERAL FUELS, LUBRICANTS, ETC."].plot(color='green', linewidth=3)
g1.set_ylabel('RM (millions)', fontsize = 15)
g1.set_xlabel('Year', fontsize = 15)
g1.set_title('Trend of Mineral Fuels, Lubricants, etc.', size = 20)

plt.subplot(413)
g2 = exp_comm_table.loc["MISCELLANEOUS MANUFACTURED ARTICLES"].plot(color='green', linewidth=3)
g2.set_ylabel('RM (millions)', fontsize = 15)
g2.set_xlabel('Year', fontsize = 15)
g2.set_title('Trend of Miscellaneous Manufactured Articles', size = 20)


plt.subplot(414)
g3 = exp_comm_table.loc["MANUFACTURED GOODS"].plot(color='green', linewidth=3)
g3.set_ylabel('RM (millions)', fontsize = 15)
g3.set_xlabel('Year', fontsize = 15)
g3.set_title('Trend of Manufactured Goods', size = 20)

plt.subplots_adjust(hspace = 0.5)
plt.show()


# In[107]:


bold('**Trend of the Most Importing Goods(In Values) From 2010 to 2021**')
plt.figure(figsize=(15,19))
 
plt.subplot(411)
g = imp_comm_table.loc["MACHINERY & TRANSPORT EQUIPMENT"].plot(color='red', linewidth=3)
g.set_ylabel('RM (millions)', fontsize = 15)
g.set_xlabel('Year', fontsize = 15)
g.set_title('Trend of Machinery & Transport', size = 20)

plt.subplot(412)
g1 = imp_comm_table.loc["MINERAL FUELS, LUBRICANTS, ETC."].plot(color='red', linewidth=3)
g1.set_ylabel('RM (millions)', fontsize = 15)
g1.set_xlabel('Year', fontsize = 15)
g1.set_title('Trend of Mineral Fuels, Lubricants, etc.', size = 20)

plt.subplot(413)
g2 = imp_comm_table.loc["MISCELLANEOUS MANUFACTURED ARTICLES"].plot(color='red', linewidth=3)
g2.set_ylabel('RM (millions)', fontsize = 15)
g2.set_xlabel('Year', fontsize = 15)
g2.set_title('Trend of Miscellaneous Manufactured Articles', size = 20)


plt.subplot(414)
g3 = imp_comm_table.loc["MANUFACTURED GOODS"].plot(color='red', linewidth=3)
g3.set_ylabel('RM (millions)', fontsize = 15)
g3.set_xlabel('Year', fontsize = 15)
g3.set_title('Trend of Manufactured Goods', size = 20)

plt.subplots_adjust(hspace = 0.4)
plt.show()


# COUNTRY-WISE ANALYSIS

# In[108]:


#Country export/Import count
print('Total number of country Export to:', df_export['COUNTRY'].nunique())
print('Total number of country Import from:', df_import['COUNTRY'].nunique())


# In[109]:


#Coverting dataset in Country wise
exp_country = df_export.groupby('COUNTRY').agg({'VALUE':'sum'})
exp_country = exp_country.rename(columns={'VALUE': 'Export'})
exp_country = exp_country.sort_values(by = 'Export', ascending = False)
exp_country = exp_country[:20]

imp_country = df_import.groupby('COUNTRY').agg({'VALUE':'sum'})
imp_country = imp_country.rename(columns={'VALUE': 'Import'})
imp_country = imp_country.sort_values(by = 'Import', ascending = False)
imp_country = imp_country[:20]


# In[111]:


#Visualization of Export/Import Country wise'''
bar_plot(exp_country.Export, exp_country.index, 'RM (millions)', 'Countries', 'Export of Malaysia (Country wise from 2010 to 2020)', 'plasma')
bar_plot(imp_country.Import, imp_country.index, 'RM (millions)', 'Countries', 'Import of Malaysia (Country wise from 2010 to 2020)', 'viridis')


# In[114]:


#Calculating trade deficit
total_country = pd.concat([exp_country, imp_country], axis = 1)
total_country['Trade Deficit'] = imp_country.Import-exp_country.Export 
total_country = total_country.sort_values(by = 'Trade Deficit', ascending = False)
total_country = total_country[:11]

bold('**Direction of Foreign Trade Export and Import and Trade Balance of Malaysia**')
display(total_country)
bold('**Descriptive statistics**')
display(total_country.describe())


# In[115]:


#Visualization of Export/Import and Trade Deficit
trace1 = go.Bar(
                x = total_country.index,
                y = total_country.Export,
                name = "Export",
                marker = dict(color = 'rgb(55, 83, 109)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Export)
# create trace2 
trace2 = go.Bar(
                x = total_country.index,
                y = total_country.Import,
                name = "Import",
                marker = dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Import)
# create trace3
trace3 = go.Bar(
                x = total_country.index,
                y = total_country['Trade Deficit'],
                name = "Trade Deficit",
                marker = dict(color = 'crimson',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year['Trade Deficit'])

layout = go.Layout(hovermode= 'closest', title = 'Export and Import and Trade Deficit of Malaysia Trade from 2010 to 2020(Country Wise)' , xaxis = dict(title = 'Country'), yaxis = dict(title = 'RM (millions)'))
fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)
fig.show()


# In[118]:


#Calculating trade surplus
total_country = pd.concat([exp_country, imp_country], axis = 1)
total_country['Trade Surplus'] = exp_country.Export - imp_country.Import
total_country = total_country.sort_values(by = 'Trade Surplus', ascending = False)
total_country = total_country[:11]

bold('**Direction of Foreign Trade Export and Import and Trade Balance of Malaysia**')
display(total_country)
bold('**Descriptive statistics**')
display(total_country.describe())


# In[122]:


#Visualization of Export/Import and Trade Surplus
trace1 = go.Bar(
                x = total_country.index,
                y = total_country.Export,
                name = "Export",
                marker = dict(color = 'rgb(55, 83, 109)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Export)
# create trace2 
trace2 = go.Bar(
                x = total_country.index,
                y = total_country.Import,
                name = "Import",
                marker = dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year.Import)
# create trace3
trace3 = go.Bar(
                x = total_country.index,
                y = total_country['Trade Surplus'],
                name = "Trade Surplus",
                marker = dict(color = 'crimson',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = total_year['Trade Deficit'])

layout = go.Layout(hovermode= 'closest', title = 'Export and Import and Trade Surplus of Malaysia Trade from 2010 to 2020(Country Wise)' , xaxis = dict(title = 'Country'), yaxis = dict(title = 'RM (millions)'))
fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)
fig.show()


# In[124]:


#Create pivot table of export/import (country wise)
exp_country_table = pd.pivot_table(df_export, values = 'VALUE', index = 'COUNTRY', columns = 'YEAR')
imp_country_table = pd.pivot_table(df_import, values = 'VALUE', index = 'COUNTRY', columns = 'YEAR')
bold('**Direction of Foreign Trade Export in Malaysia**')
display(exp_country_table.sample(n=5))
bold('**Direction of Foreign Trade Import in Malaysia**')
display(imp_country_table.sample(n=5))


# In[125]:


bold('**Trend of the Direction of Foreign Trade Export in Malaysia From 2010 to 2021**')
plt.figure(figsize=(15,19))
 
plt.subplot(411)
g = exp_country_table.loc["CHINA"].plot(color='purple', linewidth=3)
g.set_ylabel('RM (millions)', fontsize = 15)
g.set_xlabel('Year', fontsize = 15)
g.set_title('Trend of Export to the China', size = 20)

plt.subplot(412)
g1 = exp_country_table.loc["SINGAPORE"].plot(color='purple', linewidth=3)
g1.set_ylabel('RM (millions)', fontsize = 15)
g1.set_xlabel('Year', fontsize = 15)
g1.set_title('Trend of Export to the Singapore', size = 20)

plt.subplot(413)
g2 = exp_country_table.loc["UNITED STATES"].plot(color='purple', linewidth=3)
g2.set_ylabel('RM (millions)', fontsize = 15)
g2.set_xlabel('Year', fontsize = 15)
g2.set_title('Trend of Export to the United States', size = 20)


plt.subplot(414)
g3 = exp_country_table.loc["JAPAN"].plot(color='purple', linewidth=3)
g3.set_ylabel('RM (millions)', fontsize = 15)
g3.set_xlabel('Year', fontsize = 15)
g3.set_title('Trend of Export to the Japan', size = 20)

plt.subplots_adjust(hspace = 0.4)
plt.show()


# In[126]:


bold('**Trend of the Direction of Foreign Trade Import in Malaysia From 2010 to 2021**')
plt.figure(figsize=(15,19))
 
plt.subplot(411)
g = imp_country_table.loc["CHINA"].plot(color='coral', linewidth=3)
g.set_ylabel('RM (millions)', fontsize = 15)
g.set_xlabel('Year', fontsize = 15)
g.set_title('Trend of Import From the China', size = 20)

plt.subplot(412)
g1 = imp_country_table.loc["SINGAPORE"].plot(color='coral', linewidth=3)
g1.set_ylabel('RM (millions)', fontsize = 15)
g1.set_xlabel('Year', fontsize = 15)
g1.set_title('Trend of Import From the Singapore', size = 20)

plt.subplot(413)
g2 = imp_country_table.loc["UNITED STATES"].plot(color='coral', linewidth=3)
g2.set_ylabel('RM (millions)', fontsize = 15)
g2.set_xlabel('Year', fontsize = 15)
g2.set_title('Trend of Import From the United States', size = 20)


plt.subplot(414)
g3 = imp_country_table.loc["JAPAN"].plot(color='coral', linewidth=3)
g3.set_ylabel('RM (millions)', fontsize = 15)
g3.set_xlabel('Year', fontsize = 15)
g3.set_title('Trend of Import From the Japan', size = 20)

plt.subplots_adjust(hspace = 0.4)
plt.show()


# In[127]:


plt.rcParams['figure.figsize'] = (20, 7)
plt.style.use('seaborn-dark-palette')

sns.boxenplot(df_export['SITC'], df_export['VALUE'], palette = 'coolwarm')
plt.title('Comparison of Exports and SITC Code/Commodity', fontsize = 20)
plt.show()


# In[128]:


plt.rcParams['figure.figsize'] = (20, 7)
plt.style.use('seaborn-dark-palette')

sns.boxenplot(df_import['SITC'], df_export['VALUE'], palette = 'gist_heat_r')
plt.title('Comparison of Imports and SITC Code/Commodity', fontsize = 20)
plt.show()


# In[132]:


'Top expensive import and export '
expensive_import = df_import.sort_values(by='VALUE',  ascending=False).head(500)

import squarify
import matplotlib
temp1 = expensive_import.groupby(['COUNTRY']).agg({'VALUE': 'sum'})
temp1 = temp1.sort_values(by='VALUE')

norm = matplotlib.colors.Normalize(vmin=min(expensive_import.VALUE), vmax=max(expensive_import.VALUE))
colors = [matplotlib.cm.Blues(norm(VALUE)) for VALUE in expensive_import.VALUE]

value=np.array(temp1)
country=temp1.index
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(20, 5.5)
squarify.plot(sizes=value, label=country, color = colors, alpha=.6)
plt.title("Expensive Imports Countrywise Share", fontweight="bold")
plt.axis('off')
plt.show()


# In[ ]:


expensive_import = df_import.sort_values(by='value',  ascending=False).head(1000)
temp2 = expensive_import.groupby(['sitc']).agg({'value': 'sum'})
temp2 = temp2.sort_values(by='value')

norm = matplotlib.colors.Normalize(vmin=min(expensive_import.value), vmax=max(expensive_import.value))
colors = [matplotlib.cm.plasma(norm(value)) for value in expensive_import.value]

value=np.array(temp2)
country=temp2.index
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 4.5)
squarify.plot(sizes=value, label=country, color = colors, alpha=.6)
plt.title("Expensive Imports Commodity (SITCCode)", fontweight="bold")
plt.axis('off')
plt.show()


# In[134]:


import warnings
warnings.filterwarnings('ignore')

export_map = pd.DataFrame(df_export.groupby(['COUNTRY'])['VALUE'].sum().reset_index())
count = pd.DataFrame(export_map.groupby('COUNTRY')['VALUE'].sum().reset_index())

trace = [go.Choropleth(
            colorscale = 'algae',
            locationmode = 'country names',
            locations = count['COUNTRY'],
            text = count['COUNTRY'],
            z = count['VALUE'],
            reversescale=True)]

layout = go.Layout(title = 'Malaysia Export to Other Country')

fig = go.Figure(data = trace, layout = layout)
py.iplot(fig)


# In[135]:


import_map = pd.DataFrame(df_import.groupby(['COUNTRY'])['VALUE'].sum().reset_index())
count = pd.DataFrame(import_map.groupby('COUNTRY')['VALUE'].sum().reset_index())

trace = [go.Choropleth(
            colorscale = 'amp',
            locationmode = 'country names',
            locations = count['COUNTRY'],
            text = count['COUNTRY'],
            z = count['VALUE'],
            reversescale=True)]

layout = go.Layout(title = 'Malaysia Import from Other Country')

fig = go.Figure(data = trace, layout = layout)
py.iplot(fig)


# In[138]:


df_final_trade = pd.concat([df_export, df_import])
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.rcParams['figure.figsize'] = (12, 12)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 100).generate(' '.join(df_final_trade['COUNTRY']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Country in Trade With Malaysia',fontsize = 30)
plt.show()


# In[139]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.rcParams['figure.figsize'] = (12, 12)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 100).generate(' '.join(df_final_trade['COMMODITY']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Commodity in Trade With Malaysia',fontsize = 30)
plt.show()


# In[ ]:




