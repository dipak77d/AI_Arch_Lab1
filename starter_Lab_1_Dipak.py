#!/usr/bin/env python
# coding: utf-8

# # Lab: Build, Train, and Evaluate a Machine Learning Pipeline for Predicting Customer Churn
# 
# ### Lab Objectives:
# 
# - Perform **exploratory data analysis (EDA)** to understand patterns in the dataset.
# - Build a **supervised learning pipeline** for churn prediction.
# - Evaluate the model using metrics like **precision, recall, and F1-score**.
# - Prepare a short **"client update"** summarizing key insights and proposed actions.

# ## Scenario:
# Your retail client has provided customer behavior data and needs a predictive model to identify at-risk customers. Your task is to build the model and prepare a presentation summarizing findings and recommendations.

# ## Task 1: Perform EDA to understand patterns in the dataset.
# In this section, we'll explore the retail sales dataset to identify patterns, understand customer behavior, and gain insights that will help us develop a churn prediction model.

# **Import necessary libraries**

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# **Load the Dataset**

# In[5]:


df = pd.read_csv("./data/data.csv")


# In[6]:


df.head()


# **Display basic statistics and check for missing values**

# In[7]:


print(df.info())
print(df.describe())
print(df.isnull().sum())


# In[8]:





# In[9]:





# **Cleaning and Transforming Data**
# 
# Now that we have an overview of the data, let's perform some preprocessing steps to prepare it for analysis.

# In[8]:


df.drop(['Transaction ID', 'Customer ID'], axis=1, inplace= True)


# In[9]:


df['Date'] = pd.to_datetime(df['Date'])


# **Visualize Key Patterns**

# In[16]:


import pandas as pd
import matplotlib
if not hasattr(matplotlib.rcParams, '_get'): 
    matplotlib.rcParams._get = lambda key: matplotlib.rcParams.get(key)
import matplotlib.pyplot as plt


# In[18]:


# Visualize Key Patterns:
import pandas as pd
import matplotlib
if not hasattr(matplotlib.rcParams, '_get'): 
    matplotlib.rcParams._get = lambda key: matplotlib.rcParams.get(key)
import matplotlib.pyplot as plt
gender_spending = df.groupby('Gender')['Total Amount'].agg(['mean', 'sum'])
print(gender_spending)
gender_spending.plot (kind='bar', figsize=(10, 6), title='Spending by Gender')


# In[ ]:





# In[ ]:





# **Group by Age Bins**
# 
# Let's segment customers by age groups to understand spending patterns across different demographics.

# In[28]:


df['Age Group'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 64], labels=['18-30', '31-40', '41-50', '51-64'])
age_group_spending = df.groupby('Age Group', observed=True)['Total Amount'].mean()
print(age_group_spending)

age_group_spending.plot(kind='bar', figsize=(8, 6), title='Average Spending by Age Group', color='lightblue')
plt.show()


# In[16]:





# In[17]:





# In[18]:





# **Analyze monthly sales trends and weekday vs. weekend sales**
# 
# Understanding sales patterns over time and days of the week can provide valuable insights for business strategy.

# In[ ]:





# In[25]:


# Sales trends by month
df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Total Amount'].sum()
print(monthly_sales)

monthly_sales.plot(kind='line', marker='o', figsize=(10, 6), title='Monthly Sales Trends')

plt.show()


# In[26]:


df['Weekday'] = df['Date'].dt.day_name()
weekday_sales = df.groupby('Weekday')['Total Amount'].sum()
weekday_sales.sort_index().plot(kind='bar', figsize=(8, 6), title='Sales by Day of the Week')
print(weekday_sales)
plt.show()


# In[22]:





# In[23]:





# In[24]:





# In[25]:





# In[26]:





# **Analyse the months with maximum and minimum sales**
# 
# Identifying peak and low sales periods can help with inventory planning and marketing strategies.

# In[29]:


max_month = monthly_sales.idxmax()
max_sales_month = weekday_sales.max()
min_month = monthly_sales.idxmin()
min_sales_month = weekday_sales.min()

print(f"Month with maximum sales: {max_month} (${max_sales_month})")
print(f"Month with minimum sales: {min_month} (${min_sales_month})")


# In[28]:





# In[29]:





# In[30]:





# In[31]:





# **Quantities Purchased By Categories**
# 
# Let's analyze which product categories are most popular with customers in terms of quantities purchased.

# In[22]:


prod_cat = df.groupby('Product Category')['Quantity'].sum().reset_index()
plt.pie(prod_cat['Quantity'], labels= prod_cat['Product Category'], autopct = '%1.1f%%')
plt.title('QUANTITIES PURCHASED BY CATEGORY')
plt.show()


# In[ ]:





# In[ ]:





# ## Task 2: Build the Churn Prediction Pipeline
# 
# In this section, we'll build a machine learning pipeline to predict customer churn based on their purchasing behavior.

# **Import required libraries for modeling**

# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score


# In[31]:


X = df[['Age', 'Quantity', 'Price per Unit']]
y = df['Total Amount']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 


# In[ ]:





# In[ ]:





# **Prepare the data for modeling**

# In[36]:


from sklearn.ensemble import RandomForestClassifier 


clf = RandomForestClassifier() 


clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
print("Accuracy Score :", accuracy_score(y_test, y_pred)*100, "%")


# In[40]:





# In[41]:





# In[42]:





# In[43]:





# In[44]:





# In[45]:





# In[46]:





# In[47]:





# **Train a linear regression model**

# In[35]:


model = LinearRegression()
model.fit(X_train, y_train)
lr_pred = model.predict(X_test)

print("\nLinear Regression Performance:")
print(f'Mean Squared Error: {mean_squared_error(y_test, lr_pred)}')
print(f'R-squared Score: {r2_score(y_test, lr_pred)}')


# In[49]:





# In[50]:





# In[51]:





# ## Task 3: Evaluate the model using metrics like precision, recall, and F1-score.
# 
# Let's evaluate our model's performance using various metrics to understand how well it predicts customer churn.

# In[37]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
y_test = y_test.astype(int)  # Convert y_test to integer labels, if necessary
y_pred = y_pred.astype(int)
print(classification_report(y_test,y_pred))


# In[53]:





# In[54]:





# In[55]:





# In[56]:





# In[ ]:





# In this lab, we have analyzed retail sales data to understand customer purchasing patterns and built a predictive model for customer churn. We first conducted exploratory data analysis (EDA) to identify patterns in customer behavior across demographics, time periods, and product categories. Then, we developed a machine learning pipeline using various algorithms to predict which customers are likely to churn. Finally, we evaluated the model's performance using appropriate metrics and suggested strategies to reduce customer churn based on our findings. This lab demonstrated how data analysis and machine learning can help businesses make informed decisions to improve customer retention.

# 
