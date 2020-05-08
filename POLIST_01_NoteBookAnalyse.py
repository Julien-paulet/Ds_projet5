#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *


# In[2]:


olistCustomersDataset = pd.read_csv('row_data/olist_customers_dataset.csv')
olistGeolocationDataset = pd.read_csv('row_data/olist_geolocation_dataset.csv')
olistOrderItemsDataset = pd.read_csv('row_data/olist_order_items_dataset.csv')
olistOrderPaymentsDataset = pd.read_csv('row_data/olist_order_payments_dataset.csv')
olistOrderReviewsDataset = pd.read_csv('row_data/olist_order_reviews_dataset.csv')
olistOrderDataset = pd.read_csv('row_data/olist_orders_dataset.csv')
olistProductsDataset = pd.read_csv('row_data/olist_products_dataset.csv')
olistSellersDataset = pd.read_csv('row_data/olist_sellers_dataset.csv')
productCategoryNameTranslation = pd.read_csv('row_data/product_category_name_translation.csv')


# # Dataset creation

# In[3]:


data_ = pd.merge(olistCustomersDataset, olistOrderDataset,
                 left_on = 'customer_id',
                 right_on = 'customer_id', how='inner')
data_ = pd.merge(data_, olistOrderItemsDataset, left_on='order_id',
                 right_on='order_id', how='inner')
data_ = pd.merge(data_, olistProductsDataset, left_on='product_id',
                 right_on='product_id', how='inner')
data_ = pd.merge(data_, olistOrderPaymentsDataset, left_on='order_id',
                 right_on='order_id', how='inner')
data_.shape


# Let's have a look at our dataset 

# In[4]:


data_.head()


# # Data Cleaning

# In[5]:


#Let's take a copy of our Df to continue. With that, we don't have to re-launch all of the above lines 
#If there is a mistake, or if we want to get back the originel Df
data = data_.copy()


# Let's see how much rows are empty for every columns 

# In[6]:


isnan(data)


# The majority of columns with missing values are not interesting for now ; We will just remove the empty values 

# In[7]:


data = data.dropna().drop_duplicates()
data.shape


# In[8]:


data.describe()


# All columns' minimums are positive, and the max values are in the range </br>
# We can keep going and begin the features engineering

# In[9]:


dataClust = data.copy()


# # Features engineering

# ## Total number of orders

# In[10]:


#We will first do a groupby to get the number of orders by customer
nbrTotOrder = dataClust.groupby(['customer_unique_id']).count()[['order_id']]
#Let's rename the column, for better understanding
nbrTotOrder = nbrTotOrder.rename(columns={'order_id':'nbr_tot_order'})
#And we merge on our main df
dataClust = pd.merge(dataClust, nbrTotOrder, left_on='customer_unique_id',
                     right_on=nbrTotOrder.index, how='left')


# ## Average total items by order -- Not used

# In[11]:


#We will first calculate the total nbr of items by order, then on the last groupby bellow we will
#use .mean to get the average (if the customer has more than 1 order of course)
totItemsPerOrder = dataClust.groupby(['customer_unique_id',
                                      'order_id']).count()[['product_id']]
totItemsPerOrder = (totItemsPerOrder
                    .rename(columns={'product_id': 'average_product_per_order'}))
totItemsPerOrder = totItemsPerOrder.reset_index()
dataClust = pd.merge(dataClust, totItemsPerOrder,
                     left_on=['customer_unique_id', 'order_id'],
                     right_on=['customer_unique_id', 'order_id'], how='left')


# ## Number of unique products -- Not used

# In[12]:


#Let's first do a groupby and count the number of unique values
distinctProducts = dataClust.groupby(['customer_unique_id',
                                      'order_id']).agg({"product_id":"nunique"})
distinctProducts = distinctProducts.rename(columns={"product_id": "Nbr_unique_items"})
distinctProducts = distinctProducts.reset_index()
dataClust = pd.merge(dataClust, distinctProducts,
                     left_on=['customer_unique_id', 'order_id'],
                     right_on=['customer_unique_id', 'order_id'],
                     how='left')


# ## Ratio 'nbr unique products'/'nbr different products' -- Not used

# In[13]:


#We calculate Nbr_unique_items / average_product_per_order
#Here the average is actually a total, but will become an average on the last groupby
dataClust['Ratio_items'] = (dataClust['Nbr_unique_items'] /
                            dataClust['average_product_per_order'])


# ## Sum of price (value)

# In[14]:


#We just do the sum of every paiement made by customer
sumOfPrice = dataClust.groupby(['customer_unique_id']).sum()[['payment_value']]
sumOfPrice = sumOfPrice.rename(columns={'payment_value': 'sum_of_price'})
dataClust = pd.merge(dataClust, sumOfPrice, left_on='customer_unique_id',
                     right_on=sumOfPrice.index, how='left')


# ## Sum of products (quantity) -- Not used

# In[15]:


#We will just take the average product per order, but we will do a sum instead of a mean in the last
#Groupby
dataClust['tot_items'] = dataClust['average_product_per_order']


# ## Nbr of orders on current year

# Let's first create a table of date for our customers according the orders

# In[16]:


#We split on ' ' to get time and date
dataClust['date'], dataClust['time'] = (dataClust['order_purchase_timestamp']
                                        .str.split(pat=' ').str)
#We then get the year, month, day
(dataClust['Year'], dataClust['Month'],
 dataClust['Day']) = dataClust['date'].str.split(pat='-').str


# We transform the date into datetime format

# In[17]:


dataClust['date'] = pd.to_datetime(dataClust['date'])


# We can now calculate the number of order on current year

# In[18]:


nbrOrderThisYear = dataClust[dataClust['Year'] == '2018']
nbrOrderThisYear = (nbrOrderThisYear
                    .groupby(['customer_unique_id']).count()[['order_id']])
nbrOrderThisYear = (nbrOrderThisYear
                    .rename(columns={'order_id': 'nbr_orders_current_year'}))
dataClust = pd.merge(dataClust, nbrOrderThisYear,
                     left_on ='customer_unique_id',
                     right_on = nbrOrderThisYear.index,
                     how='left')
dataClust = dataClust.fillna(value={'nbr_orders_current_year':0})


# ## Nbr of days since last order

# In[19]:


#We will choose an arbitrary date for now, but we need to make sure that once the projet is on production, it
#Takes the today's date 
today = datetime.strptime('2018-09-01', '%Y-%m-%d')
nbrDaySinceLastOrder = (dataClust[['customer_unique_id', 'date']]
                        .groupby(['customer_unique_id']).max()[['date']])


# In[20]:


#We calculate the delta between today and last order's date
nbrDaySinceLastOrder['delta'] = today - nbrDaySinceLastOrder['date']
#We get the number of days
nbrDaySinceLastOrder['nbr_day_since_last_order'] = (nbrDaySinceLastOrder['delta']
                                                    .dt.days)
#We keep only the value 
nbrDaySinceLastOrder = nbrDaySinceLastOrder[['nbr_day_since_last_order']]
#We merge on main df
dataClust = pd.merge(dataClust, nbrDaySinceLastOrder,
                     left_on='customer_unique_id',
                     right_on=nbrDaySinceLastOrder.index,
                     how='left')


# ## Promotions -- Not used

# In[21]:


#We begin by a simple groupby
totalPriceForPromotion = dataClust.groupby(['customer_id',
                                            'order_id']).agg({'price': np.sum,
                                                              'freight_value': np.sum,                                                              
                                                              'payment_value': np.max})
#We then add price and freigh_value
totalPriceForPromotion['tot_price_for_promo'] = (totalPriceForPromotion['price']
                                                 + totalPriceForPromotion['freight_value'])
#Np where to get 1 if promotion, else 0
totalPriceForPromotion['nbr_promo'] = (np.where(totalPriceForPromotion['payment_value']
                                                .astype(int) == totalPriceForPromotion['tot_price_for_promo']
                                                .astype(int), 0, 1))
#We reset the index
totalPriceForPromotion = totalPriceForPromotion[['nbr_promo']].reset_index()
#Merge on main df
#dataClust = pd.merge(dataClust, totalPriceForPromotion,
                      #left_on=['customer_unique_id', 'order_id'],
                      #right_on=['customer_unique_id', 'order_id'],
                      #how='left')


# ## Is customer happy -- Not used

# In[22]:


#We take only the order id and the review score
reviews = olistOrderReviewsDataset[['order_id', 'review_score']]
#We merge on main df
dataClust = pd.merge(dataClust, reviews,
                     left_on='order_id',
                     right_on='order_id',
                     how='left')
#We'll do a np.where() after the last groupby to get the overall satisfaction of the customer (1:Happy, 0:Unhappy)


# In[23]:


dataClust = dataClust.drop_duplicates()


# # Creation of a country and category of products DataFrame

# In[24]:


#This table will help us in the next step, to add the country and the category after the clustering
dataCategCountry = data[['customer_unique_id', 'customer_state',
                         'product_category_name']]
dataCategCountry = dataCategCountry.drop_duplicates()


# # Final DataFrame : 1 line = 1 customer

# After few tries, some variables that we created are not significant for the clustering ; We will directly remove them

# In[25]:


dataClust_ = dataClust[['customer_unique_id', 'price', 'payment_value',
                        'nbr_tot_order', 'average_product_per_order', 'sum_of_price',
                        'Ratio_items', 'tot_items', 'nbr_orders_current_year',
                        'nbr_day_since_last_order', 'review_score']]


# The goal here is to have 1 customer per line, so we'll do a groupby (known as the "last groupby" in previous parts)

# In[26]:


dataClust_ = dataClust_.rename(columns={'price':'average_basket(e)'})
dataGroup = dataClust_.groupby('customer_unique_id').agg({'average_basket(e)': np.mean,
                                                     'payment_value': np.mean,
                                                     'nbr_tot_order': np.max,
                                                     'average_product_per_order': np.mean,
                                                     'sum_of_price': np.max,
                                                     'Ratio_items': np.mean,
                                                     'tot_items': np.sum,
                                                     'nbr_orders_current_year': np.max,
                                                     'nbr_day_since_last_order': np.max,
                                                     'review_score': np.mean})
#We now can perform the np.where for the customer satisfaction
dataGroup['is_happy'] = np.where(dataGroup['review_score'] >= 3, 1, 0)
dataGroup = dataGroup.drop(columns=['review_score'])


# # Analyse exploratoire

# ## Histogrammes

# In[27]:


for i in dataGroup.columns:
    dataGroup[i].hist(bins=50)
    plt.xlabel('Valeur')
    plt.ylabel("Nombre d'individus (log)")
    plt.title('Histogramme de' + ' ' + i)
    plt.xlim(0, max(dataGroup[i]))
    plt.yscale('log')
    plt.show()


# Those Histograms show us that there is not simple separation between customers on 1 variable ; We will need to perform a clustering to see if with more variables we see different clusters

# ## Pairplot

# In[28]:


sns.pairplot(dataGroup)
plt.show()


# Same thing for the pairplot, with 2 variables the clusters are not seeable (or almost not)

# ## Heatmap des corrélations

# In[29]:


pearson_corr = round(dataGroup.corr(), 2)
plt.figure(figsize=(20, 20))
sns.heatmap(pearson_corr,
            xticklabels=pearson_corr.columns,
            yticklabels=pearson_corr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
plt.savefig('Graphiques/Heatmap_des_correlations.png')
plt.show()


# We can see here that there is correlation between some variables. <br/>
# Like average basket and payment value, or total number of orders and average product per command <br/>
# As Clustering algorithms needs a lot of power to compute, we will drop those variables to gain time (and efficiency)

# In[30]:


dataGroup = dataGroup.drop(columns=['payment_value', 'average_product_per_order'])


# Is_happy, Ratio_items and tot_items are not significant on the clustering, we will remove them

# In[31]:


dataGroup = dataGroup.drop(columns=['is_happy', 'Ratio_items', 'tot_items'])


# In[32]:


dataGroup.head()


# In[33]:


dataGroup.shape


# We now have our final dataset, we can save it as csv and reuse it in the POLIST_02_notebookessais.ipynb

# # Saving

# In[34]:


dataGroup.to_csv('files_cleaned/data_group.csv')
dataCategCountry.to_csv('files_cleaned/data_categ_country.csv')


# # Evolution of Customers in time
# 

# In this part, the goal is to see the evolution of customers, that will allow us to know when we'll need to re-train the model

# ## Revenue generated by month

# In[35]:


revenue = pd.merge(olistOrderDataset, olistOrderPaymentsDataset,
                   left_on='order_id',
                   right_on='order_id',
                   how='inner')[['order_id', 'order_approved_at', 'payment_value']]
#Groupby to see the total payment
revenue = revenue.groupby(['order_id', 'order_approved_at']).sum()
revenue = revenue.reset_index()
#Rename for better understanding
revenue = revenue.rename(columns={'order_approved_at': 'date'})
#Transform to datetime
revenue['date'] = pd.to_datetime(revenue['date']).dt.date
#Drop the day (we want the result by month)
revenue['date'] = revenue['date'].apply(lambda x: x.strftime('%Y-%m'))


# Groupby date to have the revenue generated by month

# In[36]:


#OWe remove september 2018, this month isn't complete
caParMois = revenue[revenue['date'] < '2018-09']
caParMois = caParMois.groupby('date').sum()


# We then calculate the augmentation (or diminution) of revenu in function of m-1

# In[37]:


caParMois['m-1'] = caParMois.shift(1)
caParMois['diff'] = caParMois['payment_value'] - caParMois['m-1']


# And we plot 

# In[38]:


plt.figure(figsize=(20,10))
plt.plot(caParMois['payment_value'], label='Month')
plt.plot(caParMois['diff'], label='Difference m-1')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Revenue generated by month')
plt.legend()
plt.show()


# ## Evolution of nbr of customers

# In[39]:


custEvol = pd.merge(olistCustomersDataset, olistOrderDataset,
                    left_on='customer_id',
                    right_on='customer_id',
                    how='inner')[['customer_unique_id', 'order_id',
                                  'order_approved_at']]
custEvol = custEvol.dropna()
#Rename for better understanding
custEvol = custEvol.rename(columns={'order_approved_at': 'date'})
#date to datetime type
custEvol['date'] = pd.to_datetime(custEvol['date']).dt.date
#Drop the day
custEvol['date'] = custEvol['date'].apply(lambda x: x.strftime('%Y-%m'))


# We keep only the customers' id and we group by date <br/>
# The goal here is to see how many unique customers bought by month

# In[40]:


buy_per_month = custEvol[['customer_unique_id', 'date']]
buy_per_month = buy_per_month[buy_per_month['date'] < '2018-09']
buy_per_month = buy_per_month.groupby('date').count()
plt.figure(figsize=(20, 10))
plt.plot(buy_per_month)
plt.xlabel('Date')
plt.ylabel('Nbr Active Customers')
plt.title('Number of active customers by month')
plt.show()


# There were a great growth on the first months, the number is now stabilizing ; Which is good for us on the monitoring part

# ## Nbr of new customers per months

# Let's first create a df that will contain all the customers present the first month <br/>
# We will then add every new customer by month

# In[41]:


dbCust = custEvol[custEvol['date'] == '2016-09']
dbCust = dbCust.drop_duplicates(subset=['customer_unique_id'])
dbCust = dbCust[['customer_unique_id', 'date']]


# In[42]:


#We define the object we will iterate on
date_ = np.sort(custEvol['date'].unique())
#Empty df for the loop
dbCust = pd.DataFrame(columns=['customer_unique_id', 'date'])
#Iter on the unique date
for i in date_:
    cust = custEvol[custEvol['date'] == i]
    #Unique customers on the month
    cust = cust.drop_duplicates(subset=['customer_unique_id'])
    cust = cust[['customer_unique_id', 'date']]
    dbCust = dbCust.append(cust) #We add to df
    #Unique customer on the period
    dbCust.drop_duplicates(subset='customer_unique_id', keep='first')
dbCust = dbCust[dbCust['date'] < '2018-09']


# We now have a df with each customer and the date he came for the first time on the website <br/>
# We are now able to know the number of new customers by month

# In[43]:


#This df presents the number of new customers by month
newCustPerMonth = dbCust.groupby('date').count()
newCustPerMonth = (newCustPerMonth
                   .rename(columns={'customer_unique_id':'Nbr_new_customers'}))
#Cumsum will give us the total number of customers in function of the month
newCustPerMonth['nbr_existing_cust'] = newCustPerMonth['Nbr_new_customers'].cumsum()


# Let's plot those data

# In[44]:


plt.figure(figsize=(20, 10))
plt.plot(newCustPerMonth['Nbr_new_customers'], label='New Customers')
plt.plot(newCustPerMonth['nbr_existing_cust'], label='Existing Customers')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Evolution of Customers By Time')
plt.legend()
plt.savefig("Graphiques/Evol_nbr_cust.png")
plt.show()


# The number of new customers is stable in time. Which means that we will need to add this amount of customers on our model every month ; Fortunately, kmeans (if this is the model that we will choose) allows us to do a partial_fit <br/>
# The partial_fit will allow us to retrain our data without having to take all the data

# ##  Average time between two orders

# In[45]:


meanTimeOrders = pd.merge(olistCustomersDataset, olistOrderDataset,
                          left_on='customer_id', right_on='customer_id',
                          how='inner')[['customer_unique_id', 'order_id',
                                        'order_purchase_timestamp']]
meanTimeOrders = meanTimeOrders.dropna()
#Rename for better understanding
meanTimeOrders = meanTimeOrders.rename(columns={'order_purchase_timestamp':'date'})
#Date to datetime
meanTimeOrders['date'] = pd.to_datetime(meanTimeOrders['date']).dt.date
#drop the day
meanTimeOrders['date'] = meanTimeOrders['date'].apply(lambda x: x.strftime('%Y-%m-%d'))


# We are now going to create a Df that contains only customers who orders more than 1 time

# In[46]:


nbrOrders = meanTimeOrders.groupby('customer_unique_id').count()
nbrOrders = nbrOrders[nbrOrders['order_id'] == 2]
nbrOrders = nbrOrders.reset_index()['customer_unique_id']


# Let's get only the customers with 2 or more orders in meanTimeOrders

# In[47]:


meanTimeOrders = meanTimeOrders[meanTimeOrders['customer_unique_id'].isin(nbrOrders)]
meanTimeOrders = meanTimeOrders.sort_values(by='customer_unique_id')


# In[48]:


funcs = {
        'First Order': 'min',
        'Second Order': 'max'
    }
meanTimeOrders = (meanTimeOrders.groupby(by='customer_unique_id')['date']
                  .agg(funcs).reset_index())


# In[49]:


meanTimeOrders['First Order'] = pd.to_datetime(meanTimeOrders['First Order'])
meanTimeOrders['Second Order'] = pd.to_datetime(meanTimeOrders['Second Order'])
meanTimeOrders['MeanTime'] = (meanTimeOrders['Second Order']
                              - meanTimeOrders['First Order'])
meanTimeOrders['MeanTime'] = meanTimeOrders['MeanTime'].dt.days
meanTimeOrders = meanTimeOrders[['MeanTime']]


# Let's plot the results

# In[50]:


print('Mean Time between two orders: ', meanTimeOrders.mean().values)
print('Median Time between two orders: ', meanTimeOrders.median().values)


# In[51]:


meanTimeOrders['MeanTime'].hist(bins=50)
plt.xlabel('Nbr de jours entre les deux premières commandes')
plt.ylabel('Nbr Individus')
plt.title('Histogram: Nbr of days between two orders')
plt.savefig('Graphiques/Histo_nbr_days_between_two_orders.png')
plt.show()


# The median time indicates that 50% of customers do the second order between 0 and 18 days. <br/>
# That's another point for doing a partial fit every month, we will clearly see if customers are one time customers or begin to be regular customers 
# 
# *We don't take the mean here as there is some outliers values

# # Impact of country in customers behaviors

# This part's goal is to see if the country has an impact on the customers behaviors

# In[52]:


boxPlot = pd.merge(dataGroup, dataCategCountry, left_index=True,
                   right_on='customer_unique_id', how='left')
sousEchantillon = boxPlot.copy()
modalites = sousEchantillon["customer_state"].unique()
for var in dataGroup.columns:
    X = 'customer_state' # qualitative
    Y = var # quantitative
    groupes = []
    for m in modalites:
        groupes.append(sousEchantillon[sousEchantillon[X] == m][Y].dropna())
    medianProps = {'color': "black"}
    meanProps = {'marker': 'o', 'markeredgecolor': 'black',
                 'markerfacecolor': 'firebrick'}
    plt.figure(figsize=[8, 20])
    plt.boxplot(groupes, labels=modalites, showfliers=False, medianprops=medianProps,
                vert=False, patch_artist=True, showmeans=True, meanprops=meanProps)
    plt.title("Boxplot")
    plt.xlabel(var)
    plt.ylabel("customer_state")
    plt.show()


# There's a slightly difference between countries in customers behaviors, but it's not significant. <br/>
# Adding the country to our clustering won't bring any added value - But will bring a lot a computation time ! 
