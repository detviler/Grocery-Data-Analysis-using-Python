#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[17]:


import pandas as pd
import numpy as np

data = {
    'Product Key': range(1, 3001), 
    'Product Description': ['Product_' + str(i) for i in range(1, 3001)],
    'Product Department': np.random.choice(['Produce', 'Frozen Food', 'Grocery', 'Baked Goods', 'Dairy', 'Meat', 'Pantry', 'Breakfast', 'Beverages','Canned Goods', 'Snacks','Condiments'], size=3000),
    'Product Cost': np.random.randint(0.5, 20.0, size=3000),
    'Date and Time of Purchase': pd.date_range(start='1/1/2023', periods=3000, freq='H'),
    'Age': np.random.randint(18, 80, size=3000),
    'Gender': np.random.choice(['Male', 'Female'], size=3000),
    'Location': np.random.choice(['New York', 'Chicago', 'Los Angeles', 'Houston', 'Miami', 'San Jose', 'Dallas','Austin','Jacksonville','San Diego','Portland','San Francisco'], size=3000),
    'Item Name': np.random.choice(['Tomatoes', 'Flour', 'Milk', 'Eggs', 'Bread', 'Apples', 'Bananas', 'Rice', 'Pasta', 'Cheese', 'Yogurt', 'Potatoes', 'Onions', 'Chicken', 'Beef', 'Pork', 'Butter', 'Sugar', 'Salt', 'Cereal', 'Canned Soup', 'Olive Oil', 'Cucumbers', 'Carrots', 'Lettuce', 'Spinach', 'Broccoli', 'Cauliflower', 'Bell Peppers', 'Orange Juice', 'Coffee', 'Tea', 'Peanut Butter', 'Jelly', 'Frozen Pizza', 'Frozen Vegetables', 'Ice Cream', 'Chocolate', 'Chips', 'Salsa', 'Tortilla Chips', 'Sliced Bread', 'Avocados', 'Sausages', 'Mustard', 'Ketchup', 'Mayonnaise', 'Salad Dressing', 'Frozen Desserts'], size=3000),
    'Item Category': np.random.choice(['Dairy', 'Bakery', 'Produce', 'Pantry', 'Meat', 'Breakfast', 'Canned Goods', 'Beverages', 'Frozen Foods', 'Snacks', 'Condiments'], size=3000),
    'Quantity': np.random.randint(1, 10, size=3000),
    'Price per Item': np.random.randint(1.0, 25.0, size=3000),
    'Total Cost': np.random.randint(3.0, 500.0, size=3000),
    'Payment Method': np.random.choice(['Cash', 'Credit Card', 'Debit Card', 'Mobile Payment'], size=3000),
    'Discounts/Promotions Applied': np.random.choice(['10% off', 'Free gift', 'Buy one get one free', '5% cashback'], size=3000)
}

df = pd.DataFrame(data)


# In[18]:


df.to_csv('sample_grocery_dataset_with_products.csv', index=False)


# In[1]:


import random
import csv

def generate_random_mobile_numbers(num_numbers):
    mobile_numbers = set()  # Using a set to ensure uniqueness
    while len(mobile_numbers) < num_numbers:
        # Generate a random 10-digit number
        mobile_number = '9' + ''.join(random.choice('0123456789') for _ in range(9))
        mobile_numbers.add(mobile_number)

    return list(mobile_numbers)

# Generate 50 random mobile numbers
random_numbers = generate_random_mobile_numbers(50)


csv_file_path = 'random_mobile_numbers.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['Mobile Number'])
    
   
    csv_writer.writerows([[number] for number in random_numbers])

print(f"CSV file '{csv_file_path}' has been created with the generated mobile numbers.")


# In[2]:


import random
import string

def generate_random_email():
    
    email_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'example.com', 'hotmail.com']

   
    username = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))

   
    domain = random.choice(email_domains)

  
    email = f"{username}@{domain}"

    return email


for _ in range(50):
    random_email = generate_random_email()
    print(random_email)


# In[3]:


import random

def generate_random_digits():
   
    random_number = random.randint(100000, 999999)
    return random_number


for _ in range(50):
    random_digits = generate_random_digits()
    print(random_digits)


# In[5]:


import random
import string

def generate_random_transaction_id():
  
    transaction_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return transaction_id


for _ in range(3000):
    random_transaction_id = generate_random_transaction_id()
    print(random_transaction_id)


# In[6]:


import random
from datetime import datetime, timedelta

def generate_random_date():
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)

    
    days_diff = (end_date - start_date).days

    
    random_days = random.randint(0, days_diff)

   
    random_date = start_date + timedelta(days=random_days)

    
    formatted_date = random_date.strftime("%d/%m/%Y")

    return formatted_date


for _ in range(3000):
    random_date = generate_random_date()
    print(random_date)


# In[1]:


get_ipython().system('pip install scikit-learn')


# In[5]:


import pandas as pd


file_path = '/Users/detviler/Downloads/Big Data Project/Python_Play.xlsx'
data = pd.read_excel(file_path)


data.head()


# In[8]:


data.head()


# In[9]:


# I am choosing the product "Broccoli"
selected_product = "Broccoli"

# Filtering data for the selected product
product_data = data[data['Product_Name'] == selected_product]

# Create a target variable 'Purchase' based on the 'Total Cost'

threshold = 10  
product_data['Purchase'] = product_data['Total Cost'].apply(lambda x: 'Yes' if x > threshold else 'No')


product_data.head()


# In[11]:


print(product_data.columns)


# In[19]:


import matplotlib.pyplot as plt


selected_product = "Frozen Desserts"


product_data = data[data['Product_Name'] == selected_product]


purchased_customers = product_data[product_data['Customers_name'].notnull()]


plt.figure(figsize=(10, 6))
plt.hist(purchased_customers['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Ages for Purchased Frozen Dessert')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # Code 2

# ## Code in Presentation 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Load your Excel data
data = pd.read_excel('/Users/detviler/Downloads/Big Data Project/sample_grocery_dataset_WIP.xls')


target_variable = 'Item Category'
features = ['Location', 'Total Cost']


location_cost_data = data[data[target_variable].notnull()]


X = pd.get_dummies(location_cost_data[features], drop_first=True)
y = location_cost_data[target_variable]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


max_depth = 3  # Set the maximum depth to 3
model_location_cost = DecisionTreeClassifier(max_depth=max_depth)
model_location_cost.fit(X_train, y_train)


tree_rules_location_cost = export_text(model_location_cost, feature_names=X.columns.tolist())
print(tree_rules_location_cost)


plt.figure(figsize=(15, 10))


plot_tree(
    model_location_cost,
    filled=True,
    feature_names=X.columns.tolist(),
    class_names=list(model_location_cost.classes_),  # Use the class names from the model as a list
    rounded=True,
    proportion=True,  # Show proportions in each class instead of counts
    fontsize=10  # Adjust fontsize for better readability
)

plt.title("Decision Tree for Item Category and Total Cost-Based Product Category (Max Depth = 3)", fontsize=16)
plt.show()


# In[47]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


product_counts = data['Product_Name'].value_counts()


plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x=product_counts.values, y=product_counts.index, palette='viridis')  # You can choose any color palette
plt.title('Term Frequency of Purchased Products')
plt.xlabel('Count')
plt.ylabel('Products')


for index, value in enumerate(product_counts.values):
    bar_plot.text(value, index, str(value), ha='left', va='center', fontsize=10, color='black')

plt.show()


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt


item_category_counts = data['Item Category'].value_counts()


plt.figure(figsize=(12, 6))
item_category_counts.sort_values(ascending=False).plot(kind='bar', color='orange')  # You can choose any color
plt.title('Term Frequency of Purchased Item Categories')
plt.xlabel('Item Categories')
plt.ylabel('Count')
plt.show()


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


item_category_counts = data['Item Category'].value_counts()


plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x=item_category_counts.values, y=item_category_counts.index, palette='viridis')  # You can choose any color palette
plt.title('Term Frequency of Purchased Item Categories')
plt.xlabel('Count')
plt.ylabel('Item Categories')


for index, value in enumerate(item_category_counts.values):
    bar_plot.text(value, index, str(value), ha='left', va='center', fontsize=10, color='black')

plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=df['Age'], color='lightgreen')
# plt.title('Customer Age Distribution (Box Plot)')
# plt.xlabel('Age')
# plt.show()
# 

# ## Age distribution 

# In[12]:


plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.grid(True)
plt.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# Box Plot for Customer Age Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Age'], color='orange')
plt.title('Customer Age Distribution (Box Plot)')
plt.xlabel('Age')
plt.show()


# In[17]:


payment_method_distribution = df['Payment Method'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(payment_method_distribution, labels=payment_method_distribution.index, autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightskyblue', 'lightgreen'])
plt.title('Payment Method Distribution')
plt.show()


# In[4]:


import pandas as pd


file_path = '/Users/detviler/Downloads/Big Data Project/sample_grocery_dataset_WIP.xls'
df = pd.read_excel(file_path)


average_prices = df.groupby('Item Category')['Price per Item'].mean()


print("Average Selling Prices by Item Category:")
print(average_prices)


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt


file_path = '/Users/detviler/Downloads/sample_grocery_dataset_Final.xls'
df = pd.read_excel(file_path)


print(df.columns)


product_category_counts = df['Item Category'].value_counts()


plt.figure(figsize=(12, 6))
product_category_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Purchases Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[2]:


print(df.columns)


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/detviler/Downloads/sample_grocery_dataset_Final.xls'
df = pd.read_excel(file_path)


plt.figure(figsize=(12, 6))
sns.histplot(df['Price per Item'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Prices per Item')
plt.xlabel('Price per Item')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='Item Category', y='Price per Item', data=df, palette='viridis')
plt.title('Price Distribution Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Price per Item')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[5]:


import pandas as pd


file_path = '/Users/detviler/Downloads/sample_grocery_dataset_Final.xls'
df = pd.read_excel(file_path)


average_age = df['Age'].mean()

print(f'The average age of shoppers is: {average_age:.2f} years')


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/Users/detviler/Downloads/sample_grocery_dataset_Final.xls'
df = pd.read_excel(file_path)


plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Age'], color='skyblue')
plt.title('Distribution of Shopper Ages')
plt.xlabel('Age')
plt.show()


# In[ ]:




