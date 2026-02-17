import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 


# In[9]:


df = pd.read_csv('/Users/nickj/Downloads/Met dataset - 2015-to-2022_12months.csv')


# In[17]:


df.head()


# In[18]:


df.tail()


# In[19]:


df.describe()


# In[20]:


df.isnull().sum()


# In[21]:


df[["groundfrost_1", "groundfrost_2", "rainfall_8", "rainfall_9", "rainfall_10", "rainfall_11", "rainfall_12"]].stack().unique()


# In[22]:


df[["groundfrost_1", "groundfrost_2", "rainfall_8", "rainfall_9", "rainfall_10", "rainfall_11", "rainfall_12"]] = df[["groundfrost_1", "groundfrost_2", "rainfall_8", "rainfall_9", "rainfall_10", "rainfall_11", "rainfall_12"]].ffill()


# In[23]:


df.isna().sum()


# In[24]:


from sklearn.impute import KNNImputer
imputer=KNNImputer()


# In[25]:


for i in df.select_dtypes(include="number").columns:
    df[i] = imputer.fit_transform(df[[i]])


# In[26]:


df.isnull().sum()


# In[20]:


df.head()


# In[21]:


df.tail()


# In[22]:


df.describe()


# In[24]:


df.corr()


# In[25]:


plt.figure(figsize=(10,8))
sns.heatmap(data=df.corr(), annot=True, cmap='Blues')
plt.show()


# In[174]:


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df, x=i)
    plt.show()


# In[175]:


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df, x=i)
    plt.show()


# In[178]:


def wisker(col):
    q1,q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw

columns = [
    "psl_1", "psl_2", "psl_3", "psl_4", "psl_5", "psl_6", "psl_7", "psl_8", "psl_9", "psl_10", "psl_11", "psl_12",
    "sfcWind_1", "sfcWind_2", "sfcWind_3", "sfcWind_4", "sfcWind_5", "sfcWind_6", "sfcWind_7", "sfcWind_8", "sfcWind_9", "sfcWind_10", "sfcWind_11", "sfcWind_12",
    "rainfall_1", "rainfall_2", "rainfall_3", "rainfall_4", "rainfall_5", "rainfall_6", "rainfall_7", "rainfall_8", "rainfall_10", "rainfall_11", "rainfall_12"
]

for i in columns:
    lw, uw = wisker(df[i])
    print(f"Column: {i}, Lower Whisker: {lw}, Upper Whisker: {uw}")
    df[i] = np.where(df[i] < lw, lw, df[i])
    df[i] = np.where(df[i] > uw, uw, df[i])

# Plot a whisker plot for each column
sns.boxplot(df[columns])

plt.show()


# In[38]:


import warnings
warnings.filterwarnings("ignore")
sns.set_style(style='whitegrid')
color={2:'blue',
       3:'green',
       4:'black',
       5:'red',
       6:'violet',
       7:'brown',
       8:'orange'}
for index in range(2,9):
    plt.figure(figsize=(12,5))
    plt.xlabel('Year', fontsize=12)
    plt.title('{}'.format(df.columns[index].upper()), fontsize=15)
    sns.lineplot(data=df.iloc[:,index], 
                 color=color[index], marker='o')
    plt.show()


# In[179]:


import warnings
warnings.filterwarnings("ignore")
sns.set_style(style='whitegrid')
color={2:'blue',
       3:'green',
       4:'black',
       5:'red',
       6:'violet',
       7:'brown',
       8:'orange'}

columns_to_plot = [col for col in df.columns if '_2' in col]

for col in columns_to_plot:
    plt.figure(figsize=(12,5))
    plt.xlabel('Year', fontsize=12)
    plt.title('{}'.format(col.upper()), fontsize=15)
    sns.lineplot(data=df[['year', col]], x='year', y=col, color='blue', marker='o')
    plt.show()


# In[127]:


next_month = pd.to_datetime('2030-01-01')
df_next_month = df[(df['year'] >= next_month) & (df['year'] < next_month + pd.DateOffset(months=1))]


# In[128]:


#selection of specific columns for further analysis
indices = ['tas_2', 'hurs_2','psl_2', 'sfcWind_2']
df2 = df.loc[:,indices]
df2.head()


# In[129]:


df.duplicated().sum()


# In[39]:


df.drop_duplicates()


# In[181]:


plt.figure(figsize=(10,10))
plt.title('tas_2 vs hurs_2')
sns.scatterplot(x=df2.iloc[:,0],
                y=df2.iloc[:,1],
                hue=df2.iloc[:,3],
                s=75,
                alpha=0.3)
plt.show()


# In[180]:


plt.figure(figsize=(10,10))
plt.title('tas_2 vs hurs_2')
sns.scatterplot(x=df2.iloc[:,0],
                y=df2.iloc[:,1],
                hue=df2.iloc[:,2],
                s=75,
                alpha=0.3)
plt.show()


# In[140]:


jan_2030_data = df[df['year'] == 2030]

jan_cols = [col for col in jan_2030_data.columns if '_2' in col]
jan_2030_data = jan_2030_data[jan_cols]


# In[27]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x = df.loc[:, ['groundfrost_2', 'psl_2', 'pv_2', 'rainfall_2', 'sfcWind_2', 'snowLying_2', 'sun_2', 'tas_2']]


# In[30]:


y = df['hurs_2']


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=42)


# In[32]:


model.fit(x_train,y_train)


# In[33]:


y_predict=model.predict(x_test)
y_predict


# In[34]:


from sklearn.metrics import mean_squared_error, r2_score


# In[35]:


r2_score(y_test,y_predict)


# In[36]:


coef=model.coef_
for i,j in zip(x_train.columns,coef):
    print(f"{i})={j}")
    


# In[37]:


np.sqrt(mean_squared_error(y_test,y_predict))


# In[153]:


y_test.mean()


# In[154]:


1.725531824804055/82.96424275134265


# In[155]:


residual=y_test-y_predict


# In[156]:


residual.mean()


# In[183]:


sns.kdeplot(x=residual)
residual.skew()


# In[158]:


sns.scatterplot(x=y_test,y=residual)


# In[92]:


y_predict=model.predict(x_test)


# In[93]:


import matplotlib.pyplot as plt

# Calculate residuals
residual = y_test - y_predict - residual.mean()


plt.hist(residual, bins=50)
plt.show()


# In[160]:


from sklearn.preprocessing import StandardScaler


# In[166]:


# features for the model
feature_cols = ['groundfrost_2', 'psl_2', 'pv_2', 'rainfall_2', 'sfcWind_2', 'snowLying_2', 'sun_2', 'tas_2']
target_col = 'hurs_2'

x = df[feature_cols]
y = df[target_col]

# Normalize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Creating time-shifted features for forecasting
def create_time_features(data, target, lookback=1):
    x_t, y_t = [], []
    for i in range(len(data) - lookback):
        x_t.append(data[i:(i + lookback)])
        y_t.append(target[i + lookback])
    return np.array(x_t), np.array(y_t)

lookback = 1  # One month lookback
x_time, y_time = create_time_features(x_scaled, y, lookback)

# Split the data into training and testing sets
train_size = int(len(x_time) * 0.8)
x_train, x_test = x_time[:train_size], x_time[train_size:]
y_train, y_test = y_time[:train_size], y_time[train_size:]


# In[167]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_mlp_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Create the model
input_shape = (lookback, len(feature_cols))
model = create_mlp_model(input_shape)
model.summary()


# In[168]:


from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[171]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


y_pred = y_pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")


plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Relative Humidity')
plt.ylabel('Predicted Relative Humidity')
plt.title('Actual vs Predicted Relative Humidity')
plt.show()


# In[173]:


last_data_point = x_scaled[-1:]


input_data = last_data_point.reshape((1, lookback, len(feature_cols)))


february_2030_prediction = model.predict(input_data)

february_2030_humidity = february_2030_prediction.item()
print(f"Predicted Relative Humidity for February 2030: {february_2030_humidity:.2f}%")


def predict_future_months(model, last_data_point, num_months):
    predictions = []
    current_data = last_data_point.copy()
    
    for _ in range(num_months):
        input_data = current_data.reshape((1, lookback, len(feature_cols)))
        prediction = model.predict(input_data)[0][0]
        predictions.append(prediction)
        
        
        current_data = np.roll(current_data, -1)
        current_data[-1] = prediction
    
    return predictions

# Predicting for the next 6 months
future_predictions = predict_future_months(model, last_data_point, 6)


plt.figure(figsize=(12, 6))
plt.plot(range(1, 7), future_predictions, marker='o')
plt.title('Relative Humidity Predictions for the Next 6 Months')
plt.xlabel('Months from Now')
plt.ylabel('Predicted Relative Humidity (%)')
plt.show()


# In[194]:


import pandas as pd
import numpy as np


true_labels = np.array(true_labels).ravel()
predictions = np.array(predictions).ravel()

output_df = pd.DataFrame({
    'instance_id': range(1, len(true_labels) + 1),
    'true_label': true_labels,
    'prediction': predictions
})


output_df.to_csv('humidity_predictions.csv', index=False)


print(output_df.head())


print("\nDataset statistics:")
print(f"Number of instances: {len(output_df)}")
print(f"Mean true label: {output_df['true_label'].mean():.2f}")
print(f"Mean predicted label: {output_df['prediction'].mean():.2f}")
print(f"RMSE: {np.sqrt(((output_df['true_label'] - output_df['prediction'])**2).mean()):.4f}")

