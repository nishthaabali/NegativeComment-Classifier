#!/usr/bin/env python
# coding: utf-8

# In[18]:

import os
import pandas as pd
import tensorflow as tf
import numpy as np

# In[27]:


# Specify the path to the CSV file
file_path = r'C:\Users\sanju\OneDrive\Desktop\train.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
df.head()

# In[29]:


df.iloc[2]['comment_text']

# In[31]:


from tensorflow.keras.layers import TextVectorization

# In[33]:




# In[40]:


X = df['comment_text']
y = df[df.columns[2:]].values

# In[45]:


Max_words = 300000

# In[46]:


vectorizer = TextVectorization(max_tokens=Max_words,
                               output_sequence_length=2000,
                               output_mode='int')

# In[47]:


vectorizer.adapt(X.values)

# In[50]:


vectorizer("Hi, there")[:2]

# In[51]:


vectorized_text = vectorizer(X.values)

# In[52]:


vectorized_text

# ### Tensorflow pipeline

# In[53]:


dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

# In[55]:


batch_X, batch_y = dataset.as_numpy_iterator().next()

# In[56]:


batch_X

# In[57]:


train = dataset.take(int(len(dataset) * .7))
val = dataset.skip(int(len(dataset) * .7)).take(int(len(dataset) * .2))
test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))

# In[58]:


from tensorflow.keras.models import Sequential

# In[59]:


from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

# In[61]:


model = Sequential()
model.add(Embedding(Max_words + 1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

# In[62]:


model.compile(loss='BinaryCrossentropy', optimizer='Adam')

# In[63]:


model.summary()

# In[64]:


history = model.fit(train, epochs=1, validation_data=val)

# In[82]:


history.history

# In[84]:


batch = test.as_numpy_iterator().next()

# In[85]:


input_text = vectorizer('You freaking suck! I am going to hit you.')

# In[86]:


res = model.predict(np.array([input_text]))

# In[87]:


(res > 0.5).astype(int)

# In[88]:


batch_X, batch_y = test.as_numpy_iterator().next()

# In[89]:


(model.predict(batch_X) > 0.5).astype(int)

# In[90]:


res.shape

# In[91]:


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

# In[92]:


pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

# In[78]:


for batch in test.as_numpy_iterator():
    # Unpack the batch
    X_true, y_true = batch
    # Make a prediction
    yhat = model.predict(X_true)

    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

# In[93]:


print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

model.save('model.h5')
