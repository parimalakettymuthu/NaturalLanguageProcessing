#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas.core.computation.check import NUMEXPR_INSTALLED
import sagemaker


# In[2]:


session = sagemaker.Session()
bucket = session.default_bucket()
base = 'DEMO-loft-recommender'
prefix = 'sagemaker/'+ base

role = sagemaker.get_execution_role()


# In[3]:


print(f"Sagemaker session: {session} \n Default bucket: {bucket} \n Prefix: {prefix} \n Sagemaker role: {role}")


# In[4]:


import os
import pandas as pd
import numpy as np
import boto3
import json
import io
import matplotlib.pyplot as plt
import sagemaker.amazon.common as smac
#from sagemaker.serializers import json_deserializer
from scipy.sparse import csr_matrix


# In[5]:


os.environ["KAGGLE_USERNAME"] = ""
os.environ["KAGGLE_KEY"] = ""


# In[6]:


#from sagemaker.serializers


# In[7]:


#!mkdir /tmp/recsys/
#!aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz /tmp/recsys/


# In[8]:


#!pip install datasets


# In[9]:


if not os.path.exists("/tmp/recsys/amazon-us-customer-reviews-dataset.zip"): 
    get_ipython().system('pip install kaggle')
    get_ipython().system('kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset -p /tmp/recsys/')
else:
    print("Data already present.!")


# In[10]:


import zipfile

zipfilepath = '/tmp/recsys/amazon-us-customer-reviews-dataset.zip'
reviews_df = []
count = 0; read=0
with zipfile.ZipFile(zipfilepath, 'r') as zipPathRef:
    for file in zipPathRef.namelist():
        #if file.endswith('.tsv'):
        try:
            review_df = pd.read_csv(zipPathRef.open(file), delimiter="\t")
            reviews_df.append(review_df); read= read + 1
        except Exception as e:
            ##print(f"Error in processing file: {file} with exception {e}")
            count = count +1
print(f"Total number of failure in file reads {count} with reads successful {read}")
amazon_reviews_df = pd.concat(reviews_df, ignore_index=True)


# In[11]:


amazon_reviews_df.head()


# In[12]:


amazon_reviews_df = amazon_reviews_df[['customer_id', 'product_id', 'product_title', 'star_rating', 'review_date']]


# In[13]:


customers = amazon_reviews_df['customer_id'].value_counts()
products = amazon_reviews_df['product_id'].value_counts()
                             
quantiles = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
print('customers \n', customers.quantile(quantiles))
print('products \n', products.quantile(quantiles))


# In[14]:


customers = customers[customers>=5]
products = products[products>=10]

reduced_df = amazon_reviews_df.merge(pd.DataFrame({'customer_id': customers.index})).merge(pd.DataFrame({'product_id': products.index}))


# In[15]:


#Concatenate product titles to treat each one as a single word


# In[16]:


reduced_df['product_title'] = reduced_df['product_title'].apply(lambda x: x.lower().replace(' ', '-'))


# In[17]:


reduced_df.head()


# In[23]:


#Write customer purchase histories
first = True
product_list = []
filepath = '~/customer_purchase.txt' 
filepath = os.path.expanduser(filepath)
if os.path.exists(filepath):
    print(filepath)
else:
    print(filepath)


# In[24]:


filepath = '/home/ec2-user/customer_purchase.txt'
with open(filepath, 'w') as f:
    for customer, data in reduced_df.sort_values(['customer_id', 'review_date']).groupby('customer_id'):
        if first:
            first = False
        else:
            f.write('\n')
        f.write(' '.join(data['product_title'].tolist()))
        product_list.append(' '.join(data['product_title'].tolist()))


# In[25]:


inputs = session.upload_data(filepath, bucket, '{}/word2vec/train'.format(prefix))


# In[39]:


##TRAIN THE DATA in SAGEMAKER
train = sagemaker.estimator.Estimator(
sagemaker.amazon.amazon_estimator.get_image_uri(boto3.Session().region_name, 'blazingtext', 'latest'),
role,
train_instance_count=1,
train_instance_type='ml.m4.xlarge',
output_path='s3://{}/{}/output'.format(bucket, prefix),
sagemaker_session=session)


# In[40]:


#Set the algo hyperparameters 
#min_count : remove titles that occur less than 5 times
#vector_dim: Embed in a 100-dimensional subspace
train.set_hyperparameters(mode="skipgram", 
                         epochs=10,
                         min_count=5,
                         sampling_threshold=0.0001,
                         learning_rate=0.05,
                         window_size=5,
                         vector_dim=100,
                         negative_samples=5,
                         min_char=5,
                         max_char=10,
                         evaluation=False,
                         subwords=True)


# In[42]:


from sagemaker.inputs import TrainingInput
bucket_path = 's3://sagemaker-us-east-2-301219276882/sagemaker/DEMO-loft-recommender/word2vec/train/customer_purchase.txt'
bucket_path
s3_input_train = TrainingInput(bucket_path, distribution='FullyReplicated', content_type='text/plain')
train.fit({'train': s3_input_train})


# In[44]:


get_ipython().system('aws s3 cp $train.model_data ./')


# In[45]:


get_ipython().system('tar -xvzf model.tar.gz')


# In[46]:


vectors = pd.read_csv('vectors.txt', delimiter=' ', skiprows=2, header=None)


# In[48]:


vectors.sort_values(1)


# In[49]:


vectors.sort_values(2)


# In[50]:


product_titles = vectors[0]
vectors = vectors.drop([0, 101], axis=1)


# In[52]:


from sklearn.manifold import TSNE
tsne = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000)
embeddings = tsne.fit_transform(vectors.values[:100, ])


# In[54]:


from matplotlib import pylab
get_ipython().run_line_magic('matplotlib', 'inline')

def plot(embeddings, labels):
    pylab.figure(figsize=(20,20))
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                      ha='right', va='bottom')
    pylab.show()
plot(embeddings, product_titles[:100])


# In[55]:


#HOsting
bt_endpoint = train.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# In[61]:


#bt_endpoint.predict()


# In[64]:


#Predicting for set of titles
words = ["sherlock-season-1",
        "sherlock-season-2",
        "sherlock-season-5",
        "arbitrary-sherlock-holmes-string",
        "the-imitation-game",
        "abcdefghijklmn",
        "keeping-up-with-the-kardashians-season-1"]

#payload = {"instances" : words}
#json_payload = json.dumps(payload)
json_payload= '\n'.join(words)
print(json_payload)
response = bt_endpoint.predict(data=json_payload)

#vecs_df = pd.DataFrame(k=json.loads(response))


# In[ ]:




