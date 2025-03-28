#!/usr/bin/env python
# coding: utf-8

# # Using Our TAS-B Bert_Dot (or BERT Dense Retrieval) Checkpoint
# 
# We provide a fully retrieval trained (with topic aware and balanced margin sampling: TAS-B) DistilBert-based instance on the HuggingFace model hub here: https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco
# 
# This instance can be used to **re-rank a candidate set** or **directly for a vector index based dense retrieval**. The architecure is a 6-layer DistilBERT, without architecture additions or modifications (we only change the weights during training) - to receive a query/passage representation we pool the CLS vector. 
# 
# If you want to know more about our efficient batch composition procedure and dual supervision for dense retrieval training, check out our paper: https://arxiv.org/abs/2104.06967 🎉
# 
# This notebook gives you a minimal usage example of downloading our Bert_Dot checkpoint to encode passages and queries to create a dot-product based score of their relevance. 
# 
# 
# ---
# 
# 
# Let's get started by installing the awesome *transformers* library from HuggingFace:
# 

# In[ ]:


# The next step is to download our checkpoint and initialize the tokenizer and models:
# 

# In[1]:

from transformers import AutoTokenizer, AutoModel

# you can switch the model to the original "distilbert-base-uncased" to see that the usage example then breaks and the score ordering is reversed :O
#pre_trained_model_name = "distilbert-base-uncased"
pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"

tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name) 
bert_model = AutoModel.from_pretrained(pre_trained_model_name)


# Now we are ready to use the model to encode two sample passages and a query:

# In[3]:


# our relevant example
passage1_input = tokenizer("We are very happy to show you the 🤗 Transformers library for pre-trained language models. We are helping the community work together towards the goal of advancing NLP 🔥.",return_tensors="pt")
# a non-relevant example
passage2_input = tokenizer("Hmm I don't like this new movie about transformers that i got from my local library. Those transformers are robots?",return_tensors="pt")
# the user query -> which should give us a better score for the first passage
query_input = tokenizer("what is the transformers library",return_tensors="pt")

print("Passage 1 Tokenized:",tokenizer.convert_ids_to_tokens(passage1_input["input_ids"][0]))
print("Passage 2 Tokenized:",tokenizer.convert_ids_to_tokens(passage2_input["input_ids"][0]))
print("Query Tokenized:",tokenizer.convert_ids_to_tokens(query_input["input_ids"][0]))

# note how we call the bert model independently between passages and query :)
# [0][:,0,:] pools (or selects) the CLS vector from the full output
passage1_encoded = bert_model(**passage1_input)[0][:,0,:].squeeze(0)
passage2_encoded = bert_model(**passage2_input)[0][:,0,:].squeeze(0)
query_encoded    = bert_model(**query_input)[0][:,0,:].squeeze(0)

print("---")
print("Passage Encoded Shape:",passage1_encoded.shape)
print("Query Encoded Shape:",query_encoded.shape)


# Now that we have our encoded vectors, we can generate the score with a simple dot product! 
# 
# (This can be offloaded to a vector indexing library like Faiss)
# 

# In[4]:


score_for_p1 = query_encoded.dot(passage1_encoded)
print("Score passage 1 <-> query: ",float(score_for_p1))

score_for_p2 = query_encoded.dot(passage2_encoded)
print("Score passage 2 <-> query: ",float(score_for_p2))


# As we see the model gives the first passage a higher score than the second - these scores would now be used to generate a list (if we run this comparison on all passages in our collection or candidate set). The scores are in the 100+ range (as we create a dot-product of 768 dimensional vectors, which naturally gives a larger score)
# 
# *As a fun exercise you can swap the pre-trained model to the initial distilbert checkpoint and see that the example doesn't work anymore*
# 
# - If you use our model checkpoint please cite our work as:
# 
#     ```
# @inproceedings{Hofstaetter2021_tasb_dense_retrieval,
#  author = {Sebastian Hofst{\"a}tter and Sheng-Chieh Lin and Jheng-Hong Yang and Jimmy Lin and Allan Hanbury},
#  title = {{Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling}},
#  booktitle = {Proc. of SIGIR},
#  year = {2021},
# }
#     ```
# 
# Thank You 😊 If you have any questions feel free to reach out to Sebastian via mail (email in the paper). 
# 
