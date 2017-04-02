# Databricks notebook source
 import numpy as np
from math import log

text = sc.textFile("/FileStore/tables/v710wp2n1486562454751/big2.txt")
counts = text.flatMap(lambda line: line.lower().split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

counts = counts.sortBy(lambda (word, count): -1 * count ) #sort the rdd so most frequent word appears first

word2index = dict()
index2count = dict()
start_index = 0

#Map every word to an index. The words will get an id in the order of occurrence frequency
for (word, count) in counts.collect():
    word2index[word] = start_index
    index2count[start_index] = count
    start_index = start_index + 1

window_size = 15

def calc_X(line):
    partial_X = []
    tokens = line.lower().strip().split()
    token_ids = [word2index[word] for word in tokens]   
    for index_in_line, word_id in enumerate(token_ids):
        context_ids = token_ids[max(0, index_in_line - window_size): index_in_line]
        contexts_len = len(context_ids)    
        for context_word_index, context_word_id in enumerate(context_ids):        
            distance = contexts_len - context_word_index    
            increment = 1.0 / float(distance)
            part = ((max (word_id, context_word_id), min(word_id, context_word_id)), increment)
            partial_X.append(part)
    return partial_X

#A distributed call for co-occurrence computation
X = text.flatMap(calc_X).reduceByKey(lambda x,y: x+y)

vector_dim = 50
iterations = 15
min_words_num = 5 #VOCAB_COUNT
x_max = 100
alpha = 0.75    
learning_rate=0.05

#Calculate here the vocabulary size, so the size of the accumulators array, W will contain only the words that satisfy the frequency condition
XX = X.filter(lambda ((i_main, i_context), xij): index2count[i_main] > min_words_num and index2count[i_context] > min_words_num)

vocab_size = XX.count()

    

W = [[sc.accumulator(np.random.random() - 0.5) for j in range(vector_dim)] for i in range(vocab_size*2)]
biases = [sc.accumulator(np.random.random() - 0.5) for i in range(vocab_size*2)]
gradient_squared = [[sc.accumulator(1.0) for j in range(vector_dim)] for i in range(vocab_size*2)]  
gradient_squared_biases = [sc.accumulator(1.0) for i in range(vocab_size*2)]


def train(((i_main, i_context), xij)):
  
    v_main = W_B.value[i_main]
    v_context = W_B.value[i_context + vocab_size]
    b_main = biases_B.value[i_main : i_main + 1]
    b_context = biases_B.value[i_context + vocab_size : i_context + vocab_size + 1]
    gradsq_W_main = gradient_squared_B.value[i_main]
    gradsq_W_context = gradient_squared_B.value[i_context + vocab_size]
    gradsq_b_main = gradient_squared_biases_B.value[i_main : i_main + 1]
    gradsq_b_context = gradient_squared_biases_B.value[i_context + vocab_size: i_context + vocab_size + 1]
    cooccurrence = xij
    
    weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1
    cost_inner = np.dot(v_main, v_context) + b_main[0] + b_context[0] - log(cooccurrence)
    #Here we calculate the final J: cost function
    cost = weight * (cost_inner ** 2)
    
    # calculate the gradients for words and biases
    grad_main = np.multiply (weight * cost_inner,  v_context)
    grad_context = np.multiply (weight * cost_inner, v_main)    
    grad_bias_main = weight * cost_inner
    grad_bias_context = weight * cost_inner
    
    #Update the words embeddings and the biases via accumulators
    #Use AdaGrad
    for i in range(vector_dim):
        W[i_main][i] += -1 * learning_rate * float(grad_main[i]) / np.sqrt(gradsq_W_main[i])
        W[i_context + vocab_size][i] += -1 * learning_rate * float(grad_context[i]) / np.sqrt(gradsq_W_context[i])
        #Update the squared gradients
        gradient_squared[i_main][i] += grad_main[i] ** 2
        gradient_squared[i_context + vocab_size][i] += grad_context[i] ** 2
        
    biases[i_main] += -1 * learning_rate * grad_bias_main / np.sqrt(gradsq_b_main[0]) 
    biases[i_context + vocab_size] += -1 * learning_rate * grad_bias_context  / np.sqrt(gradsq_b_context[0])
    
    gradient_squared_biases[i_main] += grad_bias_main ** 2
    gradient_squared_biases[i_context + vocab_size] += grad_bias_context ** 2
    
for i in range(iterations):
    #First copy the accumulators for efficient broadcasting        
    W_2_broadcast = [[W[i][j].value for j in range(vector_dim)] for i in range(vocab_size*2)]
    biases_2_broadcast = [biases[i].value for i in range(vocab_size*2)]
    gradient_squared_2_broadcast = [[gradient_squared[i][j].value for j in range(vector_dim)] for i in range(vocab_size*2)]  
    gradient_squared_biases_2_broadcast = [gradient_squared_biases[i].value for i in range(vocab_size*2)]
    #Broadcast values
    W_B = sc.broadcast(W_2_broadcast)
    biases_B = sc.broadcast(biases_2_broadcast)
    gradient_squared_B = sc.broadcast(gradient_squared_2_broadcast)
    gradient_squared_biases_B = sc.broadcast(gradient_squared_biases_2_broadcast)
   
    XX.foreach(train) #A call for a distributed training
    
print W_2_broadcast #print the one before-end trained vectors

# COMMAND ----------


