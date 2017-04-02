# Databricks notebook source
from random import randint
import sys
import math


K = 20
L = 20
T = 10

myLines = sc.textFile( "/FileStore/tables/3g2j9tyt1482750752575/ratings.dat") #original file
users_ratings = myLines.map(lambda line: (int(line.split("::")[0]), int(line.split("::")[1]), int(line.split("::")[2]))) #(user_id, item_id, score)

items_ratings = myLines.map(lambda line: (int(line.split("::")[1]), int(line.split("::")[0]), int(line.split("::")[2]))) #(item_id, user_id, score)

distinct_users = users_ratings.map(lambda x: x[0]).distinct()
distinct_items = items_ratings.map(lambda x: x[0]).distinct()

U = distinct_users.map(lambda x: (x, randint(0, K-1))).collectAsMap()
V = distinct_items.map(lambda x: (x, randint(0, L-1))).collectAsMap()

U_B = sc.broadcast(U) #(uid, kid)
V_B = sc.broadcast(V) #(iid, lid)

B = [[0 for x in range(L)] for y in range(K)] #initialize B with 0

def calc_B():
    B_cells = users_ratings.map(lambda (user_id, item_id, score): ((U_B.value[user_id], V_B.value[item_id]), (score, 1))) #(kid, lid), (score, 1)
    B_cells_summed = B_cells.reduceByKey(lambda (score1, count1), (score2, count2): (score1 + score2, count1 + count2)) # <(kid,lid), (sum, count)>

    for ((kid, lid),(score_sum, score_count)) in B_cells_summed.collect():
        B[kid][lid] = float(score_sum) / score_count 
    B_B = sc.broadcast(B)
    
calc_B()

def smart_calc_B():
    for kid in range(0,K):
        for lid in range(0,L):
            B[kid][lid] = float(new_B[kid][lid][0].value)/float(new_B[kid][lid][1].value)
    B_B = sc.broadcast(B)

user_itemcluster_score = users_ratings.map(lambda (uid, iid, score): (uid, [(V_B.value[iid], score)])) # RDD: list: (user_id -> (item_cluster_id, score))
user_allhisitems = user_itemcluster_score.reduceByKey(lambda list1,list2: list1 + list2) #user_id -> list [(item_cluster_id, score)]

item_usercluster_score = users_ratings.map(lambda (uid, iid, score): (iid, [(U_B.value[uid], score)])) #RDD: list: item_id -> (user_cluster_id, score)
item_allhisusers = item_usercluster_score.reduceByKey(lambda list1,list2: list1 + list2) #item_id -> list [(user_cluster_id, score)]

def cc_row_mapper((uid, scores)):
    current_error = 0
    clusters_errors = []
    for kid in range(0,K):
        for (lid, score) in scores:
            current_error += (score - B_B.value[kid][lid])**2
        clusters_errors.append(current_error)
        current_error = 0
    best_user_cluster = clusters_errors.index(min(clusters_errors))
    for (lid, score) in scores:
        new_B[best_user_cluster][lid][0]+=score
        new_B[best_user_cluster][lid][1]+=1
    return (uid, best_user_cluster)

def cc_column_mapper((iid, scores)):  
    current_error = 0
    clusters_errors = []
    for lid in range(0,L):
        for (kid, score) in scores:
            current_error += (score - B_B.value[kid][lid])**2
        clusters_errors.append(current_error)
        current_error = 0
    best_item_cluster = clusters_errors.index(min(clusters_errors))
    for (kid, score) in scores:
        new_B[kid][best_item_cluster][0]+=score
        new_B[kid][best_item_cluster][1]+=1
    return (iid, best_item_cluster)

for t in range(0,T):
    
    new_B = [[[sc.accumulator(0), sc.accumulator(0)] for x in range(L)] for y in range(K)]    
    U = user_allhisitems.map(cc_row_mapper).collectAsMap()
    U_B = sc.broadcast(U)
    smart_calc_B()
    
    new_B = [[[sc.accumulator(0), sc.accumulator(0)] for x in range(L)] for y in range(K)]
    V = item_allhisusers.map(cc_column_mapper).collectAsMap()
    V_B = sc.broadcast(V)
    smart_calc_B()
    
print "B: ", B_B.value
print "U: ", U_B.value
print "V: ", V_B.value

# COMMAND ----------


