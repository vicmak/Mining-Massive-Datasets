from pyspark import SparkContext


def sum_three(x, y): #initial map-reduce, the biggest first scan on all data
    return[x[0]+y[0], x[1] + y[1], x[2] + y[2]]


def compute_var(x): #function for of the variance and all additional structure
    return [x[0],
            x[1][0],         # n_i
            x[1][1]/x[1][0], # mean_i
            x[1][1],         # sum_i
            (float(x[1][2])/x[1][0] - (float(x[1][1])/x[1][0] * float(x[1][1])/x[1][0]))] #variance_i


sc = SparkContext("local", "One-Way ANOVA")

myLines = sc.textFile('/Users/macbook/Downloads/big.txt')

mydata = myLines.map(lambda x: (x.split(',')[0], #Gid
                                [1,
                                float(x.split(',')[1]), #value
                                float(x.split(',')[1]) * float(x.split(',')[1]) #squared
                                ]))#sumi

summed_data = mydata.reduceByKey(sum_three)

SSw = 0
SSb = 0
total_sum = 0
total_n = 0
k = 0

groups = summed_data.map(compute_var)

groups.cache()

for group in groups.collect():
    SSw += group[1] * group[4]
    total_sum += group[3]
    total_n += group[1]
    k += 1

total_avg = float(total_sum)/total_n

for group in groups.collect():
    SSb += group[1] * (group[2] - total_avg) * (group[2] - total_avg)

F = (SSb/(k-1))/(SSw/(total_n - k))

print ("SSw", SSw)
print ("SSb", SSb)


print ("K", k)
print ("N", total_n)
print ("F", F)

