class Recurrent_node:
    def __init__(self,no_of_inputs):
        import random
        self.weight = [random.random() for i in range(no_of_inputs)]
        self.bias = random.random()
        self.bias2 = random.random()
        self.out_weight = random.random()
        self.prev_weight = random.random()
        self.inp = []
        self.prev = 0
        self.out = 0
        self.midout = 0   

class Node:
    def __init__(self):
        import random
        self.weight = [random.random()]
        self.bias = random.random()
        self.inp = [0]
        self.out = 0
        
def sigmoid(x):
    import numpy as np
    return 1/(1+np.exp(-x))

def sum_weight(node):
    n = node.bias
    for i in range(len(node.weight)):
        n += node.weight[i]*node.inp[i]
    return n

def expected_value(upper_bound,lower_bound,interval,out,n):
    #CHANGE LO AND OUTPUT TO output[lo] in main
    expected_values = [0 for i in range(int((upper_bound-lower_bound)/interval))]
    u = out[n]
    expected_values[int((u-lower_bound)/interval)] = 1
    return expected_values
    
def err(DNN,upper_bound,lower_bound,interval,output,lo,hours):
    exp = []
    err = []
    for i in range(hours):
        exp.append(expected_value(upper_bound, lower_bound, interval, output[lo[0]], i+1))
        err.append([exp[i][j]-DNN[i][j].out for j in range(len(DNN[i]))])
    return err

def compare(DNN,upper_bound,lower_bound,interval,output,lo,n):
    t = []
    y = 0
    for i in range(len(DNN)):
        x = []
        for j in range(len(DNN[i])):
            x.append(DNN[i][j].out)
        l = [1 if i == max(x) else 0 for i in x]
        r = expected_value(upper_bound,lower_bound,interval,output[lo[0]],n)
        for j in range(len(l)):
            if l[j] != r[j]:
                y = 1
        if y!=0:
            t.append(1)
        else:
            t.append(0)
        y = 0
    return t

def clear_values(RNN,DNN):
    for i in RNN:
        i.inp = []
        i.prev = 0
        i.out = 0
        i.midout = 0
        
    for i in DNN:
        for j in i:
            j.inp[0] = 0
            j.out = 0
    return RNN,DNN

def get_data():
    import pandas as pd
    name = input("Enter output file: ")
    output = pd.read_csv(name)
    i = -1
    while i<0:
        i = int(input("If 1 combined file, enter 0. Else enter number of files: "))
    
    
    data = pd.DataFrame()
    
    if i==0:
        name=input("Enter file: ")
        data = pd.read_csv(name)
    else:
        for j in range(i):
            s = "Enter file "+ str(j+1)+" : "
            name=input(s)
            d1=pd.read_csv(name)
            data = pd.concat([data,d1],axis=1)
    return data,output

def get_values():
    hours = int(input("Enter number of hours to be approximated: "))
    lower_bound,upper_bound = float(input("Enter lower bound :")),float(input("Enter upper bound: "))
    interval = float(input("Enter length of interval: "))
    l_rate = float(input("Enter learning rate: "))
    return hours,lower_bound,upper_bound,interval,l_rate

def build_network(input_size,hours,lower_bound,upper_bound,interval):
    recurrent = [Recurrent_node(input_size) for j in range(hours)]
    non_recurrent = [[Node() for j in range(int((upper_bound-lower_bound)/interval))] for k in range(hours)]
    return recurrent,non_recurrent

def forward_propogate(RNN,DNN,prev,inp):
    for i in range(len(RNN)):
        if i == 0:
            RNN[0].prev = prev
        for j in inp[i]:
            RNN[i].inp.append(j)
        f = RNN[i].prev*RNN[i].prev_weight + sum_weight(RNN[i])
        RNN[i].midout = sigmoid(f)
        if i != len(RNN)-1:
            RNN[i+1].prev = RNN[i].midout
        g = RNN[i].bias2 + RNN[i].out_weight*RNN[i].midout
        RNN[i].out = sigmoid(g)
        for j in DNN[i]:
            j.inp[0] += RNN[i].out
            m = sum_weight(j)
            j.out = sigmoid(m)
    return RNN,DNN

def back_propogate(RNN,DNN,err,l_rate):
    ernn = []
    for i in range(len(DNN)):
        n = 0
        for j in range(len(DNN[i])):
            n += err[i][j]*(DNN[i][j].out)*(1-DNN[i][j].out)*DNN[i][j].weight[0]
            x = l_rate*err[i][j]*(DNN[i][j].out)*(1-DNN[i][j].out)*DNN[i][j].inp[0]
            DNN[i][j].weight[0] += x
            
        ernn.append(n)

    for i in range(len(RNN)):
        ernn[i] *= RNN[i].out*(1-RNN[i].out)
        for j in range(i,-1,-1):
            if(i == j):
                RNN[j].bias2 += ernn[i]*l_rate
                w = RNN[j].out_weight
                RNN[j].out_weight += ernn[i]*l_rate*RNN[j].midout
                ernn[i] *= w
            e = RNN[j].prev_weight
            ernn[i] *= RNN[j].midout*(1-RNN[j].midout)
            RNN[j].prev_weight += ernn[i]*l_rate*RNN[j].prev
            RNN[j].bias += ernn[i]+l_rate
            for k in range(len(RNN[j].inp)):
                RNN[j].weight[k] += ernn[i]*l_rate*RNN[j].inp[k]
            ernn[i] *= e
    return RNN,DNN















