import RNN_node

data,output = RNN_node.get_data()
l = list(data.columns.values)
lo = list(output.columns.values)
data2 = data.values.tolist()
hours = 6
lower_bound,upper_bound = 5,50
interval = 5
l_rate = 0.2

RNN,DNN = RNN_node.build_network(len(l), hours, lower_bound, upper_bound, interval)


for i in range(10000):
    RNN,DNN = RNN_node.forward_propogate(RNN, DNN, output[lo[0]][i],data2[i:i+len(RNN)])
    err = RNN_node.err(DNN,upper_bound,lower_bound,interval,output,lo,hours)
    RNN,DNN = RNN_node.back_propogate(RNN, DNN, err, l_rate)
    RNN,DNN = RNN_node.clear_values(RNN, DNN)
    for j in DNN:
        for k in j:
            k.inp[0] = 0

s = [0 for i in range(hours)]

for i in range(10000,12000):
    RNN,DNN = RNN_node.forward_propogate(RNN, DNN, output[lo[0]][i],data2[i:i+len(RNN)])
    m = RNN_node.compare(DNN,upper_bound,lower_bound,interval,output,lo,i)
    print(m)
    for j in range(hours):
        s[j] +=m[j]
    err = RNN_node.err(DNN,upper_bound,lower_bound,interval,output,lo,hours)
    RNN,DNN = RNN_node.back_propogate(RNN, DNN, err, l_rate)
    RNN,DNN = RNN_node.clear_values(RNN, DNN)
    
    
print(s)