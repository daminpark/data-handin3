import numpy as np
import compare_anns
import itertools

#read the fasta file
def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

#translate from letters to indices
def translate_observations_to_indices(obs):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[symbol] for symbol in obs]

#translate from annotations to indices
def translate_annotations_to_indices(obs):
    mapping = {'N': 0, 'C': 7, 'R': 8}
    return [mapping[symbol] for symbol in obs]

data={}
for i in np.arange(1,11):
    temp={}
    temp[f'genome{str(i)}'] = read_fasta_file(f'genome{str(i)}.fa')[f'genome{str(i)}']
    data.update(temp)
for i in np.arange(1,6):
    temp={}
    temp[f'true-ann{str(i)}'] = read_fasta_file(f'true-ann{str(i)}.fa')[f'true-ann{str(i)}']
    data.update(temp)

#load into data
data2 = {}
for i in np.arange(1,11):
    temp={}
    temp[f'genome{str(i)}'] = translate_observations_to_indices(read_fasta_file(f'genome{str(i)}.fa')[f'genome{str(i)}'])
    data2.update(temp)
for i in np.arange(1,6):
    temp={}
    temp[f'true-ann{str(i)}'] = translate_annotations_to_indices(read_fasta_file(f'true-ann{str(i)}.fa')[f'true-ann{str(i)}'])
    data2.update(temp)




#group codons
triplet_indices=[[0],[1],[2],[3]]
triplet_indices+=[list(x) for x in itertools.product([0,1,2,3],repeat=3)]

def group_codons(ann):
    i=1
    out=[0]
    while i < len(ann):
        if out[-1]==0 and ann[i]==0:
            out.append(0)
        elif (out[-1]==1 and ann[i]==7) or (out[-1]==2 and ann[i]==7) :
            out.append(2)
            i+=2
        elif (out[-1]==4 and ann[i]==8)  or (out[-1]==5 and ann[i]==8) :
            out.append(5)
            i+=2
        elif (out[-1]==0 and ann[i]==7) :
            out.append(1)
            i+=2
        elif (out[-1]==0 and ann[i]==8) :
            out.append(4)
            i+=2
        elif (out[-1]==2 and ann[i]==0) :
            out[-1]=3
            out.append(0)
        elif (out[-1]==5 and ann[i]==0) :
            out[-1]=6
            out.append(0)
        elif (out[-1]==2 and ann[i]==8) :
            out[-1]=3
            out.append(4)
            i+=2
        elif (out[-1]==5 and ann[i]==7) :
            out[-1]=6
            out.append(1)
            i+=2
        i+=1
    return out

data3=data2

for i in np.arange(1,6):
    data3[f"true-ann{str(i)}"]=group_codons(data3[f"true-ann{str(i)}"])




class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

d=[1,3,3,3,3,3,3]

def count_transitions_and_emissions_var(K, D, x, z):
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    kxk=np.zeros((K,K))
    kxd=np.zeros((K,D))
    count=0
    total=0
    while total < len(x):
        kxd[z[count]][triplet_indices.index(x[total:total+d[z[count]]])]+=1
        total+=d[z[count]]
        count+=1
    for j in np.arange(len(z)-1):
        kxk[z[j]][z[j+1]]+=1
    return kxk,kxd

def training_by_counting_var(K, D, x, z):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    init_probs=np.zeros(K)
    init_probs[z[0]]=1
    trans_probs,emission_probs=count_transitions_and_emissions_var(K,D,x,z)
    trans_probs/=np.sum(trans_probs,1).reshape(-1,1)
    emission_probs/=np.sum(emission_probs,1).reshape(-1,1)
    return hmm(init_probs,trans_probs,emission_probs)

#count emission probs
result=[]
for i in np.arange(5):
    four_test=np.delete(np.arange(5),[i])
    four_genome=[]
    four_ann=[]
    for j in four_test:
        four_genome.extend(data3[f"genome{str(j+1)}"])
        four_ann.extend(data3[f"true-ann{str(j+1)}"])
    result.append(training_by_counting_var(7,68,four_genome,four_ann))




data4=data3

def compute_w_log_var(model, x):
    k = len(model.init_probs)
    n = len(x)
    w = np.zeros((k, n))
    # Base case: fill out w[i][0] for i = 0..k-1
    # ...
    for i in np.arange(0,k):
        w[i,0]=np.log(model.init_probs[i])+np.log(model.emission_probs[i,x[0]])
    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    # ...
    j=1
    for i in np.arange(0,k):
        if i != 0:
            w[i,j]=np.log(0)
        else:
            w[i,j]=np.log(model.emission_probs[i,triplet_indices.index(x[j-d[i]+1:j+1])])+max(w[:,j-d[i]]+np.log(np.array(model.trans_probs)[:,i]))
    j=2
    for i in np.arange(0,k):
        if i in [2,3,5,6]:
            w[i,j]=np.log(0)
        else:
            w[i,j]=np.log(model.emission_probs[i,triplet_indices.index(x[j-d[i]+1:j+1])])+max(w[:,j-d[i]]+np.log(np.array(model.trans_probs)[:,i]))
    for j in np.arange(3,n):
        for i in np.arange(k):
            w[i,j]=np.log(model.emission_probs[i,triplet_indices.index(x[j-d[i]+1:j+1])])+max(w[:,j-d[i]]+np.log(np.array(model.trans_probs)[:,i]))
    return w

w=[]
for i in np.arange(5):
    w.append(compute_w_log_var(result[i],data4[f"genome{str(i+1)}"]))




def backtrack_log_var(model, x, w):
    n=len(x)
    zstar=[]
    total=n
    zstar.append(np.argmax(w[:,total-1]))
    total-=d[zstar[0]]
    count=1
    while total>0:
        zstar.append(np.argmax(w[:,total-1]+np.log(np.array(model.trans_probs)[:,zstar[count-1]])))
        total-=d[zstar[count]]
        count+=1
    zstar.reverse()
    return zstar


for i in np.arange(5):
    data4[f"pred-ann{str(i+1)}"]=backtrack_log_var(result[i],data4[f"genome{str(i+1)}"],w[i])




def translate_indices_to_annotations(ann):
    mapping = ['N', 'CCC', 'CCC', 'CCC', 'RRR', 'RRR', 'RRR']
    return ''.join(mapping[idx] for idx in ann)

def compute_accuracy(true_ann, pred_ann):
    # Check annoation length
    if len(true_ann) != len(pred_ann):
        print("ERROR: The lengths of two predictions are different")
        print("Expected %d, but found %d" % (len(true_ann), len(pred_ann)))  
    else:
        # Print stats
        compare_anns.print_all(true_ann, pred_ann)

accuracy=[]
for i in np.arange(5):
    accuracy.append(compute_accuracy(data[f"true-ann{str(i+1)}"],translate_indices_to_annotations(data4[f"pred-ann{str(i+1)}"])))

best_model=np.argmax(accuracy)
print(best_model)






#write fasta file
def write_fasta_file(filename,string):
    f=open(f"{filename}.fa","w")
    f.write(f">{filename}\n{string}\n")
    f.close()

data5=data4
for i in np.arange(6,11):
    #decode using best model
    data5[f"pred-ann{str(i)}"]=backtrack_log_var(result[best_model],data5[f"genome{str(i)}"],compute_w_log_var(result[best_model],data5[f"genome{str(i)}"]))
    #write to file
    write_fasta_file(f"pred-ann{str(i)}",translate_indices_to_annotations(data5[f"pred-ann{str(i)}"]))