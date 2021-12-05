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

#write fasta file
def write_fasta_file(filename,string):
    f=open(f"filename.fa","w")
    f.write(f">{filename}\n{string}\n")
    f.close()

#translate from letters to indices
def translate_observations_to_indices(obs):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[symbol] for symbol in obs]

#translate from annotations to indices
def translate_annotations_to_indices(obs):
    mapping = {'N': 0, 'C': 7, 'R': 8}
    return [mapping[symbol] for symbol in obs]

triplet_indices=[0,1,2,3]
triplet_indices+=[list(x) for x in itertools.combinations_with_replacement([0,1,2,3],3)]

#group codons
def group_codons(ann):
    i=1
    while i <= len(ann):
        if ann[i-1:i+1]==[0,0]:
            pass
        elif ann[i-1:i+1]==[1,7] or ann[i-1:i+1]==[2,7]:
            ann[i]=2
            ann.pop(i+1)
            ann.pop(i+2)
        elif ann[i-1:i+1]==[4,8] or ann[i-1:i+1]==[5,8]:
            ann[i]=5
            ann.pop(i+1)
            ann.pop(i+2)
        elif ann[i-1:i+1]==[0,7]:
            ann[i]=1
            ann.pop(i+1)
            ann.pop(i+2)
        elif ann[i-1:i+1]==[0,8]:
            ann[i]=4
            ann.pop(i+1)
            ann.pop(i+2)
        elif ann[i-1:i+1]==[2,0]:
            ann[i-1]=3
        elif ann[i-1:i+1]==[5,0]:
            ann[i-1]=6
        elif ann[i-1:i+1]==[2,8]:
            ann[i-1]=3
            ann[i]=4
            ann.pop(i+1)
            ann.pop(i+2)
        elif ann[i-1:i+1]==[5,7]:
            ann[i-1]=5
            ann[i]=1
            ann.pop(i+1)
            ann.pop(i+2)
        i+=1
    return ann
"""
#translate back to observations string
def translate_indices_to_observations(indices):
    mapping = ['A', 'C', 'G', 'T']
    return ''.join(mapping[idx] for idx in indices)
"""
"""            genome[i]=triplet_indices.index([genome[i],genome[i+1],genome[i+2]])+4
            genome.pop(i+1)
            genome.pop(i+2)"""

#translate states back to annotations string
def translate_indices_to_annotations(ann):
    mapping = ['N', 'C', 'C', 'C', 'R', 'R', 'R']
    return ''.join(mapping[idx] for idx in ann)

#create a class
class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

#validate the model works
def validate_hmm(model):
    return np.allclose(np.sum(model.init_probs),1) and np.allclose(np.sum(model.trans_probs,1),1) and np.allclose(np.sum(model.emission_probs,1),1) and all(np.reshape(model.init_probs,-1) <= 1) and all(np.reshape(model.trans_probs,-1) <= 1) and all(np.reshape(model.emission_probs,-1) <= 1) and all(np.reshape(model.init_probs,-1) >= 0) and all(np.reshape(model.trans_probs,-1) >= 0) and all(np.reshape(model.emission_probs,-1) >= 0)

#optimal path probability
def opt_path_prob_log(w):
    return max(w[:,-1])

#compute the joint log probability
def joint_prob_log(model, x, z):
    output=np.log(model.init_probs[z[0]])
    for i in np.arange(1,len(x)):
        output+=np.log(model.trans_probs[z[i-1]][z[i]])
    for j in np.arange(len(x)):
        output+=np.log(model.emission_probs[z[j]][x[j]])
    return output

def compute_w_log(model, x):
    k = len(model.init_probs)
    n = len(x)
    w = np.zeros((k, n))
    # Base case: fill out w[i][0] for i = 0..k-1
    # ...
    for i in np.arange(0,k):
        w[i,0]=np.log(model.init_probs[i])+np.log(model.emission_probs[i][x[0]])
    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    # ...
    for j in np.arange(1,n):
        for i in np.arange(0,k):
            w[i,j]=np.log(model.emission_probs[i][x[j]])+max(w[:,j-1]+np.log(np.array(model.trans_probs)[:,i]))
    return w

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
    for j in np.arange(1,n):
        for i in np.arange(0,k):
            w[i,j]=np.log(model.emission_probs[i,triplet_indices.index(x[j-d[i]+1:j+1])])+max(w[:,j-d[i]]+np.log(np.array(model.trans_probs)[:,i]))
    return w

def backtrack_log(model, x, w):
    n=w.shape[1]
    zstar=[None]*n
    zstar[n-1]=np.argmax(w[:,-1])
    for i in np.arange(1,n):
        zstar[n-1-i]=np.argmax(np.log(model.emission_probs[zstar[n-i],x[n-i]])+w[:,n-1-i]+np.log(np.array(model.trans_probs)[:,zstar[n-i]]))
    return zstar

def backtrack_log_var(model, x, w):
    n=len(x)
    zstar=[]
    total=n
    zstar.append(np.argmax(w[:,total-1]))
    total-=d[zstar[0]]
    count=1
    while total>0:
        #check this line
        zstar.append(np.argmax(np.log(model.emission_probs[zstar[count-1],triplet_indices.index(x[total-d[zstar[count-1]]:total])])+w[:,total-1]+np.log(np.array(model.trans_probs)[:,zstar[count-1]])))
        total-=zstar(count)
        count+=1
    zstar.reverse()
    return zstar

def count_transitions_and_emissions(K, D, x, z):
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    kxk=np.zeros((K,K))
    kxd=np.zeros((K,D))
    for i in np.arange(len(x)):
        kxd[z[i]][x[i]]+=1
    for j in np.arange(len(z)-1):
        kxk[z[j]][z[j+1]]+=1
    return kxk,kxd

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

def training_by_counting(K, D, x, z):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    init_probs=np.zeros(K)
    init_probs[z[0]]=1
    trans_probs,emission_probs=count_transitions_and_emissions(K,D,x,z)
    trans_probs/=np.sum(trans_probs,1).reshape(-1,1)
    emission_probs/=np.sum(emission_probs,1).reshape(-1,1)
    return hmm(init_probs,trans_probs,emission_probs)

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
"""
def ac(z,zstar):
    tp=
    fp=
    fn=
    tn=
    acp=(1/4)*((tp/(tp+fn))+(tp/(tp+fp))+(tn/(tn+fp))+(tn/(tn+fn)))
    out=(acp-0.5)*2
    return out
"""
def compute_accuracy(true_ann, pred_ann):
    # Check annoation length
    if len(true_ann) != len(pred_ann):
        print("ERROR: The lengths of two predictions are different")
        print("Expected %d, but found %d" % (len(true_ann), len(pred_ann)))  
    else:
        # Print stats
        compare_anns.print_all(true_ann, pred_ann)

#build model
d=[1,3,3,3,3,3,3]

#load into data
data = {}
for i in np.arange(1,11):
    temp={}
    temp[f'genome{str(i)}'] = translate_observations_to_indices(read_fasta_file(f'genome{str(i)}.fa')[f'genome{str(i)}'])
    data.update(temp)
for i in np.arange(1,6):
    temp={}
    temp[f'true-ann{str(i)}'] = translate_annotations_to_indices(read_fasta_file(f'true-ann{str(i)}.fa')[f'true-ann{str(i)}'])
    data.update(temp)

#hmm([1,0,0,0,0,0,0], np.zeros((7,7)),np.zeros((7,66)))

#potetial problem - [0] [1] [2] [3] in a list like [0,0,0]?

#group data into codons
for i in np.arange(1,6):
    data[f"true-ann{str(i)}"]=group_codons(data[f"true-ann{str(i)}"])    

#5-fold check
#count emission probs
result=[]
for i in np.arange(1,6):
    four_test=np.delete(np.arange(5),[i-1])
    four_genome=[]
    four_ann=[]
    for j in four_test:
        four_genome.extend(data[f"genome{str(j+1)}"])
        four_ann.extend(data[f"true-ann{str(j+1)}"])
    result.append(training_by_counting_var(7,66,four_genome,four_ann))

#compute best model
accuracy=[]
for i in np.arange(5):
    data[f"pred-ann{str(i+1)}"]=backtrack_log_var(result[i],data[f"genome{str(i+1)}"],compute_w_log_var(result[i],data[f"genome{str(i+1)}"]))
    accuracy.append(compute_accuracy(data[f"true-ann{str(i+1)}"],data[f"pred-ann{str(i+1)}"]))

best_model=np.argmax(accuracy)
print(best_model)

for i in np.arange(6,11):
    #decode using best model
    data[f"pred-ann{str(i)}"]=backtrack_log_var(result[best_model],data[f"genome{str(i)}"],compute_w_log_var(result[best_model],data[f"genome{str(i)}"]))
    accuracy.append(compute_accuracy(data[f"true-ann{str(i)}"],data[f"pred-ann{str(i)}"]))
    #write to file
    write_fasta_file(f"pred-ann{str(i)}",data[f"pred-ann{str(i)}"])


#extra - decode when all five used

"""