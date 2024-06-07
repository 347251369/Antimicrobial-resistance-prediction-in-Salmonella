from qiskit import QuantumCircuit, Aer, QuantumRegister, ClassicalRegister,execute
import math
import matplotlib.pyplot as plt
from qiskit.circuit.library.standard_gates import XGate


def setSample(sample):
	output = []
	mmap = {
		'A':0,
		'T':1,
		'G':2,
		'C':3
	}
	for i in sample:
		output.append(mmap.get(i))
	return output


# Set number of features (log2)
features_size = 2
# Set number of values of a feature (log2)
values_size = 2
# Set sample
sample1 = 'AAGT'
sample1 = setSample(sample1)
sample2 = 'TATA'
sample2 = setSample(sample2)



# Create a Quantum Circuit
sample_qb_num = features_size + values_size
quan = QuantumRegister(sample_qb_num * 2 + 1, name='q')
clas = ClassicalRegister(1)
cir = QuantumCircuit(quan, clas)

#### encoding
qb_feature1 = [i for i in range(0, features_size)]
qb_val1 = [i for i in range(features_size, sample_qb_num)]

qb_feature2 = [i for i in range(sample_qb_num, sample_qb_num + features_size)]
qb_val2 = [i for i in range(sample_qb_num + features_size, sample_qb_num*2)]

for i in qb_feature1:
    cir.h(i)
for i in qb_feature2:
    cir.h(i)

for i in range(len(sample1)):
    if sample1[i] != 0:
        b = features_size
        t = sample1[i]
        while t != 0:
            if t & 1 == 1:
                cir.append(XGate().control(features_size, ctrl_state=i), qb_feature1 + [b])
            b += 1
            t >>= 1

    if sample2[i] != 0:
        b = sample_qb_num + features_size
        t = sample2[i]
        while t != 0:
            if t & 1 == 1:
                cir.append(XGate().control(features_size, ctrl_state=i), qb_feature2 + [b])
            b += 1
            t >>= 1
cir.barrier()
#### Calculating similarity
swaptest_aux = sample_qb_num * 2
cir.h(swaptest_aux)
for i in range(sample_qb_num):
    cir.cswap(swaptest_aux, i, i + sample_qb_num)
cir.h(swaptest_aux)
cir.barrier()

# Measurement
cir.measure(swaptest_aux, 0)

# run circuit
simulator = Aer.get_backend('aer_simulator')
job = execute(cir, simulator, shots=10000)
result = job.result()

# get results
counts = result.get_counts(cir)
print(counts)
if (counts.get('0') is None):
    counts['0'] = 0
if (counts.get('1') is None):
    counts['1'] = 0

p = counts['1']/(counts['1']+counts['0'])
print(p)

dis = len(sample1)
if p <= 0.5:
    dis = (2**features_size) * math.sqrt(1 - 2*p)
print(dis)

# Draw the circuit
cir.draw(output='mpl')
plt.savefig('./circuit.tif',dpi=100)
plt.show()
