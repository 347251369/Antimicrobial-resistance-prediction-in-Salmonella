import tensorflow as tf
import keras
from keras import Model, layers
from keras.optimizers import AdamW
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, average_precision_score
import pandas as pd
import os

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, x):
        x = self.token_emb(x)
        return x
    
class AttributeEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        positions = tf.expand_dims(positions, axis=0)
        return positions

def fs_cmim(X, y, n_selected_features, verbose=False):
    tmp = X.values
    idxs, maxCMI = fast_cmim(tmp, y, n_selected_features=n_selected_features, verbose=verbose)
    feats  = X.iloc[:, idxs].columns
    return feats, maxCMI

def fast_cmim(X, y, **kwargs):
    n_samples, n_features = X.shape
    is_n_selected_features_specified = False

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
        F = np.nan * np.zeros(n_selected_features)
    else:
        F = np.nan * np.zeros(n_features)

    t1 = np.zeros(n_features)
    m = np.zeros(n_features) - 1
    
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    
    verbose = False
    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']

    
    for k in range(n_features):
        # uncomment to keep track
        if verbose:
            counter = int(np.sum(~np.isnan(F)))
            if counter%100 == 0 or counter <= 1:
                print("F contains %s features"%(counter))
            
        if k == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F[0] = idx
            f_select = X[:, idx]

        if is_n_selected_features_specified:
            if np.sum(~np.isnan(F)) == n_selected_features:
                break

        sstar = -1000000 # start with really low value for best partial score sstar 
        for i in range(n_features):
            
            if i not in F:
                
                while (t1[i] > sstar) and (m[i]<k-1) :
                    m[i] = m[i] + 1
                    t1[i] = min(t1[i], cmidd(X[:,i], # feature i
                                             y,  # target
                                             X[:, int(F[int(m[i])])] # conditionned on selected features
                                            )
                               )
                if t1[i] > sstar:
                    sstar = t1[i]
                    F[k+1] = i
                    
    F = np.array(F[F>-100])
    F = F.astype(int)
    t1 = t1[F]
    return (F, t1)


from math import log
def hist(sx):
    d = dict()
    for s in sx:
        d[s] = d.get(s, 0) + 1

    return map(lambda z: float(z)/len(sx), d.values())

def elog(x):
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)

def cmidd(x, y, z):
    return entropyd(list(zip(y, z)))+entropyd(list(zip(x, z)))-entropyd(list(zip(x, y, z)))-entropyd(z)

# Discrete estimators
def entropyd(sx, base=2):
    return entropyfromprobs(hist(sx), base=base)

def entropyfromprobs(probs, base=2):
    return -sum(map(elog, probs))/log(base)

def midd(x, y):
    return -entropyd(list(zip(x, y)))+entropyd(x)+entropyd(y)

class WeightedAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self, initial_weights):
        super(WeightedAveragePooling1D, self).__init__()
        self.initial_weights = initial_weights

    def build(self, input_shapes):
        self.weightsOfFeatures = tf.Variable(
            initial_value=tf.constant(self.initial_weights, dtype=tf.float32),
            trainable=False)

    def call(self, inputs):
    	# (batch_size, vector_dim)
        weighted_sum = tf.matmul(self.weightsOfFeatures, inputs, transpose_a=True)
        return weighted_sum
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_weights": self.initial_weights
        })
        return config


AMR_list = ['AMP','AUG','AXO','CHL','FIS','FOX','TET']

for AMR in AMR_list:
    print("----------------- ",AMR," -----------------")

    X = pd.read_csv("../data_process/" + AMR +"/gene.csv" , index_col=0)
    y = pd.read_csv("../data_process/" + AMR + "/" + AMR + "_label.csv", index_col=0)

    for n_selected_features in range(10, 51, 10):
        print("----------------- ",n_selected_features," -----------------")

        rpt = 3

        all_acc = []
        all_pre = []
        all_spe = []
        all_sen = []
        all_f1 = []
        all_ap = []

        for t in range(rpt):

            fold_k = 3

            kf = StratifiedKFold(n_splits=fold_k, shuffle=True, random_state=t) # k折交叉验证

            acc = [0 for _ in range(fold_k)]
            pre = [0 for _ in range(fold_k)]
            spe = [0 for _ in range(fold_k)]
            sen = [0 for _ in range(fold_k)]
            f1 = [0 for _ in range(fold_k)]
            ap = [0 for _ in range(fold_k)]
            ct = 0

            for train_index, test_index in kf.split(X, y):
                x_train = X.iloc[train_index,0:n_selected_features]
                y_train = y.iloc[train_index,]
                x_test = X.iloc[test_index,0:n_selected_features]
                y_test = y.iloc[test_index]

                feats, maxCMI = fs_cmim(x_train, y_train, n_selected_features=n_selected_features)
                x_train = x_train.loc[:,feats]
                x_test = x_test.loc[:,feats]
                isSNP = x_train.columns.str.isnumeric()
                isGene = ~x_train.columns.str.isnumeric()

                maxCMI0 = [val for ind,val in enumerate(maxCMI) if isSNP[ind] == True]
                maxCMI1 = [val for ind,val in enumerate(maxCMI) if isGene[ind] == True]
                maxCMI = maxCMI0 + maxCMI1
                maxCMI = MinMaxScaler(feature_range=(0.1,1)).fit_transform(np.array(maxCMI).reshape(-1, 1))
                maxCMI_weights = (np.array(maxCMI) / np.sum(maxCMI)).reshape(-1,1)

                x_train0 = x_train.loc[:,isSNP]
                x_train1 = x_train.loc[:,isGene]
                x_test0 = x_test.loc[:,isSNP]
                x_test1 = x_test.loc[:,isGene]


                #--------------------model----------------------------------------------
                embed_dim = 16  # Embedding size for each token
                num_heads = 4  # Number of attention heads
                ff_dim = 32  # Hidden layer size in feed forward network inside transformer
                maxlen0 = x_train0.shape[1]
                vocab_size0 = 5
                maxlen1 = x_train1.shape[1]
                vocab_size1 = 2

                inputs0 = layers.Input(shape=(maxlen0,))
                inputs1 = layers.Input(shape=(maxlen1,))

                x0_TE = TokenEmbedding(vocab_size0, embed_dim)(inputs0)
                x1_TE = TokenEmbedding(vocab_size1, embed_dim)(inputs1)
                x_all_TE = layers.Concatenate(axis=1)([x0_TE, x1_TE])
                inputs_all = layers.Concatenate(axis=1)([inputs0, inputs1])
                x_all_PE = AttributeEmbedding(maxlen0+maxlen1, embed_dim)(inputs_all)
                x = layers.Add()([x_all_TE, x_all_PE])

                x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
                x = WeightedAveragePooling1D(maxCMI_weights)(x)

                outputs = layers.Dense(1, activation="sigmoid")(x)

                #------------------------------------------------------------------
                model = Model(inputs=[inputs0, inputs1], outputs=outputs)
                epoch = int(15 + n_selected_features/10 * 3)

                model.compile(optimizer=AdamW(), loss="binary_crossentropy")
                history = model.fit(
                    [x_train0, x_train1], y_train, batch_size=32, epochs=epoch, 
                    validation_data=([x_train0, x_train1], y_train), 
                )

                y_test_predict = model.predict([x_test0, x_test1]).reshape(-1,1)

                cm = confusion_matrix(y_test, y_test_predict > 0.5)
                TP = cm[1][1]
                TN = cm[0][0]
                FP = cm[0][1]  
                FN = cm[1][0]
                acc[ct] = (TP+TN)/(TP+TN+FP+FN)
                pre[ct] = (TP)/(TP+FP)
                spe[ct] = (TN)/(TN+FP)
                sen[ct] = (TP)/(TP+FN)
                f1[ct] = 2 * pre[ct] * sen[ct] / (pre[ct] + sen[ct]) 
                ap[ct] = average_precision_score(y_test, y_test_predict)
                ct += 1
            # print("----------------- ",t+1," -----------------")
            all_acc.append(np.mean(acc))
            all_pre.append(np.mean(pre))
            all_spe.append(np.mean(spe))
            all_sen.append(np.mean(sen))
            all_f1.append(np.mean(f1))
            all_ap.append(np.mean(ap))

        addr = "Results/"+AMR
        if not os.path.exists(addr):
            os.makedirs(addr)

        data = {'Precision':all_pre, 'Recall':all_sen, 'F1':all_f1}
        df = pd.DataFrame(data, index=[i for i in range(1,4)])
        df.to_csv(addr + "/RPNN_n"+str(n_selected_features)+".csv")
