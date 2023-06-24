# I will approach my own machine learning iteratively: starting trying to do a bunch, and doing it poorly; and then, through correction and instruction, getting better at it as time goes on.

import sklearn.datasets
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset


# Note to self: lots of numpy methods have an out= field that lets you specify an array to write the result to.

# Data loading and inspection
dataset = sklearn.datasets.fetch_20newsgroups()

# inspect data
print(dataset.keys())
# dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR']) # for reference
data = dataset["data"]
print("Examples of data: \n\n---------------\n")
print(data[0:2])
print("length: ",len(data))
print("type: ",type(data))
print(dataset["target_names"])
print(dataset["target"])
# print(dataset["DESCR"])

print("\n\n\n-----------------Viewing done---------------\n\n\n")






# Train/test split
cutoff = round(len(data)*0.2)
# Question: does this create copies of the sliced bits of the bag_of_words array, or are train and test simply referencing different parts of bag_of_words?

train = data[cutoff:]
test = data[:cutoff]
print(f"train len: {len(train)} vs {len(data)} total examples")
print(f"train {len(train)} + test {len(test)} = {len(train) + len(test)} compared to {len(data)}")

# Note to self: fit_transform creates the matrix of words from a set of documents; transform() applies an already-created matrix to a set of documents.


# Actually screw it, I'm using dataloader because I think it'd be annoying and ugly to get batch_size elements from the dataset each time and keep track of where I am in the process.
class TwentyNewsgroupsDataset(Dataset):
    def __init__(self,texts,targets, vectorizer):
        self.targets = targets
        self.data = texts # converts the sparse matrix to a dense one... I think.
        self.vectorizer = vectorizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector = torch.tensor(self.vectorizer.transform([self.data[idx]]).todense()) # torch.tensor() to convert the matrix into a torch tensor for training
        label = self.targets[idx]
        return vector, label

# data_torch = TwentyNewsgroupsDataset(dataset)

# train_dataset, test_dataset = random_split(data_torch,[cutoff,test_len])

vectorizer = CountVectorizer().fit(train)

# Debug inspection of vector shapes vs length of original array
train_vec = vectorizer.transform(train) # Does the sklearn CountVectorizer work even on non-sklearn datastructures like numpy arrays?
test_vec = vectorizer.transform(test)
print("train_vec",train_vec.shape,len(train))
print("test_vec", test_vec.shape, len(test))

torch_train = TwentyNewsgroupsDataset(train,dataset["target"][cutoff:],vectorizer)
torch_test = TwentyNewsgroupsDataset(test,dataset["target"][:cutoff],vectorizer)


# Hyperparameters
input_size = train_vec.shape[1]
num_classes = len(dataset["target_names"])
num_epochs = 3
batch_size = 3
learning_rate = 1e-4 # is this too large or too small?

train_dataloader = DataLoader(torch_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(torch_test, batch_size=batch_size, shuffle=True)

train_iter = iter(train_dataloader)


print("\n\n\nBatch (a tensor of values followed by a tensor of targets, presumably, please confirm my understanding on this GPT-4):  ")
print(next(train_iter))

# personal policy: there's no need to reinvent the wheel if I just want to learn how to drive, so I'm going to use libraries for tertiary things like tokenization and whatnot wherever possible (so long as they would be something I'd possibly use in a professional setting) because the goal is to get familiar with a more in-depth machine learning library (Pytorch) as opposed to huggingface's trainer api.
# So, no custom bag of words tokenizer today.

# class DeepLClassifier(nn.Module):
#     def __init__(self,input_size,num_classes):
#         super(DeepLClassifier, self).__init__()
#         # Correct me if I'm wrong, but I THINK the best way to do architecture is wide -> narrow (in terms of parameter count). Or do I have it backwards?
#         self.input_layer = nn.Linear(input_size,4096)
#         self.hidden_layer_1 = nn.Linear(4096, 2048)
#         self.gelu_1 = nn.GELU() # I don't quite understand whether this GELU replaces the preceding linear, or whether it's its own hidden layer, or what.
#         self.hidden_layer_2 = nn.Linear(2048, 1024)
#         self.gelu_2 = nn.GELU()
#         self.hidden_layer_3 = nn.Linear(1024, 512)
#         self.gelu_3 = nn.GELU()
#         self.pre_softmax_out = nn.Linear(512,num_classes)
#         self.out = nn.Softmax(dim=1) # note to self: if output tensor has shape [batch_size, num_classes], use dim=1 s that softmax is applied to each batch

#     def forward(self,x):
#         x = self.input_layer(x)
#         x = self.hidden_layer_1(x)
#         x = self.gelu_1(x)
#         x = self.hidden_layer_2(x)
#         x = self.gelu_2(x)
#         x = self.hidden_layer_3(x)
#         x = self.gelu_3(x)
#         x = self.pre_softmax_out(x)
#         x = self.out(x)
#         return x

# model = DeepLClassifier(input_size,num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# for e in range(num_epochs):
#     for 



# GPT conversations that helped build this
