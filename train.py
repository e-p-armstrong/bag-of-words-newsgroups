# I will approach my own machine learning iteratively: starting trying to do a bunch, and doing it poorly; and then, through correction and instruction, getting better at it as time goes on.


# Note that since the point of this project was to familiarize myself with the Pytorch library (as opposed to the huggingface Trainer API, which I'd used prior), it's covered with rough notes, annotations, and comments so that it can serve as a reference for me in the future. It's not... clean. And the code isn't exactly robust either. But hey it works and gets 84% accuracy after 1 epoch! So there's that.
import sklearn.datasets
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm

# Check for mps, because I just remembered my mac has a GPU
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")


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
        self.data = texts
        self.vectorizer = vectorizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector = torch.tensor(self.vectorizer.transform([self.data[idx]]).todense(), dtype=torch.float32).squeeze() # torch.tensor() to convert the matrix into a torch tensor for training
        label = torch.tensor(self.targets[idx], dtype=torch.long).squeeze()
        return vector, label

# data_torch = TwentyNewsgroupsDataset(dataset)

# train_dataset, test_dataset = random_split(data_torch,[cutoff,test_len])

vectorizer = CountVectorizer().fit(train)

torch_train = TwentyNewsgroupsDataset(train,dataset["target"][cutoff:],vectorizer)
torch_test = TwentyNewsgroupsDataset(test,dataset["target"][:cutoff],vectorizer)


# Hyperparameters
input_size = len(vectorizer.get_feature_names_out()) # useful method for nlp presumably
num_classes = len(dataset["target_names"])
num_epochs = 1 # Setting for one for now while I test my code
batch_size = 32
learning_rate = 1e-4 # is this too large or too small?

train_dataloader = DataLoader(torch_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(torch_test, batch_size=batch_size, shuffle=True, drop_last=True)

train_iter = iter(train_dataloader)


# print("\n\n\nBatch:")
# print(next(train_iter))

print("\nasserting that the data is not shuffled before the targets are aligned:\n\n")

assert (np.array(train) == np.array(dataset.data[cutoff:])).all() # These pass
assert (np.array(test) == np.array(dataset.data[:cutoff])).all() # These pass
print("passed")

# personal policy: there's no need to reinvent the wheel if I just want to learn how to drive, so I'm going to use libraries for tertiary things like tokenization and whatnot wherever possible (so long as they would be something I'd possibly use in a professional setting) because the goal is to get familiar with a more in-depth machine learning library (Pytorch) as opposed to huggingface's trainer api.
# So, no custom bag of words tokenizer today.

class DeepLClassifier(nn.Module):
    def __init__(self,input_size,num_classes):
        super(DeepLClassifier, self).__init__()
        # Correct me if I'm wrong, but I THINK the best way to do architecture is wide -> narrow (in terms of parameter count). Or do I have it backwards? # Note to self: this has been confirmed as a standard
        self.input_layer = nn.Linear(input_size,4096)
        self.hidden_layer_1 = nn.Linear(4096, 2048)
        self.gelu_1 = nn.GELU() # I see now. GELU transforms the preceding linear layer, mapping its values across a GELU; but it is not its own layer. The linear learns like a GELU because the end outputted result of linear * weights -> GELU transform is the same as GELU * weights -> output. Remember how things like sigmoid basically take the line equation and shove it into the exponent of e, right? This is basically that. But GELU.
        # ^ I think that's right?
        self.hidden_layer_2 = nn.Linear(2048, 1024)
        self.gelu_2 = nn.GELU()
        self.hidden_layer_3 = nn.Linear(1024, 512)
        self.gelu_3 = nn.GELU()
        self.pre_softmax_out = nn.Linear(512,num_classes) # is it right to have 2 outputs for 2 classes? Or should I have just one and make it so that that one class being "0" or "1" represents the two classes in the classification?
        # self.out = nn.Softmax(dim=1) # note to self: if output tensor has shape [batch_size, num_classes], use dim=1 s that softmax is applied to each batch

    def forward(self,x):
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = self.gelu_1(x)
        x = self.hidden_layer_2(x)
        x = self.gelu_2(x)
        x = self.hidden_layer_3(x)
        x = self.gelu_3(x)
        x = self.pre_softmax_out(x)
        # x = self.out(x)
        return x.squeeze()

model = DeepLClassifier(input_size,num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Parameter count:")
p = 0
for layer in model.parameters():
    p += layer.numel()

print(p)

model.to(device)

# We define evaluate before the training loop so that we can calculate the loss after each bit of evaluation
# Egads it's like I'm building the bits of the huggingface trainer API that I took for granted
def evaluate(dataloader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad(): # TODO get GPT to explain this line, or rather how it works I know what it does
        for data, targets in tqdm(dataloader, desc=f"Evaluating..."):
            data, targets = data.to(device), targets.to(device) # This can be done because both vector and label are torch tensors
            # lots of debugs for learning:
            # print("data shape", data.shape)
            output = model(data)
            # print("output shape",output.shape)
            # print("output shape",targets.shape)
            loss = criterion(output,targets)
            # print(loss.item())
            # print(loss)
            total_loss += loss.item()

            _, predictions = output.max(1) # returns two tensors: the maximum values across the second dimension of the output (so, the highest prediction logits) and the indices of those predictions (equal to the predicted class).
            # print(predictions.shape)
            total_correct += (predictions == targets).sum().item() # Remember that dimensions are not mentioned at all here, so you don't need to specify a dimension to sum across; my guess is that it just uses the last one, ie the one with values.
            # print(data.shape)
    average_loss = total_loss / len(dataloader) # remember, Evan, dataloader has a __len__ property too, you can get the length of a dataloader's wrapped dataset
    accuracy = 100.0 * total_correct / len(dataloader.dataset)
    return average_loss, accuracy

# Evaluate the untrained model, just to make sure that the function works and also as a point of reference:
print("initial loss, accuracy")
print(evaluate(test_dataloader,model,criterion))

# tqdm carriage returns, that's why it appears on ew lines all the time


for e in range(num_epochs):
    model.train()
    for batch_idx, (data,targets) in enumerate(tqdm(train_dataloader, desc=f"Epoch: {e+1}")):
        data, targets = data.to(device), targets.to(device) # This can be done because both vector and label are torch tensors
        # print(data.shape)
        optimizer.zero_grad() # apparently none_grad might be better, look into it me
        scores = model(data)
        # print(data[0])
        # print(data.shape)
        # print(scores[1])
        # print(scores.shape)
        # print(targets)
        # print(targets.shape)
        loss = criterion(scores,targets) # calculate CrossEntropy between the predicted scores and the targets
        loss.backward()
        optimizer.step()
    train_loss, train_acc = evaluate(train_dataloader,model,criterion)
    test_loss, test_acc = evaluate(test_dataloader,model,criterion)
    print(f"Epoch {e+1}:\ntrain_loss: {train_loss:.4f}\ntrain_acc: {train_acc:.4f}\test_loss: {test_loss:.4f}\test_acc: {test_acc:.4f}")

torch.save(model.state_dict(),"bow.pt") # load by importing the model class and torch.load_state_dict
# next time: parseargs, more logging, checkpoints and loading/resuming, different device types (cuda, cpu)... and perhaps more new things to practice, as recommended by GPT-4

# 1. Why no torch.no_grad - it should've been done, well it works without it but torch.no_grad stops needless computation and storage of gradients in memory
# 2. Why do testing in batches? Does that even work? - to save memory by remembering less at once
# 3. What the hell is loss.item()? What are the important parts of the loss object - 
# 4. Explain the _, predictions = scores.max(1) line

# 5. is .max() a pytorch method? I can't find it in the docs.
# 6. In a line like (predictions == targets).sum().item(), is this pytorch-specific or is it a pure-python line, ie does calling array1 == array2 produce a filtered array with only elements that appear in both? Which can then have.sum() and other such methods called on it? Or just tensors?
# 7. Difference between len(dataloader) and len(dataloader.dataset)
# 8. Evaluate this evaluate code please.

# maybe add a parameter to the command line args of the "real deal" script that reduces/increases the parameter count of the trained model by a scalar? considering the need to rapidly switch between testing the pipeline, and building large models, this could be useful...



# GPT conversations that helped build this
# https://chat.openai.com/share/feeed1f9-dba6-4a2b-8c27-8e90bcf7893e
# https://chat.openai.com/share/2bb7c53e-a70f-491d-a011-fad42a2e5825
# https://chat.openai.com/share/ef55958e-e4eb-4ed7-9f17-68b810171479
# https://chat.openai.com/share/c9b4c188-f0d3-46fc-968d-145144f924b7
# https://chat.openai.com/share/d5d467f7-eaf8-4f8d-8b49-293c1aea6cc0
# https://chat.openai.com/share/2766b333-2924-4d8b-b600-fb340ce9e7dd
# https://chat.openai.com/share/20c40930-884a-4fdf-b583-88023706d57d
# https://chat.openai.com/share/0c668c1e-6c17-4117-9833-9d3daa85fbed
# https://chat.openai.com/share/3b41287a-bd84-4f08-823a-ce9c0c0e7f03