import torch
import torchaudio

# access the data in the dataset

# the yesno dataset in torchaudio features sixty recordings of one individual saying yes or no in Hebrew; with each recording being eight words long
# each item in the dataset is a tuple of the form (waveform, sample_rate, labels)
# you must set a `root` for the yesno dataset, which is where the training and testing dataset wille exist
# the other parameters are optional, with their default values shown
# here is some additional info on the other parameters

# `download`: If true, downloads the dataset from the internet and puts it in root directory
# if dataset is already downloaded, it will not download again
# `transform`: Using transforms on your data allows you to take it from its source state and transform it into data that's joined together, de-normalised, and ready for training
# each library in pytorch supports a growing list of transformation
# `target_transform`: A function / transform that takes in the target and transforms it

# let's access our yesno data
# a data point in yesno is a typle (waveform, sample_rate, labels) where labels is a list of integers with 1 for yes and 0 for no
yesno_data_train_set = torchaudio.datasets.YESNO("./", download = True)

# pick data point number 3 to see an example of the yesno_data
n = 3
waveform, sample_rate, labels = yesno_data_train_set[n]
print(f"waveform: {waveform}, sample_rate: {sample_rate}, labels: {labels}")

# loading the data

# now the we have access to the dataset, we must pass it through `torch.utils.data.DataLoader`
# the `DataLoader` combines the dataset and a sampler, returning an iterable over the dataset
data_loader = torch.utils.data.DataLoader(yesno_data_train_set, batch_size = 1, shuffle = True)

# iterate over the data

# our data is now iterable using the `data_loader`
# this will be necessary when we begin training our model
# you will notice that now each data entry in the `data_loader` object is converted to a tensor containing tensors representing our waveform, sample_rate, and labels
for each_data in data_loader:
    print(f"Data: {each_data}")
    print(f"waveform: {each_data[0]}, sample_rate: {each_data[1]}, labels: {each_data[2]}")
    break

# visualise the data

import matplotlib.pyplot as plt

print(each_data[0][0].numpy())
plt.figure()
plt.plot(waveform.t().numpy())
