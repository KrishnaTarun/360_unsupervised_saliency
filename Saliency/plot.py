import pickle
with open('./train_plot.pkl', 'rb') as handle:
      example_dict = pickle.load(handle)
# print(example_dict)


import numpy as np
import matplotlib.pyplot as plt


print(example_dict.keys())

epochs = range(0,len(example_dict['train_losses']),1)
plt.plot(epochs, example_dict['train_losses'], 'g', label='Training loss')
plt.plot(epochs, example_dict['val_losses'], 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.xlim(0,20)
plt.show()
print(np.argmin(example_dict['val_losses']))
# Create some mock data
#t = np.arange(0.01, 10.0, 0.01)
# data1 = example_dict['nss_accuracy']
# data2 = example_dict['nss_validate']

# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('training NSS', color=color)
# ax1.plot(epochs, data1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('validation NSS', color=color)  # we already handled the x-label with ax1
# ax2.plot(epochs, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig = plt.figure()
# plt.show()
plt.savefig('./loss.png')
