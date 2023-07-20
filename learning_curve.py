import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# data
train_data = np.load('sc_data.npy')
n_facility = 3
np.random.shuffle(train_data)
X_train = train_data[:, 0:n_facility*2]
tc_train = train_data[:, n_facility*2+2]
sl_train = train_data[:, n_facility*2+3]

'''learning_curve'''
rfr_pf = RandomForestRegressor()
train_sizes, train_scores, test_scores = learning_curve(
    estimator=rfr_pf, X=X_train, y=sl_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 1.03])
plt.tight_layout()
plt.show()
