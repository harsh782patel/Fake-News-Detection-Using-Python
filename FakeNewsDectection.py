# Basic set-up
import os
import numpy as np
import pandas as pd

# ML toolkits
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.extmath import density
from sklearn.pipeline import make_pipeline

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

input_path = '/workspaces/Fake-News-Detection-Using-Python/Datasets'
fake = pd.read_csv(os.path.join(input_path,'Fake.csv'))
real = pd.read_csv(os.path.join(input_path,'True.csv'))

display(fake.head())

display(real.head())

display(fake.info())
print('\n')
display(real.info())

display(fake.subject.value_counts())
print('\n')
display(real.subject.value_counts())

fake['label'] = 'fake'
real['label'] = 'real'

data = pd.concat([fake, real], axis=0)
data = data.sample(frac=1).reset_index(drop=True)
data.drop('subject', axis=1)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.25)
display(X_train.head())
print('\n')
display(y_train.head())

print("\nThere are {} documents in the training data.".format(len(X_train)))

my_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# fit the vectorizer and transform X_train into a tf-idf matrix,
# then use the same vectorizer to transform X_test
tfidf_train = my_tfidf.fit_transform(X_train)
tfidf_test = my_tfidf.transform(X_test)

tfidf_train

from sklearn.linear_model import PassiveAggressiveClassifier

pa_clf = PassiveAggressiveClassifier(max_iter=50)
pa_clf.fit(tfidf_train, y_train)

y_pred = pa_clf.predict(tfidf_test)

conf_mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat,
                      show_normed=True, colorbar=True,
                      class_names=['Fake', 'Real'])

accscore = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test,y_pred,pos_label='real')

print('The accuracy of prediction is {:.2f}%.\n'.format(accscore*100))
print('The F1 score is {:.3f}.\n'.format(f1score))

7# Dimensionality and density of features

print("Dimensionality (i.e., number of features): {:d}".format(pa_clf.coef_.shape[1]))
print("Density (i.e., fraction of non-zero elements): {:.3f}".format(density(pa_clf.coef_)))

# Sort non-zero weights
weights_nonzero = pa_clf.coef_[pa_clf.coef_!=0]
feature_sorter_nonzero = np.argsort(weights_nonzero)
weights_nonzero_sorted =weights_nonzero[feature_sorter_nonzero]

# Plot
fig, axs = plt.subplots(1,2, figsize=(9,3))

sns.lineplot(data=weights_nonzero_sorted, ax=axs[0])
axs[0].set_ylabel('Weight')
axs[0].set_xlabel('Feature number \n (Zero-weight omitted)')

axs[1].hist(weights_nonzero_sorted,
            orientation='horizontal', bins=500,)
axs[1].set_xlabel('Count')

fig.suptitle('Weight distribution in features with non-zero weights')

plt.show()

# Sort features by their associated weights
tokens = my_tfidf.get_feature_names_out()
tokens_nonzero = np.array(tokens)[pa_clf.coef_[0]!=0]
tokens_nonzero_sorted = np.array(tokens_nonzero)[feature_sorter_nonzero]

num_tokens = 10
fake_indicator_tokens = tokens_nonzero_sorted[:num_tokens]
real_indicator_tokens = np.flip(tokens_nonzero_sorted[-num_tokens:])

fake_indicator = pd.DataFrame({
    'Token': fake_indicator_tokens,
    'Weight': weights_nonzero_sorted[:num_tokens]
})

real_indicator = pd.DataFrame({
    'Token': real_indicator_tokens,
    'Weight': np.flip(weights_nonzero_sorted[-num_tokens:])
})

print('The top {} tokens likely to appear in fake news were the following: \n'.format(num_tokens))
display(fake_indicator)


print('\n\n...and the top {} tokens likely to appear in real news were the following: \n'.format(num_tokens))
display(real_indicator)

fake_contain_fake = fake.text.loc[[np.any([token in body for token in fake_indicator.Token])
                                for body in fake.text.str.lower()]]
real_contain_real = real.text.loc[[np.any([token in body for token in real_indicator.Token])
                                for body in real.text.str.lower()]]

print('Articles that contained any of the matching indicator tokens:\n')

print('FAKE: {} out of {} ({:.2f}%)'
      .format(len(fake_contain_fake), len(fake), len(fake_contain_fake)/len(fake) * 100))
print(fake_contain_fake)

print('\nREAL: {} out of {} ({:.2f}%)'
      .format(len(real_contain_real), len(real), len(real_contain_real)/len(real) * 100))
print(real_contain_real)

def FakeNewsDetection(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # vectorizer
    my_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = my_tfidf.fit_transform(X_train)
    tfidf_test = my_tfidf.transform(X_test)

    # model
    my_pac = PassiveAggressiveClassifier(max_iter=50)
    my_pac.fit(tfidf_train, y_train)
    y_pred = my_pac.predict(tfidf_test)

    # metrics
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat,
                          show_normed=True, colorbar=True,
                          class_names=['Fake', 'Real'])

    accscore = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test,y_pred,pos_label='real')

    print('The accuracy of prediction is {:.2f}%.\n'.format(accscore*100))
    print('The F1 score is {:.3f}.\n'.format(f1score))

    # Sort non-zero weights
    weights_nonzero = my_pac.coef_[my_pac.coef_!=0]
    feature_sorter_nonzero = np.argsort(weights_nonzero)
    weights_nonzero_sorted =weights_nonzero[feature_sorter_nonzero]

    # Sort features by their associated weights
    tokens = my_tfidf.get_feature_names_out()
    tokens_nonzero = np.array(tokens)[my_pac.coef_[0]!=0]
    tokens_nonzero_sorted = np.array(tokens_nonzero)[feature_sorter_nonzero]

    num_tokens = 10
    fake_indicator_tokens = tokens_nonzero_sorted[:num_tokens]
    real_indicator_tokens = np.flip(tokens_nonzero_sorted[-num_tokens:])

    fake_indicator = pd.DataFrame({
        'Token': fake_indicator_tokens,
        'Weight': weights_nonzero_sorted[:num_tokens]
    })

    real_indicator = pd.DataFrame({
        'Token': real_indicator_tokens,
        'Weight': np.flip(weights_nonzero_sorted[-num_tokens:])
    })

    print('The top {} tokens likely to appear in fake news were the following: \n'.format(num_tokens))
    display(fake_indicator)

    print('\n\n...and the top {} tokens likely to appear in real news were the following: \n'.format(num_tokens))
    display(real_indicator)

# Generate a copy of the "real news" dataset and remove headings f

real_copy = real.copy()
for i,body in real.text.items():
    if '(reuters)' in body.lower():
        idx = body.lower().index('(reuters)') + len('(reuters) - ')
        real_copy.text.iloc[i] = body[idx:]

real_copy.head()

# Create new data, and run the algorithm
data2 = pd.concat([fake, real_copy], axis=0)
data2 = data2.sample(frac=1).reset_index(drop=True)
data2.drop('subject', axis=1)

FakeNewsDetection(data2['text'], data2['label'])
