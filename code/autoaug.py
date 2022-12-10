from datasets import load_dataset
import pandas as pd
import random
import numpy as np
import nltk
from nltk.corpus import wordnet
nltk.download('popular')
import matplotlib
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def word_pos(word):
    pos = nltk.pos_tag([word])[0][1]
    if pos in ['NN', 'NNP', 'NNS']:
        return 'N'
    elif pos in ['VB','VBD','VBG','VBN','VBP','VBZ']:
        return 'V'
    elif pos in ['JJ', 'JJR', 'JJS']:
        return 'ADJ'
    elif pos == 'MD':
        return 'MD'
    elif pos in ['RB', 'RBR', 'RBS']:
        return 'ADV'
    return None


def find_anonym(word):
    try:
        return wordnet.synsets(word)[0].lemmas()[0].antonyms()[0].name()
    except:
        return None


def find_synonym(word):
    try:
        syn_lst = wordnet.synsets(word)[0].lemmas()
        length = min(len(syn_lst),5)
        syn = syn_lst[0].name()
        i = 1     
        # if synonym is the same as itself, iterate through the top syn until we find a distinct word
        while word == syn and i < length:
            syn = wordnet.synsets(word)[0].lemmas()[i].name()
            i += 1
        return None if syn == word else syn
    except:
        return None

def identify_spurious_word(pipeline, explainer, sentence):
    """
    Return the spurious word in the given sentence.
    If the confidence in prediction is high (prob >=0.8), return the most influential word given by LIME explainer.
    If none of the words is significant (importance >=0.1, without which the classifier will just randomly guess), return None. 
    """
    proba = pipeline.predict_proba([sentence])
    pred = max(proba[0,0], proba[0,1])
    if pred >= 0.80:
        # if random guess, prob should be p(neutral) = 0.66 for snli
        explanation = explainer.explain_instance(sentence, 
                                                pipeline.predict_proba, 
                                                num_features=5)
        key_word, importance = explanation.as_list()[0]
        if np.abs(importance) >= 0.1:
            return key_word

    return False

def lime_vis_example(explainer, text_sample, pipeline):
    # reference: https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html
    matplotlib.rcParams['figure.dpi']=300
    explanation = explainer.explain_instance(text_sample, 
                                            pipeline.predict_proba, 
                                            num_features=10)
    explanation.show_in_notebook(text=True)

def get_augment_samples(pipeline, X_train, y_train, X_premise):
    """
    Given data, first determine if it contains spurious words, then replace the word based on its POS and modify its label.
        POS=N: replace with synonum, flip lr_label, aug_label unchanged
        POS=V/MD/ADJ/ADV: replace with anonym, flip aug_label, lr_label unchanged
    Return:
        match_premise: premise of augmented data
        new_example: augmented data
        aug_label: label of augmented data for the original problem
        lr_label: label of augmented data to verify the removal of spurious correlation in function decorrelation_analysis()
        explainer: LIME explainer of word importance
    """
    
    class_names = ['Entailment/Neutral', 'Contradiction']
    explainer = LimeTextExplainer(class_names=class_names)

    new_example = []
    match_premise = []
    lr_label = []
    aug_label = []

    for i in range(len(X_train)):
        sentence = X_train[i]
        label = y_train[i]
        word = identify_spurious_word(pipeline, explainer, sentence)
        if word:
            pos = word_pos(word)

            if pos == 'N':
                rep = find_synonym(word)
            # elif pos in ['V', 'MD']:
            #     rep = 'not ' + word if pos == 'V' else word + ' not'
            elif pos in ['ADJ', 'ADV']:
                rep = find_anonym(word)
            else:
                rep = None

            if rep is not None:
                new_example.append(sentence.replace(word, rep))
                match_premise.append(X_premise[i])
                if pos == 'N':
                    lr_label.append(1-label)
                    aug_label.append(label) 
                else:
                    lr_label.append(label)
                    aug_label.append(1-label)

    return match_premise, new_example, aug_label, lr_label, explainer

def augment_data(train_data):
    X_premise = train_data['premise']
    X_train = train_data['hypothesis']
    
    # Change to a binary classification
    # SNLI [1]:entailment(0) neutral(1) VS. [0]:contradiction (2)
    y_train = [1 if x == 1 or x == 0 else 0 for x in train_data['label']]

    tfidf_vec = TfidfVectorizer(min_df = 10, token_pattern = r'[a-zA-Z]+')
    pipeline = Pipeline([ ('vectorizer',tfidf_vec), ('clf', LogisticRegression(solver='saga'))])
    pipeline.fit(X_train, y_train)
    
    match_premise, new_example, aug_label, _, _ = get_augment_samples(pipeline, X_train, y_train, X_premise)
    X_train_premise_aug = X_premise + match_premise
    X_train_hypo_aug = X_train + new_example
    y_train_aug = y_train + aug_label

    train_data_aug = {'premise': X_train_premise_aug, 'hypothesis': X_train_hypo_aug, 'label': y_train_aug}

    return train_data_aug


def transform_label(data):
    y = [1 if x == 1 or x == 0 else 0 for x in data['label']]
    return {'premise': data['premise'], 'hypothesis': data['hypothesis'], 'label': y}


def decorrelation_analysis(data_size=5000, vis=False):
    """
    1. Use logistic regression to determine whether there is a spurious correlation between hypothesis and labels.
        Expect to see results better than random guess after training.
    2. Perform data augmentation.
    3. Retrain logistic regression on original + augmented data to find whether spurious correlation is mitigated.
        Expect to see a slight drop in train acc and increase in val acc. 
    4. Visualize the top spurious words in training data based on coefficient + Visualize on LIME explainer example.
    """
    
    metadata = load_dataset('snli')
    _, val_data, train_data = metadata.values()
    train_hypothesis = train_data['hypothesis']
    train_label = [1 if x == 1 or x == 0 else 0 for x in train_data['label']]
    X_train, X_test, y_train, y_test = train_test_split(train_hypothesis[:data_size], train_label[:data_size], test_size=0.33, random_state=42)
    # Random guess acc: ~66%
    print('Accuracy (random guess):', sum(y_train)/len(y_train))

    tfidf_vec = TfidfVectorizer(min_df = 10, token_pattern = r'[a-zA-Z]+')
    X_train_bow = tfidf_vec.fit_transform(X_train)
    X_test_bow = tfidf_vec.transform(X_test)

    model_lg = LogisticRegression(solver='saga')
    model_lg.fit(X_train_bow, y_train)

    model_lg_acc_train = cross_val_score(estimator=model_lg, X=X_train_bow, y=y_train, cv=5, n_jobs=-1)
    print('Train avg acc (before augmentation):', np.average(model_lg_acc_train))
    model_lg_acc_val = cross_val_score(estimator=model_lg, X=X_test_bow, y=y_test, cv=5, n_jobs=-1)
    print('Val avg acc (before augmentation):', np.average(model_lg_acc_val))

    pipeline = Pipeline([ ('vectorizer',tfidf_vec), ('clf', LogisticRegression(solver='saga'))])
    pipeline.fit(X_train, y_train)

    dummy_premise = [0 for i in range(len(X_train))]
    _, new_example, _, lr_label, explainer = get_augment_samples(pipeline, X_train, y_train, dummy_premise)

    X_train_aug = X_train + new_example
    y_train_aug = y_train + lr_label

    tfidf_vec = TfidfVectorizer(min_df = 10, token_pattern = r'[a-zA-Z]+')
    X_train_bow_aug = tfidf_vec.fit_transform(X_train_aug)
    X_test_bow_aug = tfidf_vec.transform(X_test)

    model_lg_aug = LogisticRegression(solver='saga')
    model_lg_aug.fit(X_train_bow_aug, y_train_aug)

    model_lg_acc_train_aug = cross_val_score(estimator=model_lg_aug, X=X_train_bow_aug, y=y_train_aug, cv=3, n_jobs=-1)
    print('Train avg acc (after augmentation):', np.average(model_lg_acc_train_aug))
    model_lg_acc_val_aug = cross_val_score(estimator=model_lg_aug, X=X_test_bow_aug, y=y_test, cv=3, n_jobs=-1)
    print('Val avg acc (after augmentation):', np.average(model_lg_acc_val_aug))

    if vis:
        # visualization of spurious predictions and LIME importance of words
        for i in range(5):
            index = random.randint(0, len(X_train))
            text_sample = train_hypothesis[index]
            lime_vis_example(explainer, text_sample, pipeline)

        # visualize top spurious words base on LR coefficients
        # reference: https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/nlp/ml-sklearn-classification.html
        ## Extract the coefficients of the model from the pipeline
        importances = pipeline.named_steps['clf'].coef_.flatten()
        ## Select top 10 positive/negative weights
        top_indices_pos = np.argsort(importances)[::-1][:10] 
        top_indices_neg = np.argsort(importances)[:10]
        feature_names = np.array(tfidf_vec.get_feature_names_out()) # List indexing is different from array
        feature_importance_df = pd.DataFrame({'FEATURE': feature_names[np.concatenate((top_indices_pos, top_indices_neg))],
                                            'IMPORTANCE': importances[np.concatenate((top_indices_pos, top_indices_neg))],
                                            'LABEL': ['Contradiction' for _ in range(len(top_indices_pos))]+['Entailment/Neutral' for _ in range(len(top_indices_neg))]})

        plt.figure(figsize=(7,5), dpi=100)
        plt.bar(x = feature_importance_df['FEATURE'], height=feature_importance_df['IMPORTANCE'])
        plt.ylabel("Feature Importance")
        plt.xticks(rotation=75)
        plt.title("Top Spurious Words", color="black")
    

# decorrelation_analysis(5000, True)