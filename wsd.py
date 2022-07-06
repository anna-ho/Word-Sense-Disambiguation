# This program learns a model from line-train.txt and uses that model to perform word sense disambiguation on sentences from line-test.txt and
# outputs the features it used, the log-likelihood of each feature, and sense it predicts to my-model.txt and the predicted senses to my-line-answers.txt
# 
# To run type 'python3 wsd.py line-train.txt line-test.txt [name of document containing model] > [name of document containing answers]' into the terminal. NLTK will 
# need to be installed if it isn't already already. Directions for that can be found here: https://www.nltk.org/data.html
# 
# Decision list: features from the sentence containing the ambiguous word that had a log-likelihood greater than 0.
# 
# Accuracy of MFS baseline: 0.42857142857142855
# Accurcy w/ features: 0.8650793650793651
# 
# Confusion matrix: 
#          product  phone
# product       48      6
# phone         11     61

import math
import sys
import re
from nltk.corpus import stopwords

def main(): 

    # opens and reads the training data
    train = open(sys.argv[1])
    train_string = train.read().lower()

    # closes the training data
    train.close()

    # gets the list of stop words
    stop_words = set(stopwords.words('english'))
    
    # tokenizes the string based on the end of each instance of the ambiguous word and removes elements aren't instances
    train_array = train_string.split('</instance>')
    for element in train_array: 
        if 'instance' not in element: 
            train_array.remove(element)

    features = {}
    
    instances_of_phone = 0
    instances_of_product = 0

    # loops through each instance, gets the sense, and creates the feature vector and features dictionary
    for instance in train_array: 

        # captures the sense
        sense = re.search(r'.*senseid="(.*)".*', instance).group(1)
        
        # increments instances of phone sense
        if (sense == 'phone'): 
            instances_of_phone += 1 
        # increments instances of product sense
        if (sense == 'product'): 
            instances_of_product += 1 

        # captures the context
        context = re.search(r'.*\n<context>\n(.*)\n<\/context>\n.*', instance).group(1)

        # replaces plural 'lines' w/ singular 'line' for the head word
        context = re.sub(r'<head>lines<\/head>', '<head>line</head>', context)

        # splits the context into sentences based on where the </s> is
        sentences = [sentence for sentence in context.split('</s>')]
        
        # gets the specific sentence the head word is found in
        for element in sentences:
            if '<head>line</head>' in element:
                sentence = element

        # removes the tags that could come in front or at the end of a sentence
        sentence = re.sub(r' ?<s>|<p>|<\/p>|<@> ?', '', sentence)
        
        # removes special characters except for brackets bc that's what surrounds <head> and </head> so we know where our ambiguous word is
        sentence = re.sub(r'[^A-Za-z0-9\s<>]+', '', sentence) 

        # tokenizes the instance
        sentence_array = sentence.split() 

        # removes stop words      
        sentence_array = [word for word in sentence_array if word not in stop_words] 

        vector = {}

        # creates feature vector for the sentence
        for word in sentence_array: 
            # checks if the word is the head word (the head word is not included in the vector)
            if (word != '<head>line</head>'): 
                if (word in vector): 
                    vector[word] += 1
                else:
                    vector[word] = 1

        # adds the features to the 2d dictionary features 
        for feature in vector: 
            if (feature in features):  
                if (sense in features[feature]):  
                    features[feature][sense] += 1
                else: 
                    features[feature][sense] = 1
            else: 
                features[feature] = {}
                features[feature][sense] = 1
    
    # determines the most frequent sense
    mfs = ''
    if (instances_of_phone > instances_of_product): 
        mfs = 'phone'
    else:
        mfs = 'product'
    
    features_log_likelihood = {} # dictionary that stores the log likelihood associated with each feature
    features_sense = {} # dictionary that stores the sense associated with each feature

    # calculates log likelihood and finds the sense that appears most frequently with each feature
    for feature in features: 

        # checks if the sense is in the dictionary for that feature and sets the frequency to 0.001 if it isn't
        if ('phone' in features[feature]): 
            freq_feature_phone = features[feature]['phone']
        else: 
            freq_feature_phone = 0.001 

        if ('product' in features[feature]): 
            freq_feature_product = features[feature]['product']
        else: 
            freq_feature_product = 0.001 
        
        # calculates log_likelihood
        log_likelihood = abs(math.log(freq_feature_phone/freq_feature_product))

        features_log_likelihood[feature] = log_likelihood

        # the sense will be the sense with highest frequency 
        # if they appear an equal number of times, the sense will be the most frequent sense
        if (freq_feature_phone > freq_feature_product): 
            sense = 'phone'
        elif (freq_feature_product > freq_feature_phone):
            sense = 'product'
        else: 
            sense = mfs
        
        features_sense[feature] = sense

    
    # sorts features in descending order based on how discriminatory they are
    features_log_likelihood = dict(sorted(features_log_likelihood.items(), key = lambda x: x[1], reverse = True))

    # removes features that have a log-likelihood of 0
    features_log_likelihood = {key:value for key, value in features_log_likelihood.items() if value > 0}

    # formats the model to be printed
    model = ""
    for feature in features_log_likelihood: 
            
            model += "Feature: " + feature + "\n"
            model += "Log-likelihood: " + str(features_log_likelihood[feature]) + "\n"
            model += "Sense: " + features_sense[feature] + "\n\n"

    # prints the model
    file_name = sys.argv[3]
    f = open(file_name, "w")
    print(model, file = f) 
    f.close()

    # opens and reads the test data
    test = open(sys.argv[2])
    test_string = test.read().lower()

    # closes the test data
    test.close()
    
    # tokenizes the test string and removes elements that do not include instances of the ambiguous word
    test_array = test_string.split('</instance>')
    for element in test_array: 
        if 'instance' not in element: 
            test_array.remove(element)

    line_word_sense = {}

    for instance in test_array: 

        # captures the instance id
        id = re.search(r'.*instance id="(.*)".*', instance).group(1)
        
        # captures the context
        context = re.search(r'.*\n<context>\n(.*)\n<\/context>\n.*', instance).group(1)
        
        # replaces plural 'lines' w/ singular 'line' for the head word
        context = re.sub(r'<head>lines<\/head>', '<head>line</head>', context)

        # splits the context into sentences based on where the </s> is
        sentences = [sentence for sentence in context.split('</s>')]
        
        # gets the specific sentence the head word is found in
        for element in sentences:
            if '<head>line</head>' in element:
                sentence = element
        
        # default sense is the sense that appeared most frequently
        sense = mfs

        # loops through feature dictionary and looks for if the feature exists in this sentence
        for feature in features_log_likelihood:

            # if a feature is found, picks the sense associated with that feature
            if feature in sentence: 
                sense = features_sense[feature]
                break

        line_word_sense[id] = sense
    
    # formats and prints the answers
    for element in line_word_sense: 
        print("<answer instance=\"" + element + "\" senseid=\"" + line_word_sense[element] + "\"/>")


if __name__ == '__main__':
    main()
