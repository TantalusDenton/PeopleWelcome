import math
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Calculates entropy from a list of probabilities
def entropy(probList):
    h = 0
    for prob in probList:
        h -= (prob * math.log(prob, 2))
    return h

# Calculates info gain from a feature of a dataframe
def infoGain(df, feature, label):
    # List for probabilities of each label occurring to be inserted in
    labelProbs = []
    for count in df[label].value_counts():
        # Divide each value count by amount of rows and append to labelProbs
        labelProbs.append(count / len(df))
    # Calculate entropy of label column
    ig = entropy(labelProbs)

    for cat in [0 , 1]:
        # Create a partition for each category in this feature
        partition = df[df[feature] == cat]
        probs = []
        for count in partition[label].value_counts():
            # Divide each label count of this partition by partition length
            # and add to list in order to later calculate entropy
            probs.append(count / len(partition))
        # Subtract each (category probability * partition entropy) from label entropy
        ig = ig - ((len(partition) / len(df)) * entropy(probs))
    
    # Return the info gain for this feature
    return ig

# Returns a decision tree from the given dataframe
# and prunes with a value of 125 if prune is True
def buildDecisionTree(df, prune, label, labelAmount):
    # Save label name for frequent use
    #label = df.columns.values[len(df.columns.values) - 1]
    labels = df[label].unique()
    
    # If all instances in this dataframe have the
    # same class label, return that class label
    if len(labels) == 1:
        if type(labels[0]) != int:
            val = labels[0]
            pyval = val.item()
            return int(pyval)
        return labels[0]
    
    # If there are no more features left to examine, OR in the
    # case that prune has been set to True, if the given dataframe
    # has less than 125 rows, return the majority class label

    if len(df.columns.values) - labelAmount == 1 or (prune and len(df) <= 125):
        max = 0
        counts = df[label].value_counts()
        for i in range(1, len(labels) - 1):
            if counts[i] > counts[max]:
                max = i
        if type(labels[max]) != int:
            val = labels[max]
            pyval = val.item()
            return int(pyval)
        return labels[max]
    
    infoGains = {}
    # Create a dictionary in the form {feature: info gain}
    # that contains each feature in the given dataframe
    for col in df.columns.values[0 : len(df.columns.values) - labelAmount]:
        infoGains[col] = infoGain(df, col, label)
    
    # Set 'max' to the title of the column with the greatest info gain
    if len(df.columns.values) - labelAmount > 2:
        columns = list(infoGains.keys())
        max = columns[0]
        for col in columns:
            if infoGains.get(col) > infoGains.get(max):
                max = col
    else:
        max = df.columns.values[0]

    #print('max type:', type(max))
    # Create a tree with the max info gain feature at the root
    bestTree = {max: {}} 
    # Add each existing category of this feature as a node
    for cat in df[max].unique():
        #print('cat type:', type(cat))
        intCat = int(cat.item())
        dfPartition = df[df[max] == cat]
        dfPartition.drop(max, axis=1, inplace=True)
        # Create a node off each category node by recursively calling
        # this function with a partitioned dataset where category=true
        bestTree.get(max)[intCat] = buildDecisionTree(dfPartition, prune, label, labelAmount)
    
    # Return the tree that has been recursively built
    return bestTree

# Calls buildDecisionTree function with given dataframe
# and prune boolean after removing the primary key from
# this dataframe if one exists
def getDecisionTree(df, prune, label, labelAmount):
    partition = df
    # For each column in dataframe
    for col in (df.columns.values):
        # If every instance in column is unique
        if len(df[col].unique()) == len(df[col]):
            # Remove this column from dataframe
            del partition[col]
            
    # Return tree built from dataframe without primary key
    return buildDecisionTree(partition, prune, label, labelAmount)

# Return the predicted class label of a given row
def query(row, indexDict, tree, treeFound):
    # If tree is a class label, return it
    if not (type(tree) == dict):
        return tree
    
    # Store root of tree in 'root'
    root = list(tree.keys())[0]

    # Use indexDict - a dictionary in the form {feature: index}
    # to fetch the category of root feature from given row)
    choice = row[indexDict.get(root)]

    # correct for json formatting if necessary
    if treeFound:
        choice = str(choice)

    # Store root of choice tree in leaf
    leaf = tree.get(root).get(choice)
    
    # Return query using leaf instead of tree
    return query(row, indexDict, leaf, treeFound)