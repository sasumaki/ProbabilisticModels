import pandas as pd
from collections import Counter
from sklearn.preprocessing import normalize
import numpy as np
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def preprocess(corpus):

    documents = []
    for doc in corpus[["document"]]:
        documents.append(doc.split(","))
    words = [word for document in documents for word in document]

    ham = (corpus[corpus["class"] == "ham"])
    sentences = []
    for doc in ham["document"]:
        sentences.append(doc.split(","))
    hamwords = [word for sentence in sentences for word in sentence]

    spam = (corpus[corpus["class"] == "spam"])
    sentences = []
    for doc in spam["document"]:
        sentences.append(doc.split(","))
    spamwords = [word for sentence in sentences for word in sentence]

    hamCounts = Counter(hamwords)
    spamCounts = Counter(spamwords)

    wordCounts = Counter(hamwords + spamwords)
    print(sum(hamCounts.values()))
    print(sum(spamCounts.values()))
    print(sum(wordCounts.values()))
    words = wordCounts.keys()

    df = pd.DataFrame()
    df["word"] = words
    hamValues = []
    for word in df["word"]:
        if(word in hamCounts):
            hamValues.append((hamCounts.get(word) + 1) /
                             (sum(hamCounts.values())+len(hamCounts.values())+1))
        else:
            hamValues.append(
                0 + 1 / (sum(hamCounts.values())+len(hamCounts.values())+1))
    spamValues = []
    for word in df["word"]:
        if(word in spamCounts):
            spamValues.append((spamCounts.get(word) + 1) /
                              (sum(spamCounts.values())+len(spamCounts.values())+2))
        else:
            spamValues.append(0 + 1 / (sum(spamCounts.values()
                                           )+len(spamCounts.values())+2))

    df["ham"] = hamValues
    df["spam"] = spamValues

    hamProbability = (sum(hamCounts.values()) + 1) / \
        (sum(wordCounts.values()) + 2)
    spamProbability = (sum(spamCounts.values()) + 1) / \
        (sum(wordCounts.values()) + 2)

    return(corpus, df, hamProbability, spamProbability)


def calculatePosteriors(corpus, table, hamP, spamP):
    probabilityOfHam = []
    for doc in corpus["document"]:
        probability = hamP
        words = doc.split(",")
        for word in words:
            probability = (probability *
                           float(table[table["word"] == word]["ham"]))
        probabilityOfHam.append(probability)
    probabilityOfSpam = []
    for doc in corpus["document"]:
        probability = spamP
        words = doc.split(",")
        for word in words:
            probability = (probability *
                           float(table[table["word"] == word]["spam"]))
        probabilityOfSpam.append(probability)

    return(probabilityOfHam, probabilityOfSpam)


def createProbabilityTable(hamProb, spamProb, documents):
    sums = []
    for ham, spam in zip(hamProb, spamProb):
        sums.append(ham+spam)
    normalizedHam = []
    normalizedSpam = []
    i = 0
    for ham in hamProb:
        normalizedHam.append(ham/sums[i])
        normalizedSpam.append(1-(ham/sums[i]))
        i += 1
    classifyTable = pd.DataFrame(
        data={"doc": documents["document"], "hamprob": normalizedHam, "spamprob": normalizedSpam})
    classes = []
    for index, row in classifyTable.iterrows():

        higherValue = max(row["hamprob"], row["spamprob"])
        if higherValue == row["hamprob"]:
            classes.append("ham")
        else:
            classes.append("spam")
    classifyTable["class"] = classes
    return(classifyTable)


def generate_documents(table, hamProb, L=6, N=1000):
    documents = []
    classes = []
    for i in range(0, N):
        if(random.random() <= hamProb):
            y = "ham"
        else:
            y = "spam"

        document = np.random.choice(
            table["word"], size=L, replace=True, p=table[str(y)])
        separator = ","
        stringified = separator.join(document)
        documents.append(str(stringified))
        classes.append(y)

    return(pd.DataFrame(data={"class": classes, "document": documents}))


def testF1(table, hamProbability, spamProbability, L):
    print("generating with L = ", L)
    documents3 = generate_documents(table, hamProbability, L)
    print("calculating posteriors")
    hamProb4, spamProb4 = calculatePosteriors(
        documents3, table, hamProbability, spamProbability)
    print("creating table")
    classifyTable3 = createProbabilityTable(
        hamProb4, spamProb4, documents3)

    y_true = documents3["class"].astype("category").cat.codes
    y_pred = classifyTable3["class"].astype("category").cat.codes
    print("calculating F1-score")
    score = f1_score(y_true, y_pred)
    return(score)


if __name__ == '__main__':
    """
    Construct the multinomial naive Bayes classifier(NBC) for
    this corpus by specifying all required(conditional) probability distributions.
    Use parameter sharing. Use parameter smoothing(Laplace)
    to address the problem of zero probabilities.
    """

    corpus = pd.DataFrame()

    corpus["class"] = ["spam", "ham", "spam", "spam", "spam", "ham", "ham"]
    corpus["document"] = ["free,online,!!!,results,free", "results,repository,online,deadline,!!!",
                          "!!!,online,paper,free,!!!,paper", "!!!,conference,registration,online,!!!,deadline",
                          "free,call,free,registration,online", "conference,call,paper,registration,conference",
                          "submission,deadline,conference,paper,call,deadline"]

    print(corpus)
    # corpus.to_csv("csv/corpus.csv")
    corpus, table, hamProbability, spamProbability = preprocess(corpus)
    print(table)
    # table.to_csv("csv/probability_table.csv")
    print(sum(table["ham"]))
    print(sum(table["spam"]))
    """

    Use the NBC to calculate the posterior probability that
    each document from the corpus is spam
    """
    hamprob, spamProb = calculatePosteriors(
        corpus, table, hamProbability, spamProbability)

    hamprob = np.array(hamprob)
    hamprob = normalize(
        hamprob[:, np.newaxis], axis=0).ravel()
    hamprobTable = pd.DataFrame(
        data={"doc": corpus["document"], "hamprob": hamprob})
    print(hamprobTable)
    # hamprobTable.to_csv("csv/document_ham_table.csv")
    """
    Use the NBC to classify the following documents
    """
    documents = pd.DataFrame(data={"document": ["free,submission,online,!!!",
                                                "conference,paper,submission,deadline"]})

    hamProb, spamProb = calculatePosteriors(
        documents, table, hamProbability, spamProbability)
    classifyTable = createProbabilityTable(
        hamProb, spamProb, documents)
    # classifyTable.to_csv("csv/classify_table1.csv")
    print(classifyTable)
    """
    Design an example document D of at most five words that
    is especially hard to classify with the classifier, i.e.:
    0.49 < P(C = spam|D) < 0.51
    """
    documents2 = pd.DataFrame(data={"document": ["online,submission,free,conference,conference"]
                                    })

    hamprob2, spamprob2 = calculatePosteriors(
        documents2, table, hamProbability, spamProbability)
    classifyTable2 = createProbabilityTable(hamprob2, spamprob2, documents2)
    # classifyTable2.to_csv("csv/classify_table2.csv")
    print(classifyTable2)

    """
    Generate 1000 new documents of length L = 6 from the
    generative model and classify them in spam and ham. Compute
    the F1-score of the classifier based on the true class labels (known
    from data-generation). Repeat the procedure with L = 1, . . . , 20 and
    plot the F1-score as a function of document length L. What do you
    observe?
    """
    scores = []
    # for i in range(1, 20):
    #   scores.append(testF1(table, hamProbability, spamProbability, L=i))
    print(scores)

    plt.plot(scores)
    plt.xlabel("L")
    # plt.show()
