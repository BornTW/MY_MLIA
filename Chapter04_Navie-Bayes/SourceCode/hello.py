#-*-conding:utf-8-*-
from numpy import *
import feedparser
import re

# 创建一些实验样本
def  loadDataSet():
    postingList = [['my','dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take','him','to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 创建一个包含在所以文档中出现的不重复的列表
def createVocabList(dataSet):
    #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    #创建一个空集合
    vocabSet = set([])
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集，在求并集时重复的元素只会保留一份
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#该函数的输入参数为 词汇表及某个文档
def setOfWords2Vec(vocabList, inputSet):
    #首先创建一个和词汇表等长的向量，并将其元素都置为 0
    returnVec = [0]*len(vocabList)
    # 遍历文档中所有单词，如果出现了词汇表中的单词，则将输出文档向量中的对应值设为 1
    for word in inputSet:
        if word in vocabList:
            #list.index(obj)从列表中找出某个值第一个匹配项的索引位置
            returnVec[vocabList.index(word)] = 1
        esle: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# # print(myVocabList)
# print(setOfWords2Vec(myVocabList, listOPosts[0]))

# tainMatrix 文档矩阵已经被转化为数字 trainCategory  每篇文档类别标签所构成的向量
def trainNb0(trainMatrix, trainCategory):
    #矩阵的行数
    numTrainDocs = len(trainMatrix)
    #矩阵的列数
    numWords = len(trainMatrix[0])
    # sum() 方法对系列进行求和计算。
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 这里是二分类问题，所以创建两个和每篇文档等维度(一维数组)以 1 填充的数组
    p0Num = ones(numWords); p1Num = ones(numWords)
    # 用来记录侮辱性或正常文档词语总数
    p0Denom = 2.0; p1Denom = 2.0
    #要遍历训练集trainMatrix中的所有文档。一旦某个词语（侮辱性或正常词语）在某一文档中出现，
    #则该词对应的个数（p1Num 或者p0Num )就加1，
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 该词对应的个数p1Num 就加1， 两个一维数组相加
            p1Num += trainMatrix[i]
            # 而且在所有的文档中，该文档的总词数也相应加
            p1Denom += sum(trainMatrix[i])
        else:
            # 该词对应的个数p0Num 就加1， 两个一维数组相加
            p0Num += trainMatrix[i]
            # 而且在所有的文档中，该文档的总词数也相应加
            p0Denom += sum(trainMatrix[i])
    #对每个元素除以该类别中的总词数。利用 numpy 可以很好实现，用一个数组除以浮点数即可
    #取对数减小误差
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    #函数会返回两个向量和一个概率。
    return p0Vect, p1Vect, pAbusive

# # 测试函数
# listOPosts, listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# trainMat = []
# #通过for 循环可以把字符型文档转换为 数值型
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
#
# #trainNb0()函数的 trainMat 文档举证只能为 数值型， listClasses 文档类别标签向量也只能为数值型
# p0V, p1V, pAb = trainNb0(trainMat, listClasses)
# print(p0V, " \n\n", p1V, "\n\n ", pAb)

#测试算法：根据现实情况修改分类器
#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    #通过for 循环可以把字符型文档转换为 数值型
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #trainNb0()函数的 trainMat 文档举证只能为 数值型， listClasses 文档类别标签向量也只能为数值型
    p0V, p1V, pAb = trainNb0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))

#print(testingNB())

#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            #index() 函数用于从列表中找出某个值第一个匹配项的索引位置。
            returnVec[vocabList.index(word)] +=1
    return returnVec
#准备数据：切分文本
mySent = 'This book is the best book on Python or M.L I have laid eyes upon'
#print(mySent.split())
regEx = re.compile('\\W')
listOfTokens = regEx.split(mySent)
#print(listOfTokens)
#只返回长度大于0的字符串
#print([tok.lower() for tok in listOfTokens if len(tok) > 0])
emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
#print(listOfTokens)

# 测试算法：使用朴素贝叶斯进行交叉验证
def textPare(bigString):
    import re
    listOfTokens = re.split(r'\W',bigString)
    #接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    #spam 文件夹下有 25 个txt文件
    for i in range(1,26):
        wordList = textPare(open('email/spam/%d.txt'% i,'r',encoding='UTF-8',errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textPare(open('email/ham/%d.txt'% i,'r',encoding='UTF-8',errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        #classList 列表为类别向量
        classList.append(0)
    #利用 createVocabList()函数去重复单词
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        #uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内。
        #其中的10封电子邮件被随机选择为测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        #选择出的数字所对应的文档被添加到测试集，同时也将其从训练集中剔除
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    #接下来的for循环遍历训练集的所有文档，对每封邮件基于词汇表并使用setOfWords2Vec()函数来构建词向量
    for docIndex in trainingSet:
        #把训练集转换为词向量
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # trainClasses 列表为训练集的类别向量
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNb0(array(trainMat),array(trainClasses))
    errorCount = 0
    #然后遍历测试集，对其中每封电子邮件进行分类。如果邮件分类错误，则错误数加1，最后给出总的错误百分比
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam)!= classList[docIndex]:
            errorCount += 1
    print("the error rate is:",float(errorCount)/len(testSet))


#print(spamTest())
#实例：使用朴素贝叶斯分类器从个人广告中获取区域倾向
#RSS源分类器及高频词去除函数
def calMostFreq(vocabList,fullTest):
    #导入操作符
    import operator
    #创建新的字典
    freqDict={}
    #遍历词条列表中的每一个词
    for token in vocabList:
        #将单词/单词出现的次数作为键值对存入字典
        freqDict[token]=fullTest.count(token)
    #按照键值value(词条出现的次数)对字典进行排序，由大到小
    sortedFreq=sorted(freqDict.items(),keys=operator.itemgetter(1),reverse=true)
    #返回出现次数最多的前30个单词
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    #新建三个列表
    docList=[];classList=[];fullTest=[]
    #获取条目较少的RSS源的条目数
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    #遍历每一个条目
    for i in range(minLen):
        #解析和处理获取的相应数据
        wordList=textPare(feed1['entries'][i]['summary'])
        #添加词条列表到docList
        docList.append(wordList)
        #添加词条元素到fullTest
        fullTest.extend(wordList)
        #类标签列表添加类1
        classList.append(1)
        #同上
        wordList=textPare(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        #此时添加类标签0
        classList.append(0)
    #构建出现的所有词条列表
    vocabList=createVocabList(docList)
    #找到出现的单词中频率最高的30个单词
    top30Words=calMostFreq(vocabList,fullTest)
    #遍历每一个高频词，并将其在词条列表中移除
    #这里移除高频词后错误率下降，如果继续移除结构上的辅助词
    #错误率很可能会继续下降
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    #下面内容与函数spaTest完全相同
    trainingSet=range(2*minLen);testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNb0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
rocabList,pSF,pNY = localWords(ny,sf)
print()

