import numpy as np
import pandas as pd
import csv
import datetime
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from operator import itemgetter
from numpy import exp, array, random, dot
pandas2ri.activate()
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

readRDS = robjects.r['readRDS']
teams = readRDS('team_info.rds')
teams = pandas2ri.ri2py(teams)
teamArray = teams.values
games = readRDS('game_info.rds')
games = pandas2ri.ri2py(games)
games = games.values
conferences = readRDS('conference_info.rds')
conferences = pandas2ri.ri2py(conferences)
conferences = conferences.values
idArray = [0]*len(teamArray)
teamNames = [0]*len(teamArray)
ranks = '2016rank.csv'

teamsTest = readRDS('team_info (2015).rds')
teamsTest = pandas2ri.ri2py(teamsTest)
print teamsTest
teamArrayTest = teamsTest.values
gamesTest = readRDS('game_info (2015).rds')
gamesTest = pandas2ri.ri2py(gamesTest)
gamesTest = gamesTest.values
conferencesTest = readRDS('conference_info (2015).rds')
conferencesTest = pandas2ri.ri2py(conferencesTest)
conferencesTest = conferencesTest.values
idArrayTest = [0]*len(teamArrayTest)
teamNamesTest = [0]*len(teamArrayTest)
ranksTest = '2015rank.csv'

teams2014 = readRDS('team_info (2015).rds')
teams2014 = pandas2ri.ri2py(teams2014)
teamArray2014 = teams2014.values
games2014 = readRDS('game_info (2014).rds')
games2014 = pandas2ri.ri2py(games2014)
games2014 = games2014.drop('home_abbr', 1)
games2014 = games2014.drop('away_abbr', 1)
games2014 = games2014.values
conferences2014 = readRDS('conference_info (2015).rds')
conferences2014 = pandas2ri.ri2py(conferences2014)
conferences2014 = conferences2014.values
idArray2014 = [0]*len(teamArray2014)
teamNames2014 = [0]*len(teamArray2014)
ranks2014 = '2014rank.csv'

teams2013 = readRDS('team_info (2015).rds')
teams2013 = pandas2ri.ri2py(teams2013)
teamArray2013 = teams2013.values
games2013 = readRDS('game_info (2013).rds')
games2013 = pandas2ri.ri2py(games2013)
games2013 = games2013.drop('home_abbr', 1)
games2013 = games2013.drop('away_abbr', 1)
games2013 = games2013.values
conferences2013 = readRDS('conference_info (2015).rds')
conferences2013 = pandas2ri.ri2py(conferences2013)
conferences2013 = conferences2013.values
idArray2013 = [0]*len(teamArray2013)
teamNames2013 = [0]*len(teamArray2013)
ranks2013 = '2013rank.csv'

teams2012 = readRDS('team_info (2015).rds')
teams2012 = pandas2ri.ri2py(teams2012)
teamArray2012 = teams2012.values
games2012 = readRDS('game_info (2012).rds')
games2012 = pandas2ri.ri2py(games2012)
games2012 = games2012.drop('home_abbr', 1)
games2012 = games2012.drop('away_abbr', 1)
games2012 = games2012.values
conferences2012 = readRDS('conference_info (2015).rds')
conferences2012 = pandas2ri.ri2py(conferences2012)
conferences2012 = conferences2012.values
idArray2012 = [0]*len(teamArray2012)
teamNames2012 = [0]*len(teamArray2012)
ranks2012 = '2012rank.csv'

teams2011 = readRDS('team_info (2015).rds')
teams2011 = pandas2ri.ri2py(teams2011)
teamArray2011 = teams2011.values
games2011 = readRDS('game_info (2011).rds')
games2011 = pandas2ri.ri2py(games2011)
games2011 = games2011.drop('home_abbr', 1)
games2011 = games2011.drop('away_abbr', 1)
games2011 = games2011.values
conferences2011 = readRDS('conference_info (2015).rds')
conferences2011 = pandas2ri.ri2py(conferences2011)
conferences2011 = conferences2011.values
idArray2011 = [0]*len(teamArray2011)
teamNames2011 = [0]*len(teamArray2011)
ranks2011 = '2011rank.csv'




def sigmoid(x):
  return 1 / (1 + exp(-x))
  
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] # size of input layer
    n_h = 12
    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)
    
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(12) # we set up a seed so that your output matches ours although the initialization is random.
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = W1.dot(X) + b1
    A1 = np.tanh(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
    
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example
    # Compute the cross-entropy cost
    logprobs = Y*np.log(A2) + (1-Y)*np.log(1-A2)
    cost = -1.0/m * np.sum(logprobs)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2= A2 - Y
    dW2 = 1.0/m * dZ2.dot(A1.T)
    db2 = 1.0/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * (1 - np.power(A1, 2))
    dW1 = 1.0/m * dZ1.dot(X.T)
    db1 = 1.0/m * np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
 
def update_parameters(parameters, grads, learning_rate = 0.01):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters 
    

def nn_model(X, Y, X_assess, Y_assess, n_h, num_iterations = 10000, print_cost=True):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(32)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
         
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)
 
        parameters = update_parameters(parameters, grads)
        
        predictions = predict(parameters, X_assess)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            print(np.corrcoef(Y_assess, predictions)[0][1])
            r = 0
            w = 0
            for x in range(0, len(predictions[0])):
                if ((predictions[0][x] > .5) & (Y_assess[0][x] > .5)) | ((predictions[0][x] < .5) & (Y_assess[0][x] < .5)):
                    r = r+1
                else:
                    w = w+1
            print float(r)/float((r+w))
            

    return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    A2, cache = forward_propagation(X, parameters)

    
    return A2
    
    
def nameArray(teamNames, teamArray):
    for x in range(0, len(teamArray)):
        teamNames[x] = teamArray[x][3]
    return teamNames

def getRanks(teamArray, rank):
    ranksArray=[[]]
    with open(rank) as csvDataFile:
        ranks = csv.reader(csvDataFile)
        for row in ranks:
            tn = row[3]
            end = 0;
            for x in range(0, len(tn)):
                if (tn[x:x+1] == "("):
                    end = x-1
                    break;
            tn = tn[:end]
            if (tn == "Western Michigan"):
                tn = "W Michigan"
            if (tn == "Ohio State"):
                tn = "OSU"
            if (tn == "Florida State"):
                tn = "FSU"
            if (tn == "Virginia Tech"):
                tn = "VT"
            if (tn == "South Florida"):
                tn = "USF"
            row[3] = tn
            ranksArray.append(row)
    ranksArray = ranksArray[2:27]
    teamRanks = [0]*len(teamArray)
    for x in range(0, len(ranksArray)):
        d = [j for j in teamArray if j[3] == ranksArray[x][3]]
        for y in range(0, len(teamArray)):
            if ranksArray[x][3] == teamArray[y][3]: #Confuses Western Michigan and Michigan adding them together when they should be seperate
                teamRanks[y] = teamRanks[y] + ((25-(float)(ranksArray[x][2]))/25.0)
    return teamRanks






def getP5(conferences):
    power5 = [False]*len(conferences)
    #Assigns power5 conferences to true or false. It doesn't look like the conferences array lines up with the teams array, so we will need to sort by ID.
    for x in range(0, len(conferences)):
        if conferences[x][1] == "Atlantic Coast Conference" or conferences[x][1] == "Big Ten Conference" or conferences[x][1] == "Big 12 Conference" or conferences[x][1] == "Southeastern Conference" or conferences[x][1] == "Pac-12 Conference":
            power5[x] = True
        else:
            power5[x] = False
    return power5
    
def getColleyRankings(games, teamArray, week):
    n = len(teamArray);
    games = np.array(games[41:])
    games = games[games[:, 7].argsort()]
    indexVar = 0
    for x in range(0, len(games)):
        if (int(games[x][7]) == week+1):
                indexVar = x
                break;
    games = games[:indexVar]
    wins = [0] * n;
    losses = [0] * n;
    awayTeam = np.squeeze([row[1] for row in games])
    homeTeam = np.squeeze([row[2] for row in games])
    aScore = np.squeeze([row[3] for row in games])
    hScore = np.squeeze([row[4] for row in games])
    c = [ [ 0 for i in range(n) ] for j in range(n) ]
    
    for m in range(0, len(homeTeam)):
        hTeam = homeTeam[m];
        aTeam = awayTeam[m];
        c[hTeam][aTeam] = c[hTeam][aTeam] - 1;
        c[aTeam][hTeam] = c[aTeam][hTeam] - 1;
        if(hScore[m] > aScore[m]):
            wins[hTeam] = wins[hTeam] + 1;
            losses[aTeam] = losses[aTeam] + 1;
        else:
            losses[aTeam] = losses[aTeam] + 1;
            wins[aTeam] = wins[aTeam] + 1;
    b = [0] * n
    for j in range(0, n):
        b[j] = 1+(wins[j]-losses[j])/2;


    for l in range(0, n):
        c[l][l] = 2 + wins[l] + losses[l];


    c = np.linalg.inv(np.asarray(c));
    values = np.matmul(c, b);
    
    return values

    
    
    
 
def indices(teamArray, conferences, idArray):
    for x in range(0, len(teamArray)):
        idArray[x] = int(teamArray[x][0])
        teamArray[x][0] = x
    for x in range(0, len(conferences)):
        conferences[x][0] = idArray.index(int(conferences[x][0]))
    conferences = sorted(conferences, key=itemgetter(0))
    return teamArray, conferences
    
def gameDataCompile(teams, conferences, games, teamNames):
    gameRanksHome = []
    gameRanksAway = []
    gameP5Home = []
    gameP5Away = []
    colleyRanksHome = []
    colleyRanksAway = []
    results = [0] * len(games)
    for x in range(0, len(games)):
        awayTeam = games[x][1]
        homeTeam = games[x][2]
        if(awayTeam in teamNames):
            y = teamNames.index(awayTeam)
            games[x][1] = y
        else:
            games[x][1] = 206
            print awayTeam
        if (homeTeam in teamNames):
            z = teamNames.index(homeTeam)
            games[x][2] = z
        else :
            games[x][2] = 206
            print homeTeam
    for x in range(0, len(games)):
        awayTeam = games[x][1]
        homeTeam = games[x][2]
        gameRanksHome.append(teams[homeTeam][5])
        gameRanksAway.append(teams[awayTeam][5])
        if (games[x][4] > games[x][3]):
            results[x] = 1
        passedH = False;
        passedA = False;
        for sublist in conferences:
            if (sublist[0] == homeTeam) & (sublist[3] == True):
                gameP5Home.append(1)
                passedH = True
            if (sublist[0] == awayTeam) & (sublist[3] == True):
                gameP5Away.append(1)
                passedA = True
        if (passedH == False):
            gameP5Home.append(0)
        if (passedA == False):
            gameP5Away.append(0)

            
    finGames = np.atleast_2d(gameRanksAway).T
    finGames = np.hstack((finGames, np.atleast_2d(gameRanksHome).T))  
    finGames = np.hstack((finGames, np.atleast_2d(gameP5Away).T))   
    finGames = np.hstack((finGames, np.atleast_2d(gameP5Home).T))
    
    for x in range(1, 14):
        colleyRanksHome = []
        colleyRanksAway = []
        colley = getColleyRankings(games, teams, x)
        for x in range(0, len(games)):
            awayTeam = games[x][1]
            homeTeam = games[x][2]
            colleyRanksHome.append(colley[homeTeam])
            colleyRanksAway.append(colley[awayTeam])
        finGames = np.hstack((finGames, np.atleast_2d(colleyRanksHome).T))   
        finGames = np.hstack((finGames, np.atleast_2d(colleyRanksAway).T)) 
    
    

    return finGames, results
    
        

            
if __name__ == "__main__":
    ranks = getRanks(teamArray, ranks)
    p5 = getP5(conferences)
    teamNames = nameArray(teamNames, teamArray)
    conferences = np.hstack((conferences, np.atleast_2d(p5).T))
    teamArray, conferences = indices(teamArray, conferences, idArray)
    teamArray = np.hstack((teamArray, np.atleast_2d(ranks).T))
    teamArray = np.hstack((teamArray, np.full((1, len(teamArray)), True).T))
    gameData, results = gameDataCompile(teamArray, conferences, games, teamNames)
    gameData = np.squeeze(array(gameData))
    results = np.squeeze(array(results))
    gameData = np.atleast_2d(gameData)
    results = np.atleast_2d(results)
    results = np.transpose(results)
    
    ranksTest = getRanks(teamArrayTest, ranksTest)
    p5Test = getP5(conferencesTest)
    teamNamesTest = nameArray(teamNamesTest, teamArrayTest)
    conferencesTest = np.hstack((conferencesTest, np.atleast_2d(p5Test).T))
    teamArrayTest, conferencesTest = indices(teamArrayTest, conferencesTest, idArrayTest)
    teamArrayTest = np.hstack((teamArrayTest, np.atleast_2d(ranksTest).T))
    teamArrayTest = np.hstack((teamArrayTest, np.full((1, len(teamArrayTest)), True).T))
    gameDataTest, resultsTest = gameDataCompile(teamArrayTest, conferencesTest, gamesTest, teamNamesTest)
    gameDataTest = np.squeeze(array(gameDataTest))
    resultsTest = np.squeeze(array(resultsTest))
    gameDataTest = np.atleast_2d(gameDataTest)
    resultsTest = np.atleast_2d(resultsTest)
    resultsTest = np.transpose(resultsTest)
    
    ranks2014 = getRanks(teamArray2014, ranks2014)
    p52014 = getP5(conferences2014)
    teamNames2014 = nameArray(teamNames2014, teamArray2014)
    conferences2014 = np.hstack((conferences2014, np.atleast_2d(p52014).T))
    teamArray2014, conferences2014 = indices(teamArray2014, conferences2014, idArray2014)
    teamArray2014 = np.hstack((teamArray2014, np.atleast_2d(ranks2014).T))
    teamArray2014 = np.hstack((teamArray2014, np.full((1, len(teamArray2014)), True).T))
    gameData2014, results2014 = gameDataCompile(teamArray2014, conferences2014, games2014, teamNames2014)
    gameData2014 = np.squeeze(array(gameData2014))
    results2014 = np.squeeze(array(results2014))
    gameData2014 = np.atleast_2d(gameData2014)
    results2014 = np.atleast_2d(results2014)
    results2014 = np.transpose(results2014)
    
    ranks2013 = getRanks(teamArray2013, ranks2013)
    p52013 = getP5(conferences2013)
    teamNames2013 = nameArray(teamNames2013, teamArray2013)
    conferences2013 = np.hstack((conferences2013, np.atleast_2d(p52013).T))
    teamArray2013, conferences2013 = indices(teamArray2013, conferences2013, idArray2013)
    teamArray2013 = np.hstack((teamArray2013, np.atleast_2d(ranks2013).T))
    teamArray2013 = np.hstack((teamArray2013, np.full((1, len(teamArray2013)), True).T))
    gameData2013, results2013 = gameDataCompile(teamArray2013, conferences2013, games2013, teamNames2013)
    gameData2013 = np.squeeze(array(gameData2013))
    results2013 = np.squeeze(array(results2013))
    gameData2013 = np.atleast_2d(gameData2013)
    results2013 = np.atleast_2d(results2013)
    results2013 = np.transpose(results2013)
    
    ranks2012 = getRanks(teamArray2012, ranks2012)
    p52012 = getP5(conferences2012)
    teamNames2012 = nameArray(teamNames2012, teamArray2012)
    conferences2012 = np.hstack((conferences2012, np.atleast_2d(p52012).T))
    teamArray2012, conferences2012 = indices(teamArray2012, conferences2012, idArray2012)
    teamArray2012 = np.hstack((teamArray2012, np.atleast_2d(ranks2012).T))
    teamArray2012 = np.hstack((teamArray2012, np.full((1, len(teamArray2012)), True).T))
    gameData2012, results2012 = gameDataCompile(teamArray2012, conferences2012, games2012, teamNames2012)
    gameData2012 = np.squeeze(array(gameData2012))
    results2012 = np.squeeze(array(results2012))
    gameData2012 = np.atleast_2d(gameData2012)
    results2012 = np.atleast_2d(results2012)
    results2012 = np.transpose(results2012)
    
    ranks2011 = getRanks(teamArray2011, ranks2011)
    p52011 = getP5(conferences2011)
    teamNames2011 = nameArray(teamNames2011, teamArray2011)
    conferences2011 = np.hstack((conferences2011, np.atleast_2d(p52011).T))
    teamArray2011, conferences2011 = indices(teamArray2011, conferences2011, idArray2011)
    teamArray2011 = np.hstack((teamArray2011, np.atleast_2d(ranks2011).T))
    teamArray2011 = np.hstack((teamArray2011, np.full((1, len(teamArray2011)), True).T))
    gameData2011, results2011 = gameDataCompile(teamArray2011, conferences2011, games2011, teamNames2011)
    gameData2011 = np.squeeze(array(gameData2011))
    results2011 = np.squeeze(array(results2011))
    gameData2011 = np.atleast_2d(gameData2011)
    results2011 = np.atleast_2d(results2011)
    results2011 = np.transpose(results2011)
    
    variables = gameDataTest[:41]
    variables = np.concatenate((variables, gameData2014[:41]), axis=0)
    variables = np.concatenate((variables, gameData2013[:41]), axis=0)
    variables = np.concatenate((variables, gameData2012[:35]), axis=0)
    variables = np.concatenate((variables, gameData2011[:35]), axis=0)
    variables = np.transpose(variables)
    results1 = resultsTest[:41]
    results1 = np.concatenate((results1, results2014[:41]), axis=0)
    results1 = np.concatenate((results1, results2013[:41]), axis=0)
    results1 = np.concatenate((results1, results2012[:35]), axis=0)
    results1 = np.concatenate((results1, results2011[:35]), axis=0)
    print variables.shape
    results1 = np.transpose(results1)
    X_assess = gameData[:41]
    X_assess = np.transpose(X_assess)
    Y_assess = results[:41]
    Y_assess = np.transpose(Y_assess)
    parameters = nn_model(variables, results1, X_assess, Y_assess, 3, num_iterations=14300, print_cost=True)
    
    

    predictions = predict(parameters, X_assess)
    r = 0
    w = 0
    for x in range(0, len(predictions[0])):
        if ((predictions[0][x] > .5) & (Y_assess[0][x] > .5)) | ((predictions[0][x] < .5) & (Y_assess[0][x] < .5)):
            r = r+1
            print predictions[0][x]
        else:
            w = w+1
    print float(r)/float((r+w))
    np.save('predictions.npy', predictions)
    np.save('actualResults.npy', Y_assess)
    print(np.corrcoef(Y_assess, predictions))