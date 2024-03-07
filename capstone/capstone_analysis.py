# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:58:20 2022

Capstone Project
Pascal Wallisch
Intro to Data Science
Summer 2022

@author: Nick Ferrara
"""

#%% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

#%% Load Data
'''
Data Description:
This dataset features ratings data of 400 movies from 1097 research participants.
First row: Headers (Movie titles/questions)
Row 2-1098: Invididual participants
Columns 1-400: These columns contain the ratings for the 400 movies (0 to 4, and missing data)
Columns 401-421: These columns contain self-assessments on sensation seeking behaviors (1-5)
Columns 422-464: These columns contain responses to personality questions (1-5)
Columns 465-474: These columns contain self-reported movie experience ratings (1-5)
Column 475: Gender identity (1 = female, 2 = male, 3 = self-described)
Column 476: Only child (1 = yes, 0 = no, -1 = no response)
Column 477: Movies are best enjoyed alone (1 = yes, 0 = no, -1 = no response)
'''
movieData = np.genfromtxt('movieReplicationSet.csv',delimiter=',')
# Clear first row of labels
movieData = movieData[1:,:]
movieDataFrame = pd.read_csv('movieReplicationSet.csv', encoding='latin-1', header=None)

#%% Display Questions
def getQuestions(filename, indexA, indexB, sectionName):
    questions = pd.read_csv(filename, encoding='latin-1', header=None)
    questions = questions[[i for i in range(indexA,indexB)]]
    # Display the questions:
    print(sectionName+' Questions:')
    for i, col in enumerate(questions.loc[0]):
        print(i+1, col)

#%% Q1) What is the relationship between sensation seeking and movie experience?
# Load Sensation Seeking and Movie Experience data
sensationSeeking = movieData[:,400:421]
movieExperience = movieData[:,464:474]

#%% Filter data by removing rows with NaNs
# Removing row-wise NaNs is effective in this case because 
# 1. We need to preserve the participants answers across all the questions
# 2. We only lost 68 participants, which is a very small number of participants, and so we maintain our sample size
# 3. Imputing data could lead to biased results, which I would generally want to avoid, especially when we can maintain our large sample size
temp = np.hstack((sensationSeeking,movieExperience))
sensationAndExperience = temp[~np.isnan(temp).any(axis=1)]

#%% Compute Correlation Matrix
# We want to identify any relationships between the questions within the raw data
corrMatrix = np.corrcoef(sensationAndExperience,rowvar=False)

plt.imshow(corrMatrix) 
plt.xlabel('Question')
plt.ylabel('Question')
plt.title('Correlation (Sensation Seeking 0-20 and Movie Experience 21-30)')
plt.colorbar()
plt.show()

# We can see correlations ranging 0.2 to 0.6 between sensation seeking questions (0 to 20) and movie experience questions (21 to 30)
# We should investigate this further with PCA to understand potential relationships

#%% Compute PCAs for both Sensation Seeking and Movie Experience
sensationSeeking = sensationAndExperience[:,:21]
movieExperience = sensationAndExperience[:,21:]

#%% Compute PCA for Sensation Seeking
# First Z-Score the data to avoid potential issues with unequal variance and having the first principal component pointing to the mean
zScoredSensationSeeking = stats.zscore(sensationSeeking)

# Initialize PCA
pcaSensationSeeking = PCA().fit(zScoredSensationSeeking)

# Get sorted eigenvalues
eigValsSensationSeeking = pcaSensationSeeking.explained_variance_

# Loadings Matrix
loadingsSensationSeeking = pcaSensationSeeking.components_

# Transform the sensation seeking data according to the PCA
rotatedSensationSeeking = pcaSensationSeeking.fit_transform(zScoredSensationSeeking)

#%% Plot the sensation seeking eigenvalues
numQuestions = len(sensationSeeking[0,:])
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigValsSensationSeeking, color='blue')
plt.plot([0,numQuestions],[1,1],color='red')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Sensation Seeking Scree Plot')
plt.show()
# According to the Elbow criterion, we will keep the first 5 principal components since the 6th factor is where we can see the plateu begin after the elbow.

#%% Investigate the loadings matrix of sensation seeking
getQuestions('movieReplicationSet.csv', 400, 421, 'Sensation Seeking')

#%% Principal Component 1
plt.bar(x,loadingsSensationSeeking[0,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Sensation Seeking - Principal Component 1 (Risk Avoidance)')
plt.show()

# No risk - No fun (11)
# I like to be surprised even if it startles or scares me (12)
# I had a sheltered upbringing (15)
# I value my life to be well ordered and predictable (17)
# Have you ever been sky-diving? (20)
# The 2 questions (15 and 17) that have the greatest positive effect make Risk Avoidance a good description for this factor.
# The 3 questions (11, 12, and 20 besides question 13 since it is basically the same as question 20) that have the greatest negative effect are associated with risk-taking, which further strengthens
# the factor being risk avoidance as people who take risks are not those who avoid risk.

#%% Principal Component 2
plt.bar(x,loadingsSensationSeeking[1,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Sensation Seeking - Principal Component 2 (Extrovertism)')
plt.show()

# Have you ever bungee-jumped? (3)
# I enjoy impulse shopping (4)
# I sometimes go out on weeknights even if I have work to do (5)
# I enjoy going to large music or dance festivals (9)
# Have you ever parachuted? (13)
# Have you ever been sky-diving? (20)
# Is talkative (21)
# These 4 questions (4, 5, 9, and 21) have the greatest positive effect, making a good description for this factor being Extrovertism
# The 3 questions (3, 13, and 20) that have the greatest negative effect could be associated with introvertism,
# since being a risk-taker may discourage casual interactions with people that build relationships with more people as opposed to fewer people.

#%% Principal Component 3
plt.bar(x,loadingsSensationSeeking[2,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Sensation Seeking - Principal Component 3 (Seeking Fear)')
plt.show()

# Have you ever bungee-jumped? (3)
# I enjoy watching horror movies (10)
# I like to be surprised even if it startles or scares me (11)
# Have you ever parachuted? (13)
# I enjoy haunted houses (14)
# Have you ever been sky-diving? (20)
# These 2 questions (10, 11, and 14) have the greatest positive effect, making a good description for this factor being Seeking Fear
# The 3 questions that have the greatest negative effect (3, 13, and 20) could be associated with risk-taking rather than fear,
# since being a risk-taker may not be driven by fear outside of their control, but rather controlling fear which may push them away from excitement due to pure uncertainty like random scares (horror movies).

#%% Principal Component 4
plt.bar(x,loadingsSensationSeeking[3,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Sensation Seeking - Principal Component 4 (Seeking Stability)')
plt.show()

# I enjoy doing things without too much planning (6)
# My life is very stressful (16)
# I value my life to be well ordered and predictable (17)
# These 2 questions (16 and 17) have the greatest positive effect, making a good description for this factor being Seeking Stability
# The 1 questions that has the greatest negative effect (6) could be associated with uncertainty rather than predictability,
# so it makes sense that those who are seeking stability don't want to introduce uncertainty in their lives.

#%% Principal Component 5
plt.bar(x,loadingsSensationSeeking[4,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Sensation Seeking - Principal Component 5 (Modesty)')
plt.show()

# I enjoy driving fast (1)
# I enjoy being in large loud crowds like the Times Square Ball Drop on New Years Eve (8)
# I had a sheltered upbringing (15)
# Have you gambled or bet for money? (19)
# These 2 questions (1 and 19) have the greatest positive effect, making a good description for this factor being Modesty
# The 2 questions (8 and 15) have the greatest negative effect which makes sense since people having a sheltered upbringing and enjoying social events where they aren't the center of attention
# may not be inclined to do things that will make them feel in the spotlight such as driving fast and gambling.

#%% Compute PCA for Movie Experience
# First Z-Score the data to avoid potential issues with unequal variance and having the first principal component pointing to the mean
zScoredMovieExperience = stats.zscore(movieExperience)

# Initialize PCA
pcaMovieExperience = PCA().fit(zScoredMovieExperience)

# Get sorted eigenvalues
eigValsMovieExperience = pcaMovieExperience.explained_variance_

# Loadings Matrix
loadingsMovieExperience = pcaMovieExperience.components_

# Transform the movie experience data according to the PCA
rotatedMovieExperience = pcaMovieExperience.fit_transform(zScoredMovieExperience)

#%% Plot the movie experience eigenvalues
numQuestions = len(movieExperience[0,:])
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigValsMovieExperience, color='blue')
plt.plot([0,numQuestions],[1,1],color='red')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Movie Experience Scree Plot')
plt.show()
# According to the Kaiser criterion shown by the red line, we will keep the first 2 principal components since they have an eigenvalue greater than 1.

#%% Investigate the loadings matrix of movie experience
getQuestions('movieReplicationSet.csv', 464, 474, 'Movie Experience')

#%% Principal Component 1
plt.bar(x,loadingsMovieExperience[0,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Movie Experience - Principal Component 1 (Empathic)')
plt.show()

# When watching a movie I feel like the things on the screen are happening to me (5)
# The emotions on the screen "rub off" on me - for instance if something sad is happening I get sad or if something frightening is happening I get scared (7)
# When watching a movie I get completely immersed in the alternative reality of the film (8)
# These 3 questions (5, 7, and 8) have the greatest effect, making a good description for this factor being empathy
#%% Principal Component 2
plt.bar(x,loadingsMovieExperience[1,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Movie Experience - Principal Component 2 (Immersive Memory)')
plt.show()

# I have trouble following the story of a movie (2)
# I have trouble remembering the story of a movie a couple of days after seeing it (3)
# As a movie unfolds I start to have problems keeping track of events that happened earlier (6)
# The emotions on the screen "rub off" on me - for instance if something sad is happening I get sad or if something frightening is happening I get scared (7)
# When watching a movie I get completely immersed in the alternative reality of the film (8)
# These 2 questions (7 and 8) have the greatest positive effect, making a good description for this factor being Immersive Memory
# The 3 questions (2, 3, and 6) that have the greatest negative effect are associated with forgetfulness rather than immersive memory when recalling a movie,
# so it makes sense that people who are more immersed in the movie have a good memory of it.

#%% Find Correlation Between Movie Experience PCA and Sensation Seeking PCA
importantSensationSeekingPC = rotatedSensationSeeking[:,:5]*-1
importantMovieExperiencePC = rotatedMovieExperience[:,:2]*-1
corrMatrixSensationAndExperiencePCA = np.corrcoef(np.hstack((importantMovieExperiencePC,importantSensationSeekingPC)),rowvar=False)
plt.imshow(corrMatrixSensationAndExperiencePCA) 
plt.xlabel('Question')
plt.ylabel('Question')
plt.title('Correlation (Movie Experience 0-1, Sensation Seeking 2-6)')
plt.colorbar()
plt.show()

#%% Plot Extrovertism versus Empathic
x = importantSensationSeekingPC[:,1]*-1
x = x.reshape((len(x),1))
y = importantMovieExperiencePC[:,0]*-1
ourModel = LinearRegression().fit(x, y) 
r_sq = ourModel.score(x, y)
slope = ourModel.coef_
intercept = ourModel.intercept_
yHat = slope * x + intercept 
plt.plot(x,y,'o',markersize=3)
plt.xlabel('Sensation Seeking - Principal Component 2 (Extrovertism)') 
plt.ylabel('Movie Experience - Principal Component 1 (Empathic)')  
plt.plot(x,yHat,color='orange',linewidth=3)
plt.title('Using scikit-learn: R^2 = {:.3f}'.format(r_sq))

#%% Plot Seeking Stability versus Empathic
x = importantSensationSeekingPC[:,3]*-1
x = x.reshape((len(x),1))
y = importantMovieExperiencePC[:,0]*-1
ourModel = LinearRegression().fit(x, y) 
r_sq = ourModel.score(x, y)
slope = ourModel.coef_
intercept = ourModel.intercept_
yHat = slope * x + intercept 
plt.plot(x,y,'o',markersize=3)
plt.xlabel('Sensation Seeking - Principal Component 4 (Seeking Stability)') 
plt.ylabel('Movie Experience - Principal Component 1 (Empathic)')  
plt.plot(x,yHat,color='orange',linewidth=3)
plt.title('Using scikit-learn: R^2 = {:.3f}'.format(r_sq)) 

#%% Plot Modesty versus Empathic
x = importantSensationSeekingPC[:,4]*-1
x = x.reshape((len(x),1))
y = importantMovieExperiencePC[:,0]*-1
ourModel = LinearRegression().fit(x, y) 
r_sq = ourModel.score(x, y)
slope = ourModel.coef_
intercept = ourModel.intercept_
yHat = slope * x + intercept 
plt.plot(x,y,'o',markersize=3)
plt.xlabel('Sensation Seeking - Principal Component 5 (Modesty)') 
plt.ylabel('Movie Experience - Principal Component 1 (Empathic)')  
plt.plot(x,yHat,color='orange',linewidth=3)
plt.title('Using scikit-learn: R^2 = {:.3f}'.format(r_sq))

# Conclusion: There is no relationship between the "immersive memory" (movie experience principal component 2) and any sensation seeking principal component with the largest correlation in magnitude being 0.059. 
# There may be a very weak relationship between the "empathic" (movie experience principal component 1) and each of "Extrovertism" (sensation seeking principal component 2), "Seeking Stability" (sensation seeking principal component 4), and "Modesty" (sensation seeking principal component 5).

# From the plots and the magnitude of the correlations, it is hard to find evidence supporting the claim that there is any substantial relationship between movie experience and sensation seeking.

#%% Q2) Is there evidence of personality types based on the data of these research participants? If so, characterize these types both quantitatively and narratively.
personality = movieData[:,421:464]
print(len(personality[:,0])-len(personality[~np.isnan(personality).any(axis=1)]))
personality = personality[~np.isnan(personality).any(axis=1)] # Only lost 97 participants

#%% Compute PCA for Personality
# First Z-Score the data to avoid potential issues with unequal variance and having the first principal component pointing to the mean
zScoredPersonality = stats.zscore(personality)

# Initialize PCA
pcaPersonality = PCA().fit(zScoredPersonality)

# Get sorted eigenvalues
eigValsPersonality = pcaPersonality.explained_variance_

# Loadings Matrix
loadingsPersonality = pcaPersonality.components_

# Transform the movie experience data according to the PCA
rotatedPersonality = pcaPersonality.fit_transform(zScoredPersonality)

#%% Plot the movie experience eigenvalues
numQuestions = len(personality[0,:])
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigValsPersonality, color='blue')
plt.plot([0,numQuestions],[1,1],color='red')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.title('Personality Scree Plot')
plt.show()
# According to the elbow criterion, we will keep the first 6 principal components since the factors plateu after the 6th component

#%% Investigate the loadings matrix of personality
getQuestions('movieReplicationSet.csv', 421, 464, 'Personality')

#%% Principal Component 1
plt.bar(x,loadingsPersonality[0,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Personality - Principal Component 1 (Extrovertism)')
plt.show()

# Is depressed/Blue (3) - negative
# Is reserved (5) - negative
# Is full of energy (10) - positive
# Generates a lot of Enthusiasm (15) - positive
# Tends to be quiet (20) - negative
# Can be cold and aloof (26) - negative
# Is outgoing/sociable (35) - positive
# The questions (10, 15, and 35) that have the greatest positive effect are associated with extrovertism, which makes sense since
# the questions (3, 5, 20, and 26) that have the greatest negative effect are associated with introvertism

#%% Principal Component 2
plt.bar(x,loadingsPersonality[1,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Personality - Principal Component 2 (Neuroticism)')
plt.show()

# Is relaxed/handles stress well (8) - negative
# Worries a lot (18) - positive
# Has an active imagination (19) - positive
# Is emotionally stable/not easily upset (23) - negative
# Values artistic/aesthetic experiences (29) - positive
# Likes to reflect/play with ideas (39) - positive
# The questions (18, 19, 29, and 39) that have the greatest positive effect are associated with neuroticism, which makes sense since
# the questions (8 and 23) that have the greatest negative effect are associated with level-headed thinking or less neurotic thinking.

#%% Principal Component 3
plt.bar(x,loadingsPersonality[2,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Personality - Principal Component 3 (conscientiousness)')
plt.show()

# Is reserved (5) - positive
# Can be somewhat careless (7) - negative
# Is a reliable worker (12) - positive
# Tends to be disorganized (17) - negative
# Tends to be quiet (20) - positive
# Is outgoing/sociable (35) - negative
# The questions (5, 12, and 20) that have the greatest positive effect are associated with conscientiousness, which makes sense since
# the questions (7, 17, and 35) that have the greatest negative effect are associated with irresponsibility given their lack of organization and care while potentially prioritizing socializing over responsibilities, which is relatively opposite to conscientiousness.

#%% Principal Component 4
plt.bar(x,loadingsPersonality[3,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Personality - Principal Component 4 (Disagreeableness)')
plt.show()

# Starts quarrels with others (11) - positive
# Has a forgiving nature (16) - negative
# Tends to be quiet (20) - negative
# Has an assertive personality (25) - positive
# Is considerate and kind to almost everyone (31) - negative
# Is sometimes rude to others (36) - positive
# The questions (11, 25, and 36) that have the greatest positive effect are associated with disagreeableness, which makes sense since
# the questions (16, 20, and 31) that have the greatest negative effect are associated with someone who is aggreeable with others.

#%% Principal Component 5
plt.bar(x,loadingsPersonality[4,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Personality - Principal Component 5 (Obedience)')
plt.show()

# Is relaxed/handles stress well (8) - negative
# Has a forgiving nature (16) - positive
# Is generally trusting (21) - positive
# Can be cold and aloof (26) - negative
# Is considerate and kind to almost everyone (31) - positive
# Remains calm in tense situations (33) - negative
# Gets nervous easily (38) - positive
# The questions (16, 21, 31, and 38) that have the greatest positive effect are associated with obedience, which makes sense since
# the questions (8, 26, and 33) that have the greatest negative effect are associated with someone who is less influenced by others thinking,
# thinking more for themselves, which is associated with less obedience.

#%% Principal Component 6
plt.bar(x,loadingsPersonality[5,:]*-1)
plt.xlabel('Question')
plt.ylabel('Loading Effect')
plt.title('Personality - Principal Component 6 (Apathetic)')
plt.show()

# Is depressed/Blue (3) - negative
# Is original/comes up with new ideas (4) - negative
# Can be somewhat careless (7) - positive
# Tends to be lazy (22) - positive
# Can be cold and aloof (26) - positive
# Has few artistic interests (40) - positive
# Values artistic/aesthetic experiences (29) - negative
# Is sophisticated in art or music or literature (43) - negative
# The questions (7, 22, 26, and 40) that have the greatest positive effect are associated with apathetic, which makes sense since
# the questions (3, 4, 29, and 43) that have the greatest negative effect are associated with someone who is curious and interested about life, which is opposite of being apathetic.

#%% Build a KMeans Cluster Model
def kMeansCluster(x1, x2, x1Name, x2Name, minClusters=2, maxClusters=9):
    x = np.column_stack((x1,x2))
    sSum = np.empty([maxClusters-minClusters+1,1])*np.NaN
    fig1 = plt.figure()
    fig1.suptitle('Silhouette Scores By Cluster Between {} and {}'.format(x1Name,x2Name))
    for ii in range(minClusters, maxClusters+1):
        kMeans = KMeans(n_clusters = int(ii)).fit(x) # compute kmeans using scikit
        cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
        cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
        s = silhouette_samples(x,cId) # compute the mean silhouette coefficient of all samples
        sSum[ii-minClusters] = sum(s) # take the sum
        # Plot data:
        plt.subplot(3,3,ii-minClusters+1) 
        plt.hist(s,bins=20) 
        plt.xlim(-0.2,1)
        plt.ylim(0,250)
        plt.xlabel('Silhouette score')
        plt.ylabel('Count')
        plt.title('Sum: {}'.format(int(sSum[ii-minClusters]))) # sum rounded to nearest integer
        plt.tight_layout() # adjusts subplot
        
    fig2 = plt.figure()
    fig2.suptitle('Silhouette Scores Plot Between {} and {}'.format(x1Name,x2Name))
    # Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
    plt.plot([i for i in range(minClusters,maxClusters+1)],sSum)
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of silhouette scores')
    plt.show()

    # Build Model
    numClusters = np.argmax(sSum[:,0])+minClusters
    kMeans = KMeans(n_clusters = numClusters).fit(x) 
    cId = kMeans.labels_ 
    cCoords = kMeans.cluster_centers_ 
    
    fig3 = plt.figure()
    fig3.suptitle('KMeans {} Clusters Between {} and {}'.format(numClusters,x1Name,x2Name))
    # Plot the color-coded data:
    for ii in range(numClusters):
        plotIndex = np.argwhere(cId == int(ii))
        plt.plot(x[plotIndex,0],x[plotIndex,1],'o',markersize=1)
        plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
        plt.xlabel(x1Name)
        plt.ylabel(x2Name)

#%% KMeans Cluster Between Each Factor
factorNames = ['Extrovertism','Neuroticism','Conscientiouness','Disagreeableness','Obedience','Apathetic']
combinations = []
for ii in range(len(factorNames)):
    for jj in range(len(factorNames)):
        if ii != jj and (ii,jj) not in combinations and (jj,ii) not in combinations:
            combinations.append((ii,jj))
            kMeansCluster(rotatedPersonality[:,ii]*-1, rotatedPersonality[:,jj]*-1, factorNames[ii], factorNames[jj], minClusters=2, maxClusters=9)

# Conclusion: We can see that the first principal component (Extrovertism) consistently produces two clusters, interpretted as two personality types,
# when clustering against all other important principal components (Neuroticism, Conscientiousness, Disagreeableness, Obediences, and Apathetic). I would not
# have expected, for instance, extrovertism and apathetic to have only two personality types since I would expect an apathetic person to more introverted and less extroverted, 
# which could be indicated by a distinct grouping. Otherwise, the pattern of two clusters split where extrovertism is 0 makes sense for the rest of the factors as I can understand
# both introverts and extroverts be expressed normally across the other factors, meaning introverts can be equally disagreeable as agreeable and be distributed similarly as for extroverts.
# One's reasons for being an introvert or an extrovert can be independent from their agreeableness with others.

# Show Extrovertism against the other factors

#%% Q3) Are movies that are more popular rated higher than movies that are less popular?
# Popularity can be based off the number of ratings for each movie
# We can create a two-sample hypothesis test with Mann-Whitney U

#%% Load Movie Ratings
movieRatings = movieData[:,:400]

#%% Find Rating Counts for each movie
numParticipants = len(movieRatings[:,0])
numMovies = len(movieRatings[0,:])
ratingsInfo = np.empty((numMovies,3))*np.NaN
for ii in range(numMovies):
    ratingsInfo[ii,0] = numParticipants-sum(np.isnan(movieRatings[:,ii]))
    ratingsInfo[ii,1] = np.median(movieRatings[~np.isnan(movieRatings[:,ii]),ii])
    ratingsInfo[ii,2] = np.mean(movieRatings[~np.isnan(movieRatings[:,ii]),ii])

#%% Create Popular and Unpopular Groups With Median Split
ratingsCountMedian = np.median(ratingsInfo[:,0])
popularMedianRatings = ratingsInfo[np.where(ratingsInfo[:,0] > ratingsCountMedian)[0],1]
popularMeanRatings = ratingsInfo[np.where(ratingsInfo[:,0] > ratingsCountMedian)[0],2]
unpopularMedianRatings = ratingsInfo[np.where(ratingsInfo[:,0] < ratingsCountMedian)[0],1]
unpopularMeanRatings = ratingsInfo[np.where(ratingsInfo[:,0] < ratingsCountMedian)[0],2]

#%% Plot Popular and Unpopular Ratings
plt.subplot(2,1,1)
plt.hist([popularMeanRatings,unpopularMeanRatings], bins=21, label=['Popular','Unpopular'])
plt.xlabel('Movie Rating (0-4)')
plt.ylabel('Number of Movies')
plt.legend(loc='upper right')
plt.title('Mean Ratings')
plt.subplot(2,1,2)
plt.hist([popularMedianRatings,unpopularMedianRatings], bins=21, label=['Popular','Unpopular'])
plt.xlabel('Movie Rating (0-4)')
plt.ylabel('Number of Movies')
plt.legend(loc='upper right')
plt.title('Median Ratings')
plt.tight_layout()

#%% Run Mann-Whitney U Test
medianPopularMedianRatings = np.median(popularMedianRatings)
medianPopularMeanRatings = np.median(popularMeanRatings)
medianUnpopularMedianRatings = np.median(unpopularMedianRatings)
medianUnpopularMeanRatings = np.median(unpopularMeanRatings)
uMedians,pMedians = stats.mannwhitneyu(popularMedianRatings,unpopularMedianRatings)
uMeans,pMeans = stats.mannwhitneyu(popularMeanRatings,unpopularMeanRatings)

# Conclusion: I conducted two Mann-Whitney U Tests, one between popular median ratings and unpopular median ratings (p-value=1.986e-34) and another between popular mean ratings and unpopular mean ratings (p-value=1.697e-40).
# Since these ratings are in ordinal scale, it seemed better to use the median ratings, but the mean ratings were better for visualizing the distributions. Nonetheless, both tests yielded p-values well below alpha=0.05, 
# thus we have statistically significant evidence to conclude the median rating of the popular median/mean ratings is not equal to the median rating of the unpopular median/mean ratings.
# The distributions of the popular mean/median ratings and unpopular mean/median ratings and our tests suggest that popular movies are rated higher than unpopular movies.
# It is worth noting that these tests were conducted without considering one-sided or two-sided tests since the p-values are already extremely small.

#%% Q4) Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?

#%% Load Shrek Ratings Data
shrekRatings = movieDataFrame.loc[:,movieDataFrame.loc[0,:] == 'Shrek (2001)'].to_numpy()[1:,:].astype(float)

#%% Group Ratings By Gender
genderData = movieData[:,474]
maleIndices = np.where(genderData == 1)[0]
femaleIndices = np.where(genderData == 2)[0]
shrekMaleRatings = shrekRatings[maleIndices]
shrekMaleRatings = shrekMaleRatings[~np.isnan(shrekMaleRatings)]
shrekFemaleRatings = shrekRatings[femaleIndices,:]
shrekFemaleRatings = shrekFemaleRatings[~np.isnan(shrekFemaleRatings)]

#%% Plot Male and Female Ratings
plt.hist([shrekMaleRatings,shrekFemaleRatings], bins=11, label=['Male','Female'])
plt.xlabel('Movie Rating (0-4)')
plt.ylabel('Frequency Ratings')
plt.legend(loc='upper right')
plt.title('Male and Female Ratings')

#%% Run Mann-Whitney U Test Between Gender Ratings
uGender,pGender = stats.mannwhitneyu(shrekMaleRatings,shrekFemaleRatings)
medianShrekMaleRatings = np.median(shrekMaleRatings)
medianShrekFemaleRatings = np.median(shrekFemaleRatings)

# Conclusion: I conducted a Mann-Whitney U test on the Shrek (2001) movie between male ratings and female ratings (p-value=0.0505).
# Although the median male ratings (3.5) was larger than the median female ratings (3.0), we found no statistical evidence that the median
# male ratings are different from the female ratings as the p-value is larger than alpha=0.05. This suggests the enjoyment of "Shrek (2001)" is not gendered. Since the p-value is close to alpha, it may be worth
# conducting different statistical tests with a larger sample size. 

#%% Q5) Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?

#%% Load Lion King Ratings Data
lionKingRatings = movieDataFrame.loc[:,movieDataFrame.loc[0,:] == 'The Lion King (1994)'].to_numpy()[1:,:].astype(float)

#%% Group Ratings By Siblings
siblingData = movieData[:,475]
onlyChildIndices = np.where(siblingData == 0)[0]
hasSiblingsIndices = np.where(siblingData == 1)[0]
onlyChildRatings = lionKingRatings[onlyChildIndices]
onlyChildRatings = onlyChildRatings[~np.isnan(onlyChildRatings)]
hasSiblingsRatings = lionKingRatings[hasSiblingsIndices]
hasSiblingsRatings = hasSiblingsRatings[~np.isnan(hasSiblingsRatings)]

#%% Plot only child and sibling Ratings
plt.hist([onlyChildRatings,hasSiblingsRatings], bins=21, label=['Only Child','Has Siblings'])
plt.xlabel('Movie Rating (0-4)')
plt.ylabel('Frequency Ratings')
plt.legend(loc='upper right')
plt.title('Only Child and Sibling Ratings')

#%% Run Mann-Whitney U Test Between Siblings Ratings
uSiblings,pSiblings = stats.mannwhitneyu(onlyChildRatings,hasSiblingsRatings,alternative='greater')
medianOnlyChildRatings = np.median(onlyChildRatings)
medianHasSiblingsRatings = np.median(hasSiblingsRatings)

# Conclusion: I conducted a one-sided Mann-Whitney U test on "The Lion King (1994)" movie to test if ratings of participants with no siblings are greater than ratings of participants with siblings (p-value=0.0216).
# Since the median only child ratings (4.0) was larger than the median has siblings ratings (3.5) and the p-value is less than alpha=0.05, we found statistical evidence that the median
# only child ratings is higher than the median has siblings ratings. This suggests that people who are only children enjoy "The Lion King (1994)" more than people with siblings.

#%% Q6) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who prefer to watch them alone?

#%% Load Wolf of Wall Street Ratings Data
wolfRatings = movieDataFrame.loc[:,movieDataFrame.loc[0,:] == 'The Wolf of Wall Street (2013)'].to_numpy()[1:,:].astype(float)

#%% Group Ratings By Yes and No to Movies Alone
aloneData = movieData[:,476]
yesAloneIndices = np.where(aloneData == 1)[0]
noAloneIndices = np.where(aloneData == 0)[0]
yesAloneRatings = wolfRatings[yesAloneIndices]
yesAloneRatings = yesAloneRatings[~np.isnan(yesAloneRatings)]
noAloneRatings = wolfRatings[noAloneIndices]
noAloneRatings = noAloneRatings[~np.isnan(noAloneRatings)]

#%% Plot Ratings
plt.hist([yesAloneRatings,noAloneRatings], bins=21, label=['Yes','No'])
plt.xlabel('Movie Rating (0-4)')
plt.ylabel('Frequency Ratings')
plt.legend(loc='upper right')
plt.title('Ratings Between Yes and No to Movies Alone')

#%% Run Mann-Whitney U Test Between Yes and No to Movies Alone Ratings
uAlone,pAlone = stats.mannwhitneyu(yesAloneRatings,noAloneRatings,alternative='greater')
uAlone2,pAlone2 = stats.mannwhitneyu(yesAloneRatings,noAloneRatings,alternative='less')
medianYesAloneRatings = np.median(yesAloneRatings) # 3.5
medianNoAloneRatings = np.median(noAloneRatings) # 3.0

# Conclusion: I conducted a one-sided Mann-Whitney U test on "The Wolf of Wall Street (2013)" movie to test if ratings of participants that prefer watching movies alone are greater than ratings of participants that prefer watching movies socially (p-value=0.0564).
# Since the median ratings of participants that prefer watching movies alone (3.5) was larger than the median ratings of participants that prefer watching movies socially (3.0), we already know without a test (note that the p-value of this one-sided test would be 0.9437) 
# that there will not be significant evidence that will suggest people who like to watch movies socially will enjoy "The Wolf of Wall Street (2013)" more than those who prefer to watch them alone. Furthermore, since our test yielded a p-value greater than alpha=0.05, 
# we cannot say the median ratings of both groups are different, and so, do not have statistical evidence to suggest people who like to watch movies alone will enjoy "The Wolf of Wall Street (2013)" more than those who prefer to watch them socially, which should have been the question in the first place.
# and the p-value is less than alpha=0.05, we found statistical evidence that the median only child ratings is higher than the median has siblings ratings. This suggests that people who are only children enjoy "The Lion King (1994)" more than people with siblings.

#%% Q7) There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) 
# in this dataset. How many of these are of inconsistent quality, as experienced by viewers?

#%% Load Franchise Groups
starWarsMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('Star Wars')].to_numpy()[1:,:].astype(float)
harryPotterMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('Harry Potter')].to_numpy()[1:,:].astype(float)
matrixMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('The Matrix')].to_numpy()[1:,:].astype(float)
indianaJonesMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('Indiana Jones')].to_numpy()[1:,:].astype(float)
jurassicParkMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('Jurassic Park')].to_numpy()[1:,:].astype(float)
piratesMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('Pirates of the Caribbean')].to_numpy()[1:,:].astype(float)
toyStoryMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('Toy Story')].to_numpy()[1:,:].astype(float)
batmanMovies = movieDataFrame.loc[:,movieDataFrame.loc[0,:].str.contains('Batman')].to_numpy()[1:,:].astype(float)

#%% Run Kruskal Test for each group
hStarWars,pKStarWars = stats.kruskal(starWarsMovies[~np.isnan(starWarsMovies[:,0]),0],starWarsMovies[~np.isnan(starWarsMovies[:,1]),1],starWarsMovies[~np.isnan(starWarsMovies[:,2]),2],starWarsMovies[~np.isnan(starWarsMovies[:,3]),3],starWarsMovies[~np.isnan(starWarsMovies[:,4]),4],starWarsMovies[~np.isnan(starWarsMovies[:,5]),5])
hHarryPotter,pKHarryPotter = stats.kruskal(harryPotterMovies[~np.isnan(harryPotterMovies[:,0]),0],harryPotterMovies[~np.isnan(harryPotterMovies[:,1]),1],harryPotterMovies[~np.isnan(harryPotterMovies[:,2]),2],harryPotterMovies[~np.isnan(harryPotterMovies[:,3]),3])
hMatrix,pKMatrix = stats.kruskal(matrixMovies[~np.isnan(matrixMovies[:,0]),0],matrixMovies[~np.isnan(matrixMovies[:,1]),1],matrixMovies[~np.isnan(matrixMovies[:,2]),2])
hIndianaJones,pKIndianaJones = stats.kruskal(indianaJonesMovies[~np.isnan(indianaJonesMovies[:,0]),0],indianaJonesMovies[~np.isnan(indianaJonesMovies[:,1]),1],indianaJonesMovies[~np.isnan(indianaJonesMovies[:,2]),2],indianaJonesMovies[~np.isnan(indianaJonesMovies[:,3]),3])
hJurassicPark,pKJurassicPark = stats.kruskal(jurassicParkMovies[~np.isnan(jurassicParkMovies[:,0]),0],jurassicParkMovies[~np.isnan(jurassicParkMovies[:,1]),1],jurassicParkMovies[~np.isnan(jurassicParkMovies[:,2]),2])
hPirates,pKPirates = stats.kruskal(piratesMovies[~np.isnan(piratesMovies[:,0]),0],piratesMovies[~np.isnan(piratesMovies[:,1]),1],piratesMovies[~np.isnan(piratesMovies[:,2]),2])
hToyStory,pKToyStory = stats.kruskal(toyStoryMovies[~np.isnan(toyStoryMovies[:,0]),0],toyStoryMovies[~np.isnan(toyStoryMovies[:,1]),1],toyStoryMovies[~np.isnan(toyStoryMovies[:,2]),2])
hBatman,pKBatman = stats.kruskal(batmanMovies[~np.isnan(batmanMovies[:,0]),0],batmanMovies[~np.isnan(batmanMovies[:,1]),1],batmanMovies[~np.isnan(batmanMovies[:,2]),2])

harryPotterMedians = [np.median(harryPotterMovies[~np.isnan(harryPotterMovies[:,ii]),ii]) for ii in range(len(harryPotterMovies[0,:]))]

# Conclusion: After conducting 8 Kruskal-Wallis tests, one for each franchise, we found Harry Potter was the only franchise with no statistical evidence suggesting inconsistent quality with each of the 6 Harry Potter movies having a median rating of 3.5. 
# We found statistical evidence that the 6 Star Wars movies are of inconsistent quality with at least one of the Star Wars movies
# having a different median rating since the p-value=8.016e-48 is much less than alpha=0.05. We found no statistical evidence that the 4 Harry Potter movies are of inconsistent quality with not one of the Harry Potter movies
# having a different median rating since the p-value=0.3433 is greater than alpha=0.05. We found statistical evidence that the 3 Matrix movies are of inconsistent quality with at least one of the Matrix movies
# having a different median rating since the p-value=3.124e-11 is much less than alpha=0.05. We found statistical evidence that the 4 Indiana Jones movies are of inconsistent quality with at least one of the Indiana Jones movies
# having a different median rating since the p-value=6.273e-10 is much less than alpha=0.05. We found statistical evidence that the 3 Jurassic Park movies are of inconsistent quality with at least one of the Jurassic Park movies
# having a different median rating since the p-value=7.637e-11 is much less than alpha=0.05. We found statistical evidence that the 3 Pirates of the Caribbean movies are of inconsistent quality with at least one of the Pirates of the Caribbean movies
# having a different median rating since the p-value=3.290e-05 is much less than alpha=0.05. We found statistical evidence that the 3 Toy Story movies are of inconsistent quality with at least one of the Toy Story movies
# having a different median rating since the p-value=5.066e-06 is much less than alpha=0.05. We found statistical evidence that the 3 Batman movies are of inconsistent quality with at least one of the Batman movies
# having a different median rating since the p-value=4.225e-42 is much less than alpha=0.05.

#%% Q8) Build a prediction model of your choice (regression or supervised learning) to predict movie
# ratings (for all 400 movies) from personality factors only. Make sure to use cross-validation
# methods to avoid overfitting and characterize the accuracy of your model.

#%% Load Movie Ratings and Personality Data
personality = movieData[:,421:464]
personalityIndices = ~np.isnan(personality).any(axis=1)
personality = personality[personalityIndices]
movieRatings = movieData[:,:400]

#%% Impute Missing Movie Ratings with Median Column Rating
for ii in range(len(movieRatings[0,:])):
    np.nan_to_num(movieRatings[:,ii],nan=np.nanmedian(movieRatings[:,ii]),copy=False)
movieRatings = movieRatings[personalityIndices,:]

#%% Build Training and Testing Datasets
X = personality # (1000,43)
Y = movieRatings # (1000,400)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=12)

#%% Build Random Forest Regressor Model For Decimal Ratings
forestModel = RandomForestRegressor()
numTrees = [2**i for i in range(1,9)] # 2 to 256
maxDepth = [2**i for i in range(1,9)] # 2 to 256
minSamplesLeaf = [i*2 for i in range(1,7)] # 2 to 12
minSamplesSplit = [i*2 for i in range(1,7)] # 2 to 12
forestRandomized = RandomizedSearchCV(estimator=forestModel,param_distributions={'n_estimators': numTrees,'max_depth': maxDepth,'min_samples_leaf': minSamplesLeaf,'min_samples_split': minSamplesSplit,},n_iter=200,cv=5,verbose=3,random_state=12,n_jobs=-1)
forestRandomized.fit(X, Y)
print(forestRandomized.best_params_)

#%% Build Model with best parameters
clfPersonality = RandomForestRegressor(n_estimators=128, min_samples_split=2, min_samples_leaf=12, max_depth=4)
clfPersonalityBaseline = RandomForestRegressor()
clfPersonality.fit(Xtrain,Ytrain)
clfPersonalityBaseline.fit(Xtrain,Ytrain)

#%% Compute Model Accuracy
print('Baseline Classifier (RMSE, Train Score, Test Score): ({},{},{})'.format((np.mean((Ytest-clfPersonalityBaseline.predict(Xtest))**2))**0.5, clfPersonalityBaseline.score(Xtrain,Ytrain), clfPersonalityBaseline.score(Xtest,Ytest)))
print('Tuned Classifier (RMSE, Train Score, Test Score): ({},{},{})'.format((np.mean((Ytest-clfPersonality.predict(Xtest))**2))**0.5, clfPersonality.score(Xtrain,Ytrain), clfPersonality.score(Xtest,Ytest)))

# Conclusion: I first imputed the movie ratings dataset of 400 movies by filling the nan values with the movie median rating for each movie.
# Then I removed participants who didn't answer all of the personality questions because we only lost 97 participants, which I would consider better since imputation with the column median
# on personality questions doesn't seem like a good representation of each participant's potential answers, although other methods of imputation may be better. After splitting the movie ratings
# data (to be predicted) and personality data (predictors) into a 20% test dataset and 80% training dataset, I then decided to build a 
# random forest regressor supervised learning model to predict a participants movie ratings based on their answers to the personality questions. In order to obtain a better model, I tuned selected hyper parameters
# (Number of trees from 2 to 256, Maximum depth from 2 to 256, Minimum sample leaf from 2 to 12, and Minimum sample split from 2 to 12) using the "RandomizedSearchCV" class from the sklearn library, which randomly sampled (without replacement)
# the hyper parameters 200 times without replacement using a 5-fold cross validation (computing the average score on 5 different samples of n-1 training data with 1 test data). Note that I used a regressor model to deal with the non-integer movie ratings rather than a classifier.
# I finally compared the tuned model to a baseline model and found that it had slightly better results as seen in the figure. However, both the tuned model and baselines models were still slightly worse than just guessing, which is unsurprising given how many movie ratings are being 
# predicted by a few predictors. What is interesting is that the tuned model, which is underfitting the training set actually performed better than the baseline model which fit the training set well. 
# I also am not surprised by these results because I don't expect one's personality to be a good predictor of their ratings of multiple movies.

#%% Q9) Build a prediction model of your choice (regression or supervised learning) to predict movie
# ratings (for all 400 movies) from gender identity, sibship status and social viewing preferences
# (columns 475-477) only. Make sure to use cross-validation methods to avoid overfitting and
# characterize the accuracy of your model.

#%% Load 3 categorical variables and movie ratings data
answeredIndices = ~np.isnan(movieData[:,474:477]).any(axis=1)
print(len(movieData[:,0])-sum(answeredIndices)) # 24 removed participants
genderData = movieData[answeredIndices,474]
siblingData = movieData[answeredIndices,475]
viewingData = movieData[answeredIndices,476]
movieRatings = movieData[:,:400]

#%% Impute Missing Movie Ratings with Median Column Rating
for ii in range(len(movieRatings[0,:])):
    np.nan_to_num(movieRatings[:,ii],nan=np.nanmedian(movieRatings[:,ii]),copy=False)
Y = movieRatings[answeredIndices,:]

#%% One Hot Encoding
genderData = np.where(genderData == 3, -1, genderData)
X = pd.DataFrame({
    'Gender': genderData,
    'Siblings': siblingData,
    'Viewing': viewingData})
categories = ['Gender','Siblings','Viewing']
encoder = OneHotEncoder(sparse=False,drop='first')
encoderArray = encoder.fit_transform(X[categories])
featureNames = encoder.get_feature_names_out(categories)
encoderDataFrame = pd.DataFrame(encoderArray, columns=featureNames)
encoderNumpyArray = encoderDataFrame.to_numpy() # concat 
X = encoderNumpyArray
#%% Build Training and Testing Datasets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=12)

#%% Build Random Forest Regressor Model For Decimal Ratings
forestModel = RandomForestRegressor()
numTrees = [2**i for i in range(1,9)] # 2 to 256
maxDepth = [2**i for i in range(1,9)] # 2 to 256
minSamplesLeaf = [i*2 for i in range(1,7)] # 2 to 12
minSamplesSplit = [i*2 for i in range(1,7)] # 2 to 12
forestRandomized = RandomizedSearchCV(estimator=forestModel,param_distributions={'n_estimators': numTrees,'max_depth': maxDepth,'min_samples_leaf': minSamplesLeaf,'min_samples_split': minSamplesSplit,},n_iter=200,cv=5,verbose=3,random_state=12,n_jobs=-1)
forestRandomized.fit(X, Y)
print(forestRandomized.best_params_)

#%% Build Model with best parameters
clfCategories = RandomForestRegressor(n_estimators=128, min_samples_split=4, min_samples_leaf=2, max_depth=2)
clfCategoriesBaseline = RandomForestRegressor()
clfCategories.fit(Xtrain,Ytrain)
clfCategoriesBaseline.fit(Xtrain,Ytrain)

#%% Compute Model Accuracy
print('Baseline Classifier (RMSE, Train Score, Test Score): ({},{},{})'.format((np.mean((Ytest-clfCategoriesBaseline.predict(Xtest))**2))**0.5, clfCategoriesBaseline.score(Xtrain,Ytrain), clfCategoriesBaseline.score(Xtest,Ytest)))
print('Tuned Classifier (RMSE, Train Score, Test Score): ({},{},{})'.format((np.mean((Ytest-clfCategories.predict(Xtest))**2))**0.5, clfCategories.score(Xtrain,Ytrain), clfCategories.score(Xtest,Ytest)))

# Conclusion: I first imputed the movie ratings dataset of 400 movies by filling the nan values with the movie median rating for each movie.
# Then I removed participants who didn't answer all of the categorical questions (gender, only child, and viewing preference) because we only lost 24 participants, which I would consider the best option
# since imputation with the column median on the categorical questions doesn't seem like a good representation of each participant's potential answers. I applied "One Hot Encoding" to the categorical variables
# so that the numerical values associated with each category wouldn't cause the model to intrepret them as numerically larger than one another since that wouldn't make sense. I dropped the no response (-1) from the only child and viewing preference categories
# and self-described (3) from the gender category as dummy variables to avoid multicollinearity. After splitting the movie ratings data (to be predicted) and categorical data via one hot encoding (predictors) into a 20% test dataset and 80% training dataset, 
# I then decided to build a random forest regressor supervised learning model to predict a participants movie ratings based on their answers to the categorical questions. In order to obtain a better model, I tuned selected hyper parameters
# (Number of trees from 2 to 256, Maximum depth from 2 to 256, Minimum sample leaf from 2 to 12, and Minimum sample split from 2 to 12) using the "RandomizedSearchCV" class from the sklearn library, which randomly sampled (without replacement)
# the hyper parameters 200 times without replacement using a 5-fold cross validation (computing the average score on 5 different samples of n-1 training data with 1 test data). Note that I used a regressor model to deal with the non-integer movie ratings rather than a classifier.
# I finally compared the tuned model to a baseline model and found that it had slightly better results as seen in the figure. However, both the tuned model and baselines models were still slightly worse than just guessing, which is unsurprising given how many movie ratings are being 
# predicted by only 3 predictors which are categorical. What is interesting is that the tuned model, which is underfitting the training set actually performed better than the baseline model which fit the training set well. 
# I also am not surprised by these results because I don't expect one's gender, only child, and viewing preference alone to be good predictors of their ratings of multiple movies.

#%% Q10) Build a prediction model of your choice (regression or supervised learning) to predict movie
# ratings (for all 400 movies) from all available factors that are not movie ratings (columns 401-
# 477). Make sure to use cross-validation methods to avoid overfitting and characterize the
# accuracy of your model.

#%% Load 3 categorical variables and numerical variables
answeredIndices = ~np.isnan(movieData[:,400:477]).any(axis=1)
print(len(movieData[:,0])-sum(answeredIndices)) # 151 removed participants
genderData = movieData[answeredIndices,474]
siblingData = movieData[answeredIndices,475]
viewingData = movieData[answeredIndices,476]
numericalData = movieData[answeredIndices,400:474]

#%% Impute Missing Movie Ratings with Median Column Rating
movieRatings = movieData[:,:400]
for ii in range(len(movieRatings[0,:])):
    np.nan_to_num(movieRatings[:,ii],nan=np.nanmedian(movieRatings[:,ii]),copy=False)
Y = movieRatings[answeredIndices,:]

#%% One Hot Encoding
genderData = np.where(genderData == 3, -1, genderData)
X = pd.DataFrame({
    'Gender': genderData,
    'Siblings': siblingData,
    'Viewing': viewingData})
categories = ['Gender','Siblings','Viewing']
encoder = OneHotEncoder(sparse=False,drop='first')
encoderArray = encoder.fit_transform(X[categories])
featureNames = encoder.get_feature_names_out(categories)
encoderDataFrame = pd.DataFrame(encoderArray, columns=featureNames)
encoderNumpyArray = encoderDataFrame.to_numpy()
X = np.hstack((numericalData,encoderNumpyArray))

#%% Build Training and Testing Datasets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

#%% Build Random Forest Regressor Model For Decimal Ratings
forestModel = RandomForestRegressor(verbose=10)
numTrees = [2**i for i in range(1,11)] # 2 to 1024
maxDepth = [2**i for i in range(1,9)] # 2 to 256
minSamplesLeaf = [i*2 for i in range(1,7)] # 2 to 12
minSamplesSplit = [i*2 for i in range(1,7)] # 2 to 12
forestRandomized = RandomizedSearchCV(estimator=forestModel,param_distributions={'n_estimators': numTrees,'max_depth': maxDepth,'min_samples_leaf': minSamplesLeaf,'min_samples_split': minSamplesSplit,},n_iter=200,cv=5,verbose=10,n_jobs=-1)
forestRandomized.fit(X, Y)
print(forestRandomized.best_params_)

#%% Build Model with best parameters
clfAll = RandomForestRegressor(n_estimators=1024, min_samples_split=12, min_samples_leaf=2, max_depth=256, n_jobs=-1)
clfAllBaseline = RandomForestRegressor(n_jobs=-1)
clfAll.fit(Xtrain,Ytrain)
clfAllBaseline.fit(Xtrain,Ytrain)

#%% Compute Model Accuracy
print('Baseline Classifier (RMSE, Train Score, Test Score): ({},{},{})'.format((np.mean((Ytest-clfAllBaseline.predict(Xtest))**2))**0.5, clfAllBaseline.score(Xtrain,Ytrain), clfAllBaseline.score(Xtest,Ytest)))
print('Tuned Classifier (RMSE, Train Score, Test Score): ({},{},{})'.format((np.mean((Ytest-clfAll.predict(Xtest))**2))**0.5, clfAll.score(Xtrain,Ytrain), clfAll.score(Xtest,Ytest)))

# Conclusion: I first imputed the movie ratings dataset of 400 movies by filling the nan values with the movie median rating for each movie.
# Then I removed participants who didn't answer all of the categorical questions (gender, only child, and viewing preference) because we only lost 151 participants, which I would consider the best option
# since imputation with the column median on the questions doesn't seem like a good representation of each participant's potential answers. I applied "One Hot Encoding" to the categorical variables
# so that the numerical values associated with each category wouldn't cause the model to intrepret them as numerically larger than one another since that wouldn't make sense. I dropped the no response (-1) from the only child and viewing preference categories
# and self-described (3) from the gender category as dummy variables to avoid multicollinearity. After splitting the movie ratings data (to be predicted) and categorical data via one hot encoding with all the other questions (predictors) into a 20% test dataset and 80% training dataset, 
# I then decided to build a random forest regressor supervised learning model to predict a participants movie ratings based on their answers to the questions. In order to obtain a better model, I tuned selected hyper parameters
# (Number of trees from 2 to 1024, Maximum depth from 2 to 256, Minimum sample leaf from 2 to 12, and Minimum sample split from 2 to 12) using the "RandomizedSearchCV" class from the sklearn library, which randomly sampled (without replacement)
# the hyper parameters 200 times without replacement using a 5-fold cross validation (computing the average score on 5 different samples of n-1 training data with 1 test data). Note that I used a regressor model to deal with the non-integer movie ratings rather than a classifier.
# I finally compared the tuned model to a baseline model and found that it had better results as seen in the figure. Although the baseline model was still slightly worse than just guessing, the tuned model was actually slightly better than just guessing which is unsurprising 
# given how many more predictors there were in this model compared to the other models in questions 8 and 9. What is interesting is that the tuned model, which is underfitting the training set actually performed better than the baseline model which fit the training set well. 
# I also am not surprised by these results because I don't expect a model with a number of predictors that is much less than the number of output variables to be a good model. There are just too few predictors for this model.