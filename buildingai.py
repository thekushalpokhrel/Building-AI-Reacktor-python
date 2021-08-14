#listing pineapple routes----------------


portnames = ["PAN", "AMS", "CAS", "NYC", "HEL"]
def permute(route, ports):
    i = 0
    for j in range(1, 5):
        for k in range(1, 5):
            for l in range(1, 5):
                for m in range(1, 5):
                    route = [j, k, l, m]

                    if all(elem in route for elem in ports):
                        route = [i, j, k, l, m]
                        print(' '.join([portnames[i] for i in route]))

permute([0], list(range(1, len(portnames))))

#### SVAR
def permutations(route, ports):
    if len(ports) < 1:
        print(' '.join([portnames[i] for i in route]))
    else:
        for i in range(len(ports)):
            permutations(route+[ports[i]], ports[:i]+ports[i+1:])
 
permutations([0], list(range(1, len(portnames))))

#pineapple route emissions-------------------

portnames = ["PAN", "AMS", "CAS", "NYC", "HEL"]

# https://sea-distances.org/
# nautical miles converted to km

smallest = 1000000
bestroute = [0, 0, 0, 0, 0]
D = [
    [0,8943,8019,3652,10545],
    [8943,0,2619,6317,2078],
    [8019,2619,0,5836,4939],
    [3652,6317,5836,0,7825],
    [10545,2078,4939,7825,0]
]

# https://timeforchange.org/co2-emissions-shipping-goods
# assume 20g per km per metric ton (of pineapples)

co2 = 0.020

# DATA BLOCK ENDS

# these variables are initialised to nonsensical values
# your program should determine the correct values for them


def permutations(route, ports):
    global smallest, bestroute
    # write the recursive function here
    # remember to calculate the emissions of the route as the recursion ends
    # and keep track of the route with the lowest emissions
    if len(ports) < 1:
        shortest = 1000000
        count = D[0][route[1]]
        count += D[route[1]][route[2]]
        count += D[route[2]][route[3]]
        count += D[route[3]][route[4]]
        if count < smallest:
            smallest = count
            bestroute = route
    else:
        for i in range(len(ports)):
            permutations(route+[ports[i]], ports[:i]+ports[i+1:])

    pass

def main():
    global smallest
    # this will start the recursion 
    permutations([0], list(range(1, len(portnames))))
    smallest = smallest*co2
    # print the best route and its emissions
    print(' '.join([portnames[i] for i in bestroute]) + " %.1f kg" % smallest)

main()

#reach the highest summit--------------

import math
import random             	# just for generating random mountains                                 	 

# generate random mountains                                                                               	 

w = [.05, random.random()/3, random.random()/3]
h = [1.+math.sin(1+x/.6)*w[0]+math.sin(-.3+x/9.)*w[1]+math.sin(-.2+x/30.)*w[2] for x in range(100)]

def climb(x, h):
    # keep climbing until we've found a summit
    summit = False

    # edit here
    while not summit:
        summit = True         # stop unless there's a way up
        if h[x + 1] > h[x]:
            x = x + 1         # right is higher, go there
            summit = False    # and keep going
        else:
            for x_new in range(max(0, x-5), min(99, x+5)):
                if h[x_new] > h[x]:
                    summit = False
                    x = x_new
    return x


def main(h):
    # start at a random place                                                                                  	 
    x0 = random.randint(1, 98)
    x = climb(x0, h)
    return x0, x

main(h)

#probabilities-------------

import random

def main():
    prob = random.random()
    if prob < 0.1:
        favourite = "bats"
    if prob < 0.2 and prob > 0.1:
        favourite = "cats"
    if prob > 0.2:
        favourite = "dogs"
      # change this
    print("I love " + favourite) 


main()

#Warm-up Temperature---------------

import random
import numpy as np

def accept_prob(S_old, S_new, T):
    # this is the acceptance "probability" in the greedy hill-climbing method
    # where new solutions are accepted if and only if they are better
    # than the old one.
    # change it to be the acceptance probability in simulated annealing

    if S_new > S_old:
        return 1.0
    else:   
    #return 0.0
        return np.exp(-(S_old - S_new) / T)

# the above function will be used as follows. this is shown just for
# your information; you don't have to change anything here
def accept(S_old, S_new, T):
    if random.random() < accept_prob(S_old, S_new, T):
        print(True)
    else:
        print(False)
accept(150, 140, 5)

#simulated annealing----------

import numpy as np
import random

N = 100     # size of the problem is N x N                                      
steps = 3000    # total number of iterations                                        
tracks = 50

# generate a landscape with multiple local optima                                          
def generator(x, y, x0=0.0, y0=0.0):
    return np.sin((x/N-x0)*np.pi)+np.sin((y/N-y0)*np.pi)+\
        .07*np.cos(12*(x/N-x0)*np.pi)+.07*np.cos(12*(y/N-y0)*np.pi)

x0 = np.random.random() - 0.5
y0 = np.random.random() - 0.5
h = np.fromfunction(np.vectorize(generator), (N, N), x0=x0, y0=y0, dtype=int)
peak_x, peak_y = np.unravel_index(np.argmax(h), h.shape)

# starting points                                                               
x = np.random.randint(0, N, tracks)
y = np.random.randint(0, N, tracks)

def main():
    global x
    global y

    for step in range(steps):
        # add a temperature schedule here
        T = max(0, ((steps - step)/steps)**3-.005)
        # update solutions on each search track                                     
        for i in range(tracks):
            
            # try a new solution near the current one                               
            x_new = np.random.randint(max(0, x[i]-2), min(N, x[i]+2+1))
            y_new = np.random.randint(max(0, y[i]-2), min(N, y[i]+2+1))
            S_old = h[x[i], y[i]]
            S_new = h[x_new, y_new]

            # change this to use simulated annealing
            if S_new > S_old:
                x[i], y[i] = x_new, y_new   # new solution is better, go there       
            else:
                if random.random() < (np.exp(-(S_old - S_new) / T)):                     # if the new solution is worse, do nothing
                   x[i], y[i] = x_new, y_new  
    # Number of tracks found the peak
    print(sum([x[j] == peak_x and y[j] == peak_y for j in range(tracks)])) 
main()

# Flip the coin-------------
#The probability of "11111" at any given position in the sequence can be calculated as (2/3)^5 ≈ 0.13169. 
#The number of occurrences is close to 10000 times this: 1316.9. To be more precise, the expected number of occurrences 
#is about 0.13169 x 9996 ≈ 1316.3, because there are only 9996 places for a subsequence of length five in a sequence of 10000. 
#The actual number will usually (in fact, with over 99% probability) be somewhere between 1230 and 1404. We check the solution 
#allowing for an even wider margin that covers 99.99% of the cases.
import numpy as np

def generate(p1):
    # change this so that it generates 10000 random zeros and ones
    # where the probability of one is p1
    seq = np.random.choice([0,1],p=[1-p1, p1], size=9996)
    return seq

def count(seq):
    amount = 0
    i = 0
    while i < len(seq):
        if seq[i:i+5].all():
            amount += 1
        i += 1
    return amount

def main(p1):
    seq = generate(p1)
    return count(seq)

print(main(2/3))

#fishing in the nordics-------------

countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
populations = [5615000, 5439000, 324000, 5080000, 9609000]
male_fishers = [1822, 2575, 3400, 11291, 1731]
female_fishers = [69, 77, 400, 320, 26] 

def guess(winner_gender):
    if winner_gender == 'female':
        fishers = female_fishers
    else:
        fishers = male_fishers

    # write your solution here
    total_Population = 0
    for i in populations:
        total_Population += i
    fishers_Total = 0
    for i in fishers:
        fishers_Total += i
    j = 0
    temp = (populations[0] / total_Population) * (fishers[0] / populations[0])

    while j < len(populations):
        
        if (populations[j] / total_Population) * (fishers[j] / populations[j]) > temp:
            temp = (populations[j] / total_Population) * (fishers[j] / populations[j])
            guess = countries[j]
            biggest = fishers[j] / fishers_Total * 100
        j+=1
    
    return (guess, biggest)  

def main():
    country, fraction = guess("male")
    print("if the winner is male, my guess is he's from %s; probability %.2f%%" % (country, fraction))
    country, fraction = guess("female")
    print("if the winner is female, my guess is she's from %s; probability %.2f%%" % (country, fraction))

main()

#Block or not-----------


def bot8(pbot, p8_bot, p8_human):
    phuman = 1 - pbot
    p8 = p8_bot * pbot + p8_human * phuman
    pbot_8 = (p8_bot * pbot) / (p8)
    print(pbot_8)

# you can change these values to test your program with different values
pbot = 0.1
p8_bot = 0.8
p8_human = 0.05

bot8(pbot, p8_bot, p8_human)

#Naves Bayes classifier------------

import numpy as np

p1 = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]   # normal
p2 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]   # loaded

def roll(loaded):
    if loaded:
        print("rolling a loaded die")
        p = p2
    else:
        print("rolling a normal die")
        p = p1

    # roll the dice 10 times
    # add 1 to get dice rolls from 1 to 6 instead of 0 to 5
    sequence = np.random.choice(6, size=10, p=p) + 1 
    for roll in sequence:
        print("rolled %d" % roll)
        
    return sequence

def bayes(sequence):
    odds = 1.0           # start with odds 1:1
    for roll in sequence:
        if roll == 6:
            r = 0.5 / (1/6)
        else:
            r = 0.1 / (1/6)
        odds = odds * r
    if odds > 1:
        return True
    else:
        return False

sequence = roll(True)
if bayes(sequence):
    print("I think loaded")
else:
    print("I think normal")
    np.ndarray(sequence)
    
#Real Estate price predictions

# input values for three mökkis: size, size of sauna, distance to water, number of indoor bathrooms, 
# proximity of neighbors
X = [[66, 5, 15, 2, 500], 
     [21, 3, 50, 1, 100], 
     [120, 15, 5, 2, 1200]]
c = [3000, 200, -50, 5000, 100]    # coefficient values

def predict(X, c):
    price = 0 
    for cabin in X:
        for coefficient in c:
            price += cabin[c.index(coefficient)]*coefficient
        print(price)
        price = 0
               
predict(X, c)

#Least squares-------------

import numpy as np

# data
X = np.array([[66, 5, 15, 2, 500],
              [21, 3, 50, 1, 100],
              [120, 15, 5, 2, 1200]])
y = np.array([250000, 60000, 525000])

# alternative sets of coefficient values
c = np.array([[3000, 200 , -50, 5000, 100],
              [2000, -250, -100, 150, 250],
              [3000, -100, -150, 0, 150]])

def find_best(X, y, c):
    smallest_error = np.Inf
    best_index = 0
    
    x_and_c = []
    for coeff in c:
        x_and_c.append(X @ coeff)
    x_and_X = np.array(x_and_c)
    coefficient = []
    for i in x_and_X:
        coefficient.append([(n - j)**2 for n, j in zip(y, i)])
    number = sum(coefficient[0])
    for i in coefficient:
        
        if sum(i)<number:
            number = sum(i)
            best_index = coefficient.index(i)


    print("the best set is set %d" % best_index)


find_best(X, y, c)

#Prediction with more data-----------------

import numpy as np
from io import StringIO

input_string = '''
25 2 50 1 500 127900
39 3 10 1 1000 222100
13 2 13 1 1000 143750
82 5 20 2 120 268000
130 6 10 2 600 460700
115 6 10 1 550 407000
'''

np.set_printoptions(precision=1)    # this just changes the output settings for easier reading
 
def fit_model(input_file):

    data = np.genfromtxt(input_file, skip_header=1)
    c = np.asarray([])
    x = np.asarray([])
    y = np.asarray([])

    i = len(data) - 1
    while i >= 0:
        last_element = data[i][-1]
        y = np.insert(y, 0, last_element, axis = 0)
        i-=1
    c = data[:,:-1]
    new = np.linalg.lstsq(c, y)[0]
    print(new)
    print(c @ new)

# simulate reading a file
input_file = StringIO(input_string)
fit_model(input_file)

#training data vs test data

import numpy as np
from io import StringIO


train_string = '''
25 2 50 1 500 127900
39 3 10 1 1000 222100
13 2 13 1 1000 143750
82 5 20 2 120 268000
130 6 10 2 600 460700
115 6 10 1 550 407000
'''

test_string = '''
36 3 15 1 850 196000
75 5 18 2 540 290000
'''

def main():
    np.set_printoptions(precision=1)

    data_train = np.genfromtxt(train, skip_header=1)
    data_test = np.genfromtxt(test, skip_header=1)

    x_train = data_train[:,:-1]
    y_train = np.asarray([])

    i = len(data_train) - 1
    while i >= 0:
        last_element = data_train[i][-1]
        y_train = np.insert(y_train, 0, last_element, axis = 0)
        i-=1

    x_test = data_test[:,:-1]

    coeff = np.linalg.lstsq(x_train, y_train)[0]

    print(coeff)
    print(x_test @ coeff)

train = StringIO(train_string)
test = StringIO(test_string)

main()


#Vector distances-------------

import numpy as np

x_train = np.random.rand(10, 3)   # generate 10 random vectors of dimension 3
x_test = np.random.rand(3)        # generate one more random vector of the same dimension

def dist(a, b):
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return np.sqrt(sum)
    
def nearest(x_train, x_test):
    nearest = 0
    min_distance = np.Inf
    x = 0
    min_distance = dist(x_test, x_train[0])
    for i in x_train:
        distance = dist(i, x_test)
        if distance < 0:
            abs(distance)
        if min_distance > distance:
            min_distance = distance
            nearest = x
        x+=1
            
    
    # add a loop here that goes through all the vectors in x_train and finds the one that
    # is nearest to x_test. return the index (between 0, ..., len(x_train)-1) of the nearest
    # neighbor
    print(nearest)

nearest(x_train, x_test)

#nearest neighbour ----------------------

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


X, Y = make_blobs(n_samples=16, n_features=2, centers=2, center_box=(-2, 2))
X = MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=2)
y_predict = np.empty(len(y_test), dtype=np.int64)

lines = []


def dist(a, b):
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return np.sqrt(sum)


def main(X_train, X_test, y_train, y_test):
    global y_predict
    global lines
    k = 3

    for i, test_item in enumerate(X_test):

        distances = [dist(train_item, test_item) for train_item in X_train]

        print(distances)
        nearest_indices = np.argsort(distances)
        print(nearest_indices)
        nearest_labels = y_train[nearest_indices[:k]]
        print(nearest_labels)

        y_predict[i] = np.round(np.mean(nearest_labels))
    print(y_predict)

main(X_train, X_test, y_train, y_test)

#bagof words----------------

import numpy as np
data = [[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]]
def find_nearest_pair(data):
    N = len(data)
    dist = np.inf
    dist = np.empty((N, N), dtype=np.float)
    
    for i in range(N):
        for j in range(N):
            sum = 0
            for k in range(len(data[i])):
                
                sum += (data[j][k] - data[i][k])
                dist[i][j] = sum
    print(dist)
    print(np.unravel_index(np.argmin(dist), dist.shape))
find_nearest_pair(data)

#looking out for overfitting---------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

# do not edit this
# create fake data
x, y = make_moons(
    n_samples=500,  # the number of observations
    random_state=42,
    noise=0.3
)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(x_train)
# Create a classifier and fit it to our data
knn = KNeighborsClassifier(n_neighbors=133)
knn.fit(x_train, y_train)

y_predict = np.empty(len(y_test), dtype=np.int64)

lines = []

def dist(ai, bi):
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return np.sqrt(sum)

def main(X_train, X_test, y_train, y_test):
    global y_predict
    global lines
    k = 3

    for i, test_item in enumerate(X_test):

        distances = [dist(train_item, test_item) for train_item in X_train]
        nearest_indices = np.argsort(distances)
        nearest_labels = y_train[nearest_indices[:k]]


        y_predict[i] = np.round(np.mean(nearest_labels))
    print(y_predict)

main(x_train, y_train, x_test, y_test)

print("training accuracy: %f" % 0.0)
print("testing accuracy: %f" % 0.0)

#Logistic regression-------------

import math
import numpy as np

x = np.array([4, 3, 0])
c1 = np.array([-.5, .1, .08])
c2 = np.array([-.2, .2, .31])
c3 = np.array([.5, -.1, 2.53])

def sigmoid(c1):
    # add your implementation of the sigmoid function here
    test = sum(c1)
    print(test)

# calculate the output of the sigmoid for x with all three coefficients

sigmoid(c3)

#Neural networks------------------


import numpy as np

w0 = np.array([[ 1.19627687e+01,  2.60163283e-01],
               [ 4.48832507e-01,  4.00666119e-01],
                   [-2.75768443e-01,  3.43724167e-01],
                   [ 2.29138536e+01,  3.91783025e-01],
                   [-1.22397711e-02, -1.03029800e+00]])

w1 = np.array([[11.5631751 , 11.87043684],
                   [-0.85735419,  0.27114237]])

w2 = np.array([[11.04122165],
                   [10.44637262]])

b0 = np.array([-4.21310294, -0.52664488])
b1 = np.array([-4.84067881, -4.53335139])
b2 = np.array([-7.52942418])

x = np.array([[111, 13, 12, 1, 161],
                 [125, 13, 66, 1, 468],
                 [46, 6, 127, 2, 961],
                 [80, 9, 80, 2, 816],
                 [33, 10, 18, 2, 297],
                 [85, 9, 111, 3, 601],
                 [24, 10, 105, 2, 1072],
                 [31, 4, 66, 1, 417],
                 [56, 3, 60, 1, 36],
                 [49, 3, 147, 2, 179]])
y = np.array([335800., 379100., 118950., 247200., 107950., 266550.,  75850.,
                93300., 170650., 149000.])


def hidden_activation(z):
    # ReLU activation. fix this!
        return 0

def output_activation(z):
    # identity (linear) activation. fix this!
        return 0

x_test = [[82, 2, 65, 3, 516]]
for item in x_test:
    h1_in = np.dot(item, w0) + b0 # this calculates the linear combination of inputs and weights
    h1_out = hidden_activation(h1_in) # apply activation function
    
    # fill out the missing parts:
    # the output of the first hidden layer, h1_out, will need to go through
    # the second hidden layer with weights w1 and bias b1
    # and finally to the output layer with weights w2 and bias b2.
    # remember correct activations: relu in the hidden layers and linear (identity) in the output
    
    print(0)
    
 

