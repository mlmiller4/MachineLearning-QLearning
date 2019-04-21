"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			    		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			    		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			    		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
Student Name: Matthew Miller
GT User ID: mmiller319
GT ID: 903056227
"""  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import random as rand  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
class QLearner(object):  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			    		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  		   	  			    		  		  		    	 		 		   		 		  
        self.s = 0  		   	  			    		  		  		    	 		 		   		 		  
        self.a = 0

        # Add other variables to constructor
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr =radr
        self.dyna = dyna
        self.verbose = verbose

        # Initialize the Q matrix
        self.Q = np.zeros([num_states, num_actions])

        """ Initialize the T, Tc, and R matrices for Dyna-Q """
        # T matrix, T[s,a,s'], holds the probability that state s and action a will result in state s'
        self.T = np.zeros((num_states, num_actions, num_states))

        # Tc matrix, Tc[s,a,s'], holds the number of times that s,a -> s' is observed.
        # The value Tc[s, a, s_prime] will be incremented by one every time state s and action a result in s_prime.
        # Tc is initialized with 0.00001 to avoid possible divide-by-zero errors.
        self.Tc = np.full((num_states, num_actions, num_states), 0.00001)

        # R[s,a] is the expected reward for state s and action a.  R[s,a] will be updated every time s,a is observed
        # with the equation: R'[s,a] = (1 - alpha) * R[s,a] + alpha * r
        # where r = immediate reward in an experience tuple containing s,a
        self.R = np.zeros((num_states, num_actions))

  		   	  			    		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			    		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        self.s = s  		   	  			    		  		  		    	 		 		   		 		  
        action = rand.randint(0, self.num_actions-1)  		   	  			    		  		  		    	 		 		   		 		  
        if self.verbose: print "s =", s,"a =",action  		   	  			    		  		  		    	 		 		   		 		  
        return action  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    def query(self,s_prime,r):  		   	  			    		  		  		    	 		 		   		 		  
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			    		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """

        """ Update Q[s,a] according to the equation:
            Q[s,a] = (1-alpha) * Q[s,a] + alpha * (r + gamma * Q[s', argmax(Q[s'])]) 
        """
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
                    r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])])


        """ Check for Dyna cases (if dyna != 0), and implement Dyna-Q algorithm according to
            MC3 Lesson 7 """
        """----------------------------------------------------------------------------------------------------"""
        if self.dyna != 0:

            # Increment Tc[s, a, s'] by one - this tracks the number of times that [s,a,s'] have occurred,
            # meaning that state s and action a have resulted in state s'
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1

            # Calculate probabilities that s and a will lead to s' using eqn:
            # T[s,a,s'] = Tc[s,a,s'] / sum(T[s,a,i]
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)

            # Update R matrix (expected reward) using eqn: R'[s,a] = (1-alpha) * R[s,a] + alpha * r
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            """ Hallucinate a number of experiences equal to dyna value"""
            for i in range(0, self.dyna):

                # Hallucinate by choosing a random state and a random action
                randomState = rand.randint(0, self.num_states - 1)  # random state
                randomAction = rand.randint(0, self.num_actions - 1)  # random action

                # Infer s' from T matrix using random state and random action: argmax of T[random state, random action]
                inferred_s_prime = np.argmax(self.T[randomState, randomAction])

                # Get expected reward of that random state and action
                expected_r = self.R[randomState, randomAction]

                # Use update equation with randomState, randomAction, inferred_s_prime and expected_r:
                self.Q[randomState, randomAction] = (1 - self.alpha) * self.Q[randomState, randomAction] + self.alpha * (
                            expected_r + self.gamma * self.Q[inferred_s_prime, np.argmax(self.Q[inferred_s_prime])])

        """--------------------------------------------------------------------------------------------------------------"""

        # Generate a random value between 0 and 1 which will be compared to the random action rate (rar)
        # to determine whether a random action is taken or not
        randValue = rand.uniform(0,1)

        # if random value is less than rar, then a random action will be taken
        # Otherwise, the action taken is the argmax of Q[s']
        if randValue < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])

        # Set rar value to current rar multiplied by random action decay rate (radr)
        self.rar = self.rar * self.radr


        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        # Set new values of s and a
        self.s = s_prime
        self.a = action

        # return chosen action
        return action




    def author(self):
        return 'mmiller319'
  		   	  			    		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			    		  		  		    	 		 		   		 		  
    print "Remember Q from Star Trek? Well, this isn't him"  		   	  			    		  		  		    	 		 		   		 		  
