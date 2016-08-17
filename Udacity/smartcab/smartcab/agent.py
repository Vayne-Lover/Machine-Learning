import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        self.color = 'red'
        super(LearningAgent, self).__init__(env)
        self.planner = RoutePlanner(self.env, self)
        
        # TODO: Initialize any additional variables here
        self.Q0=3
        self.QSA={}
        self.init=True
        self.alpha=0.5
        self.epsilon=0.1
        self.gamma=0.3
        self.totalSteps=0
        self.totalRewards=0
        
        self.lastState=None
        self.lastAction=None
        self.lastReward=None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        # TODO: Update state
        self.state=(self.next_waypoint,inputs['light'],inputs['oncoming'],inputs['left'],inputs['right'])
        
        # TODO: Select action according to your policy
        QSA=self.QSA
        if self.state not in self.QSA.keys():
            self.QSA[self.state]={None:self.Q0,'forward':self.Q0,'left':self.Q0,'right':self.Q0}
            action = random.choice([None, 'forward', 'left', 'right'])
        else:
            if random.random()>self.epsilon:
                bestAction={action:Q for action, Q in QSA[self.state].items() if Q == max(QSA[self.state].values())}
                action=random.choice(bestAction.keys())
            else:
                action=random.choice([None, 'forward', 'left', 'right'])
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        if self.init==False:
            QLast=QSA[self.lastState][self.lastAction]
            QLast=QLast*(1-self.alpha)+self.alpha*(self.lastReward+self.gamma*(max(QSA[self.state].values())))
            self.QSA[self.lastState][self.lastAction]=QLast
        
        self.init=False
        self.totalSteps+=1
        self.lastAction=action
        self.lastState=self.state
        self.lastReward=reward
        self.totalRewards+=reward

def run():
    """Run the agent for a finite number of trials."""
    trialCount=100
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    # Now simulate it
    sim = Simulator(e, update_delay=0.001)  # create simulator (uses pygame when display=True, if available)
    
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    sim.run(n_trials=trialCount)  # run for a specified number of trials
    
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print a.totalSteps
    print a.totalRewards
    print e.successCount

if __name__ == '__main__':
    run()
