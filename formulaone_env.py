import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

class FormulaOne(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
      #define number of racers
      self.racers = 2
      #define laps in reset method as well for training nn
      self.total_laps = 50

      #define tire lives
      self.soft_life = 10
      self.medium_life = 20
      self.hard_life = 40

      '''
      action space defined as follows
      action 0 = do nothing
      action 1 = pit for soft tires (fastest tire, deteriorates quicker)
      action 2 = pit for medium tire (middle between soft and hard)
      action 3 = pit for hard tire (slowest tire, lasts the longest)
      '''
      self.action_space = spaces.Discrete(4)
      self.racer_obs_keys = ['position','driver','laps_remaining','pace','interval','tire', 'tire_laps','elapsed_race_time','pitstops','tire_compound_flag','pace_difference','old_tire_flag']

      racer_space = spaces.Dict({'position': spaces.Discrete(self.racers),
                                 'driver':spaces.Discrete(self.racers),
                                 'laps_remaining': spaces.Discrete(self.total_laps),
                                 'pace': spaces.Box(low=0.0,high=150.0,shape=(1,1)),
                                 'interval':spaces.Box(low=0.0,high=np.finfo(np.float32).max,shape=(1,1)),
                                 'tire':spaces.Discrete(3), #0 - soft, 1 - medium, 2 - hard
                                 'tire_laps':spaces.Discrete(self.total_laps),
                                 'elapsed_race_time':spaces.Box(low=0.0,high=np.finfo(np.float32).max,shape=(1,1)),
                                 'pitstops':spaces.Discrete(self.total_laps),
                                 'tire_compound_flag':spaces.Discrete(2),
                                 'pace_difference':spaces.Box(low=0.0,high=150.0,shape=(1,1)),
                                 'old_tire_flag':spaces.Discrete(2)}
      )

      self.observation_space = spaces.Tuple((spaces.Discrete(self.racers), racer_space))
      self.state = None



  def update_intervals(self):
      #all based on pace difference
      prev_interval = self.state[1][4] - self.state[0][4]
      pace_diff = self.state[0][3] - self.state[1][3]
      #if second car is faster than the gap, it goes ahead
      self.state[1][4] = prev_interval - pace_diff

      #if new interval is negative, swap drivers and make interval non-negative
      state = self.state
      if self.state[1][4] < 0:
        new_leader_row = np.array(state[1])
        new_trailer_row = np.array(state[0])
        state[0,1:] = new_leader_row[1:]
        state[1,1:] = new_trailer_row[1:]
        state[1][4] = -state[0][4]
        state[0][4] = 0
        self.state = state

      return


  def pace_calculation(self,driver_state):
      # this method determines the pace of the car for the upcoming lap
      current_tire = driver_state[5]
      tire_age = driver_state[6]
      race_laps = driver_state[2]
      lap_percentage_remaining = race_laps/self.total_laps

      #calculate base pace of current tire compound, and "life" of tire
      if current_tire == 0:
          base_pace = 51
          tire_life = self.soft_life
      elif current_tire == 1:
          base_pace = 52
          tire_life = self.medium_life
      elif current_tire == 2:
          base_pace = 53
          tire_life = self.hard_life

      '''
      key assumptions for pace: 
       - fuel makes a difference of about 5 seconds total, goes down linearly throughout the race
       - you lose 0.05 second per lap of tire age if age is less than life, otherwise you lose time exponentially on "old" tires
       - random variance of 0.05 seconds added
      '''
      fuel_diff = 5 * lap_percentage_remaining
      old_tire_age = max(tire_age - tire_life,0)
      sum_to_age = (old_tire_age+1) * (old_tire_age) / 2.0
      tire_diff = 0.05 * min(tire_age,tire_life) + 0.2 * sum_to_age
      random_variance = 0.05 * np.random.random()
      pace = base_pace + fuel_diff + tire_diff + random_variance

      return pace

  def update_pace(self):
      #see pace_calculation for details
      for row in self.state:
          row[3] = self.pace_calculation(driver_state = row)


      return self.state[:,3]

  def pitstop(self,racer,action):
      # pitstops take 20 seconds
      self.state[racer][8] += 1
      self.state[racer][3] += 20
      # check if medium or hard tire was put on to update tire_compound_flag
      if (action > 1) & (self.state[racer][9] == 0):
          self.state[racer][9] = 1

      # put on new tires
      self.state[racer][5] = action - 1
      # reset tire laps to 0
      self.state[racer][6] = 0

      return

  def get_rewards(self):
      #returns rewards of racers in correct order
      rewards = [0,0]
      racer_1_pos = int(np.where(self.state[:, 1] == 1)[0][0])
      rewards[0] = -self.state[racer_1_pos][3]
      rewards[1] = -self.state[1-racer_1_pos][3]

      return rewards

  def get_pace(self,driver_number):
      racer_pos = int(np.where(self.state[:, 1] == driver_number)[0][0])
      pace = self.state[racer_pos][3]
      return pace

  def update_pace_diff(self,driver_number,pace_diff):
      racer_pos = int(np.where(self.state[:, 1] == driver_number)[0][0])
      self.state[racer_pos][10] = pace_diff
      return

  def check_tire_age_flag(self):
      #depending tire compound and laps on current tire, determine if the tire has been on for longer than it's "life"
      for row in self.state:
          #do for each tire compound
          if row[5] == 0:
              #soft tire
              tire_life = self.soft_life
              if row[6] > tire_life:
                  #flag true
                  row[11] = 1
              else:
                  row[11] = 0
          if row[5] == 1:
              #medium tire
              tire_life = self.medium_life
              if row[6] > tire_life:
                  #flag true
                  row[11] = 1
              else:
                  row[11] = 0
          if row[5] == 2:
              #hard tire
              tire_life = self.hard_life
              if row[6] > tire_life:
                  #flag true
                  row[11] = 1
              else:
                  row[11] = 0
      return


  def multi_step(self,actions):
      #environment step function that takes actions of all drivers as argument

      #last pace and next pace done in this loop
      old_pace_1 = self.get_pace(1)
      old_pace_2 = self.get_pace(2)

      # update pace for upcoming lap
      self.update_pace()

      leading_racer = int(self.state[0][1])
      trailing_racer = int(self.state[1][1])

      leading_racer_action = actions[leading_racer-1]
      trailing_racer_action = actions[trailing_racer-1]

      #test to see if trailing racer can pass leading racer
      interval = self.state[1][4]
      pace_diff = self.state[0][3] - self.state[1][3]

      if leading_racer_action > 0:
          self.pitstop(racer = 0,action = leading_racer_action)

      if trailing_racer_action > 0:
          self.pitstop(racer = 1, action = trailing_racer_action)

      #move cars based on pace
      self.update_intervals()
      # take away laps remaining from all cars
      self.state[:, 2] -= 1
      # add tire laps to all cars
      self.state[:, 6] += 1
      #check tire age
      self.check_tire_age_flag()
      # add lap time to elapsed race time
      self.state[:, 7] += self.state[:, 3]

      #do the pace change calculations
      new_pace_1 = self.get_pace(1)
      new_pace_2 = self.get_pace(2)
      pace_diff_1 = new_pace_1-old_pace_1
      pace_diff_2 = new_pace_2 - old_pace_2
      self.update_pace_diff(1, pace_diff_1)
      self.update_pace_diff(2,pace_diff_2)


      state = self.state



      rewards = self.get_rewards()
      done = False
      if self.state[0][2] == 0:
          done = True
      info = {}

      return state, rewards, done, info


  def reset(self,start_laps = 30):
    #resets to beginning of the race with laps as an argument
    self.total_laps = start_laps
    state_array = np.zeros(shape=(self.racers,len(self.racer_obs_keys)))

    i=0
    for row in state_array:
        #position of car
        row[0] = i+1
        #index of racer
        row[1] = i+1
        #lap
        row[2] = start_laps
        #pace
        row[3] = self.pace_calculation(driver_state=row)

        if(i==0):
            #interval to car ahead
            row[4] = 0

        else:
            #set starting intervals at 0.2
            row[4] = 0.2

        #start on soft tire
        row[5] = 0
        # 0 laps on current tire
        row[6] = 0
        # 0 elapsed race time
        #for second driver this is equal to gap to car ahead since they start behind
        row[7] = row[4]
        # 0 pitstops
        row[8] = 0
        # false tire compound flag
        row[9] = 0
        # pace difference
        row[10] = 0
        # false old tire flag
        row[11] = 0
        i += 1

    self.state = state_array


    return self.state

  def get_racer_1_obs(self,state):
      racer_1_pos = int(np.where(state[:, 1] == 1)[0][0])
      racer_1_obs = np.array(state[racer_1_pos])[[0, 2, 3, 5, 6, 8,10,11]]
      return racer_1_obs

  def get_racer_1_obs_interval(self,state):
      racer_1_pos = int(np.where(state[:, 1] == 1)[0][0])
      racer_1_obs = np.array(state[racer_1_pos])[[0, 2, 3, 5, 6, 8,10,11,4,9]]
      return racer_1_obs

  def get_racer_2_obs_interval(self,state):
      racer_2_pos = int(np.where(state[:, 1] == 2)[0][0])
      racer_2_obs = np.array(state[racer_2_pos])[[0, 2, 3, 5, 6, 8,10,11,4,9]]
      return racer_2_obs

  def render(self, mode='human'):
    #prints and returns dataframe that summarizes the current state in the race
    render_df = pd.DataFrame(self.state,columns = self.racer_obs_keys)
    print(render_df)
    return render_df

