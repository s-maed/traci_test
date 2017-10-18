#!/usr/bin/env python
"""
@file    runner.py
@author  Lena Kalleske
@author  Daniel Krajzewicz
@author  Michael Behrisch
@author  Jakob Erdmann
@date    2009-03-26
@version $Id: runner.py 24864 2017-06-23 07:47:53Z behrisch $

Tutorial for traffic light control via the TraCI interface.

SUMO, Simulation of Urban MObility; see http://sumo.dlr.de/
Copyright (C) 2009-2017 DLR/TS, Germany

This file is part of SUMO.
SUMO is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random
import numpy as np
import time

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci


def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 40000  # number of time steps
    # demand per second from different directions
    pWE = 1. / 25
    pEW = 1. / 15
    pNS = 1. / 10
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        lastVeh = 0
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>


def run():
    """execute the TraCI control loop"""
    num_episode = 10
    step_interval = 10
    num_dizitized = 10
    action_space = 2
    q_table = np.random.uniform(
        low=-1, high=1, size=(num_dizitized, num_dizitized, num_dizitized, num_dizitized, action_space))

    for episode in range(num_episode):
        step = 0
        # we start with phase 2 where EW has green
        traci.trafficlights.setPhase("0", 2)
        # get the num of halting car of each lane
        lane1_halting_num = traci.lane.getLastStepHaltingNumber("1i_0")
        lane2_halting_num = traci.lane.getLastStepHaltingNumber("2i_0")
        lane3_halting_num = traci.lane.getLastStepHaltingNumber("3i_0")
        lane4_halting_num = traci.lane.getLastStepHaltingNumber("4i_0")
        total_halting_num = lane1_halting_num + lane2_halting_num + lane3_halting_num + lane4_halting_num
        if lane1_halting_num > 9:
            lane1_halting_num = 9
        if lane2_halting_num > 9:
            lane2_halting_num = 9
        if lane3_halting_num > 9:
            lane3_halting_num = 9
        if lane4_halting_num > 9:
            lane4_halting_num = 9

        action = np.argmax(q_table[lane1_halting_num, lane2_halting_num, lane3_halting_num, lane4_halting_num])

        # while traci.simulation.getMinExpectedNumber() > 0:
        while step < 1000:
            traci.simulationStep()

            if traci.trafficlights.getPhase("0") == 0:
                if action == 0:
                    # keep green for NS
                    traci.trafficlights.setPhase("0", 0)
                    for i in range(step_interval):
                        traci.simulationStep()
                else:
                    # otherwise try to change green for EW
                    traci.trafficlights.setPhase("0", 1)
                    while traci.trafficlights.getPhase("0") != 2:
                        traci.simulationStep()
                    for i in range(step_interval):
                        traci.simulationStep()

            elif traci.trafficlights.getPhase("0") == 2:
                if action == 0:
                    # change green for NS
                    traci.trafficlights.setPhase("0", 3)
                    while traci.trafficlights.getPhase("0") != 0:
                        traci.simulationStep()
                    for i in range(step_interval):
                        traci.simulationStep()
                else:
                    # otherwise try to keep green for EW
                    traci.trafficlights.setPhase("0", 2)
                    for i in range(step_interval):
                        traci.simulationStep()

            # get the num of halting car of each lane
            next_lane1_halting_num = traci.lane.getLastStepHaltingNumber("1i_0")
            next_lane2_halting_num = traci.lane.getLastStepHaltingNumber("2i_0")
            next_lane3_halting_num = traci.lane.getLastStepHaltingNumber("3i_0")
            next_lane4_halting_num = traci.lane.getLastStepHaltingNumber("4i_0")
            next_total_halting_num = next_lane1_halting_num + next_lane2_halting_num + next_lane3_halting_num + next_lane4_halting_num
            if next_lane1_halting_num > 9:
                next_lane1_halting_num = 9
            if next_lane2_halting_num > 9:
                next_lane2_halting_num = 9
            if next_lane3_halting_num > 9:
                next_lane3_halting_num = 9
            if next_lane4_halting_num > 9:
                next_lane4_halting_num = 9

            reward = - next_total_halting_num
            print(reward)

            q_table = update_Qtable(q_table, action, reward, lane1_halting_num, lane2_halting_num, lane3_halting_num, lane4_halting_num,
                          next_lane1_halting_num, next_lane2_halting_num, next_lane3_halting_num, next_lane4_halting_num)
            action = get_action(q_table, next_lane1_halting_num, next_lane2_halting_num, next_lane3_halting_num, next_lane4_halting_num,
                       episode)
            step += 1
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# Add for reinforcement learning -------------------------
# 行動を求める関数
def get_action(q_table, next_lane1_halting_num, next_lane2_halting_num, next_lane3_halting_num, next_lane4_halting_num, episode):
    # 徐々に最適行動のみをとる、ε-greedy法
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_lane1_halting_num, next_lane2_halting_num,
                                        next_lane3_halting_num, next_lane4_halting_num])
    else:
        next_action = np.random.choice([0, 1])
    return next_action


# Qテーブルを更新する関数
def update_Qtable(q_table, action, reward,
                  lane1_halting_num, lane2_halting_num, lane3_halting_num, lane4_halting_num,
                  next_lane1_halting_num, next_lane2_halting_num, next_lane3_halting_num, next_lane4_halting_num):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q = max(q_table[next_lane1_halting_num, next_lane2_halting_num, next_lane3_halting_num, next_lane4_halting_num, 0],
                    q_table[next_lane1_halting_num, next_lane2_halting_num, next_lane3_halting_num, next_lane4_halting_num, 1])

    q_table[lane1_halting_num, lane2_halting_num, lane3_halting_num, lane4_halting_num, action] = \
            (1 - alpha) * q_table[next_lane1_halting_num, next_lane2_halting_num, next_lane3_halting_num, next_lane4_halting_num, action] + \
            alpha * (reward + gamma * next_Max_Q)
    return q_table


# 各レーンの待ち台数に応じてインデックス値を返す関数
# def get_state_index(lane1_halting_num, lane2_halting_num, lane3_halting_num, lane4_halting_num, traffic_light_phase):
#     # 各レーンの待ち台数が10台以上の場合、待ち台数は9とする。（上限が9台）
#     if lane1_halting_num > 9:
#         lane1_halting_num = 9
#     if lane2_halting_num > 9:
#         lane2_halting_num = 9
#     if lane3_halting_num > 9:
#         lane3_halting_num = 9
#     if lane4_halting_num > 9:
#         lane4_halting_num = 9
#
#     index_num = lane1_halting_num + 10 * lane2_halting_num + 100 * lane3_halting_num + 1000 * lane4_halting_num
#
#     return index_num

# ----------------------------------------------------------

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()