#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2017 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id: runner.py 26301 2017-10-02 20:48:38Z behrisch $

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random

import numpy as np
import matplotlib.pyplot as plt
from q_learning_2 import QLearning


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
    N = 1000000  # number of time steps
    # demand per second from different directions
    pWE = 1. / 15
    pEW = 1. / 8
    pNS = 1. / 30
    pSN = 1. / 40
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="16.67" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />
        <route id="up" edges="53o 3i 4o 54i" />""", file=routes)
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
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="0,1,0"/>' % (
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
    step = 0
    
    # initialize QLearning
    phases = [0, 2]
    num_lane_occupancy_states = 10
    num_lanes = 2
    min_elapsed_time = 10
    max_elapsed_time = 40
    actions = [0, 1]
    q = QLearning(phases, num_lane_occupancy_states, num_lanes, min_elapsed_time, max_elapsed_time, actions)


    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        step += 1
        if step % 10000 == 0:
            plot_graph(q.rewards)

        # 現在の信号のフェーズ
        light_phase = traci.trafficlight.getPhase("0")

        # もし黄色信号のフェーズだったら次のステップに進む
        if light_phase == 1 or light_phase == 3:
            if q.is_set_max_duration:
                q.prev_t = step - 1
                q.is_set_max_duration = False
                ns_length = max(traci.lanearea.getJamLengthMeters("0"), traci.lanearea.getJamLengthMeters("2"))
                ew_length = max(traci.lanearea.getJamLengthMeters("1"), traci.lanearea.getJamLengthMeters("3"))
                q.max_length_prev_t = ns_length + ew_length

                if light_phase == 3:
                    q.rewards.append(q.cycle_rewards)
                    q.cycle_rewards = 0
            continue

        # もし青フェーズになったばかりだったら、点灯時間の最大値をセットする
        if not q.is_set_max_duration:
            traci.trafficlight.setPhaseDuration("0", q.max_elapsed_time*1000)
            q.is_set_max_duration = True

        # もし青フェーズの最低点灯時間に達していなかったら、次のステップに進む
        if (step - q.prev_t) < (q.min_elapsed_time + 6):
            continue

        # observation（現在のstate）
        lane_length = traci.lanearea.getLength("0")
        ns_occupancy = max(traci.lanearea.getJamLengthMeters("0")/lane_length, traci.lanearea.getJamLengthMeters("2")/lane_length)
        ew_occupancy = max(traci.lanearea.getJamLengthMeters("1")/lane_length, traci.lanearea.getJamLengthMeters("3")/lane_length)
        print(ew_occupancy)
        print(traci.lanearea.getJamLengthMeters("1"))
        print(lane_length)
        elapsed_time = min(step - q.prev_t - q.min_elapsed_time, 29)
        observation = q.digitize_state(light_phase, ns_occupancy, ew_occupancy, elapsed_time)

        # reward
        ns_length = max(traci.lanearea.getJamLengthMeters("0"), traci.lanearea.getJamLengthMeters("2"))
        ew_length = max(traci.lanearea.getJamLengthMeters("1"), traci.lanearea.getJamLengthMeters("3"))
        reward = q.calculate_reward(ns_length, ew_length)
        q.cycle_rewards += reward
        print("observation:", observation)
        print("elapsed time:", elapsed_time)
        print("step:", step)
        print("reward:", reward)
        print()

        # 前ステップのstateとactionによって得られたrewardとobservationによってQ tableを更新する
        q.update_Qtable(q.state, q.action, reward, observation)

        # 1秒後のアクションを判断する
        action = q.get_action(observation)
        q.action = action
        q.state = observation


        # もし次にとるべきフェーズが次のフェーズと異なるなら、現在のフェーズの残り時間を1秒にセットする
        if phases[action] != light_phase:
            traci.trafficlight.setPhase("0", light_phase+1)

    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def plot_graph(rewards):
    plt.figure()
    plt.plot(rewards)
    #plt.ylim(-50000, 100000)
    plt.show()

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
