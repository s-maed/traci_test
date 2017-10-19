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
from q_learning import QLearning


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
    N = 200000  # number of time steps
    # demand per second from different directions
    pWE = 1. / 18
    pEW = 1. / 15
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
    num_phase = 2
    max_num_car_stopped = 10
    num_lane = 4
    num_wait_time_category = 10
    num_action = 10
    q = QLearning(num_phase, max_num_car_stopped, num_lane, num_action)

    # we start with phase 2 where EW has green
    #traci.trafficlight.setPhase("0", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        #next_action_idx = 9

        # 現在の信号のフェーズ
        light_phase = traci.trafficlight.getPhase("0")

        # 現在のフェーズが黄色かつまだ次のアクションを決めていなかったら、次のフェーズの秒数を決める
        if (light_phase == 1 or light_phase == 3) and not q.is_calculate_next_action:

            q.is_set_duration = False

            # 次に信号が取るフェーズを取得
            next_light_phase = 0
            if light_phase == 1:
                next_light_phase = 2

            # それぞれのレーンで停まっている車の数
            count_0 = min(traci.lanearea.getLastStepHaltingNumber("0"), 9)
            count_1 = min(traci.lanearea.getLastStepHaltingNumber("1"), 9)
            count_2 = min(traci.lanearea.getLastStepHaltingNumber("2"), 9)
            count_3 = min(traci.lanearea.getLastStepHaltingNumber("3"), 9)

            # 次の信号のフェーズと現在の混雑状況
            current_state_dict = {
                'light_phase': next_light_phase,
                'nums_car_stopped': [count_0, count_1, count_2, count_3]
            }

            current_digitized_state = q.digitize_state(current_state_dict)
            q.next_action_idx = q.get_action(current_digitized_state)
            q.is_calculate_next_action = True

            # reward
            reward = - np.sum([x**1.5 for x in [count_0, count_1, count_2, count_3]])
            q.rewards.append(reward)

            # 各青赤フェーズが終了したタイミングで、以前の状況に対してとったアクションに対するリワードを計算するため、このタイミングで、前回のstateとactionに対するリワードを計算する？

            q.update_Qtable(q.previous_digitized_state, q.previous_action, reward, current_digitized_state)

            q.previous_digitized_state = current_digitized_state
            q.previous_action_idx = q.next_action_idx

        # 現在のフェーズが0か2でかつまだ秒数をセットしていなかったら、秒数をセットする
        if (light_phase == 0 or light_phase == 2) and not q.is_set_duration:
            traci.trafficlight.setPhaseDuration("0", q.action[q.next_action_idx])
            q.is_set_duration = True
            q.is_calculate_next_action = False
            print("set phase {} for {} seconds".format(light_phase, q.action[q.next_action_idx]))

        step += 1
        if step % 10000 == 0:
            plot_graph(q.rewards)

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
