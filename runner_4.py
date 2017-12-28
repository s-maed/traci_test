#!/usr/bin/env python

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
    pWE = 1. / 10
    pEW = 1. / 7
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
    
    # Initialize QLearning instance
    phases = [0, 2]                # 信号のフェーズのうち、0と2のどちらかをとる
    num_lane_occupancy_states = 10 # 各レーンの混雑具合のレベル数
    num_lanes = 2                  # レーンの数（南北で一つ、東西で一つ）
    min_elapsed_time =  5          # 信号の最小点灯時間
    max_elapsed_time = 40          # 信号の最大点灯時間
    actions = [0, 1]               # 取りうるアクションのインデックス

    # Q tableを保存してあるcsvファイルを指定
    # まっさらな状態から始めるときは何も指定しない（"" or None）
    q_table_model = ""

    q = QLearning(phases, num_lane_occupancy_states, num_lanes, min_elapsed_time, max_elapsed_time, actions, q_table_model)


    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # 10000ステップごとにrewardをプロットする
        step += 1
        if step % 50000 == 0:
            plot_graph(q.rewards)
            # ここまでのQ tableを保存
            np.savetxt("data/q_table/q_table_{}.csv".format(step), q.q_table, delimiter=",")

        # 現在の信号のフェーズ
        light_phase = traci.trafficlights.getPhase("0")

        # もし黄色信号のフェーズだったら次のステップに進む
        if light_phase == 1 or light_phase == 3:
            # 直前の青信号の情報を記憶する
            if q.is_set_max_duration:
                q.prev_t = step + 7 # 黄色信号の点灯時間分+1ステップを足しておく
                q.is_set_max_duration = False
                ns_length = traci.lanearea.getJamLengthMeters("0") + traci.lanearea.getJamLengthMeters("2")
                ew_length = traci.lanearea.getJamLengthMeters("1") + traci.lanearea.getJamLengthMeters("3")
                q.max_length_prev_t = ns_length + ew_length

                # 信号が1サイクル回ったら、そのサイクルのリワードの合計を記憶
                if light_phase == 3:
                    q.rewards.append(q.cycle_rewards)
                    q.cycle_rewards = 0
            # DEBUG
            print("Step:", step)
            print("Yellow light phase")
            print()
            continue

        # もし青フェーズになったばかりだったら、点灯時間の最大値をセットする
        # ミリ秒単位でセットするので、40 * 1000
        if not q.is_set_max_duration:
            traci.trafficlights.setPhaseDuration("0", q.max_elapsed_time*1000)
            q.is_set_max_duration = True

        # もし青フェーズの最低点灯時間に達していなかったら、そのまま次のステップに進む
        if (step - q.prev_t) < q.min_elapsed_time:
            continue

        # observation（現在のstate）
        # 南北と東西のそれぞれのレーンで一番混んでいる状況を取得
        lane_length = traci.lanearea.getLength("0")
        ns_occupancy = max(traci.lanearea.getJamLengthMeters("0")/lane_length, traci.lanearea.getJamLengthMeters("2")/lane_length)
        ew_occupancy = max(traci.lanearea.getJamLengthMeters("1")/lane_length, traci.lanearea.getJamLengthMeters("3")/lane_length)
        elapsed_time = min(step - q.prev_t, max_elapsed_time-1)
        observation = q.digitize_state(light_phase, ns_occupancy, ew_occupancy, elapsed_time)

        # reward
        # 各レーンのキューの長さをもとに計算
        ns_length = traci.lanearea.getJamLengthMeters("0") + traci.lanearea.getJamLengthMeters("2")
        ew_length = traci.lanearea.getJamLengthMeters("1") + traci.lanearea.getJamLengthMeters("3")
        reward = q.calculate_reward(ns_length, ew_length)
        q.cycle_rewards += reward

        # DEBUG
        print("Step:", step)
        print("Current Phase:", light_phase)
        print("Elapsed Time:", elapsed_time)
        print("reward:", reward)
        print()

        # 前ステップのstateとactionによって得られたrewardとobservationによってQ tableを更新する
        q.update_Qtable(q.state, q.action, reward, observation)

        # 1秒後のアクションを判断する
        action = q.get_action(observation)

        # 現状のアクションと状態を保存
        q.action = action
        q.state = observation

        # もし次にとるべきフェーズが次のフェーズと異なるなら、次のフェーズに移る黄色信号フェーズにセットする
        if phases[action] != light_phase:
            traci.trafficlights.setPhase("0", light_phase+1)

    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# rewardをプロットする
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
