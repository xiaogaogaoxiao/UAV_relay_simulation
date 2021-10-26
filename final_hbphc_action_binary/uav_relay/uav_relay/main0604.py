gpu_ids = '0'
## 设置当前程序可见的GPU
import os
if 0 != len(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
# print(os.environ["CUDA_VISIBLE_DEVICES"])
#######################################
import os
import scipy.io as sio
import random
import time
from tqdm import tqdm
import math
from Agent.Agent_Uav import agent
from Algorithm.Greedy_Strategy import greedy_choose
import copy
import torch
import velocity as vv




## 声明变量，如果有GPU可用，将变量放在GPU上

use_cuda = torch.cuda.is_available()
if (0 != len(gpu_ids)) and use_cuda:
    print('run on GPU, num of GPU is',torch.cuda.device_count())
else:
    print('run on CPU')


# ============= Environment Parameters =============#
from environment import *

# ============== Algorithm Parameters ==============#
max_episode = 1000
max_time = 1500
hot_booting = 1
hot_max_episode = 1
hot_max_time = 2000
HOT_TIMES = 3000
# ==================== 存储设定 ====================#
mat_res_path = './Results/mat_results'
img_res_path = './Results/img_results'
time_str = time.strftime("%Y%m%d_%H%M%S")

States_level = {
    'sinr_user2bs1': sinr_user2bs1_set,
    'sinr_user2uav': sinr_user2uav_set,
}

count = 1600
# for k, v in States_level.items():
#     print(k,v)
#     count *= len(v)

State_Number_J = count
State_Number_U = count

Action_Number_J = len(P_jam_set)
Action_Number_U = 2


def main():
    global Q_table_hb, pi_table_hb
    alg_name = 'Q-learning'
    """构造存储矩阵"""
    Save_utility = np.zeros((max_episode, max_time))
    Save_sinr = np.zeros((max_episode, max_time))
    Save_ber = np.zeros((max_episode, max_time))
    Save_energy = np.zeros((max_episode, max_time))
    Save_jam_power = np.zeros((max_episode, max_time))
    Save_delay = np.zeros((max_episode, max_time))

    Save_utility_Q = np.zeros((max_episode, max_time))
    Save_sinr_Q = np.zeros((max_episode, max_time))
    Save_ber_Q = np.zeros((max_episode, max_time))
    Save_energy_Q = np.zeros((max_episode, max_time))
    Save_jam_power_Q = np.zeros((max_episode, max_time))
    Save_delay_Q = np.zeros((max_episode, max_time))

    Save_utility_phc = np.zeros((max_episode, max_time))
    Save_sinr_phc = np.zeros((max_episode, max_time))
    Save_ber_phc = np.zeros((max_episode, max_time))
    Save_energy_phc = np.zeros((max_episode, max_time))
    Save_jam_power_phc = np.zeros((max_episode, max_time))
    Save_delay_phc = np.zeros((max_episode, max_time))



    save_dict = {
        "Reward": Save_utility,
        "Sinr": Save_sinr,
        "BER": Save_ber,
        "Energy": Save_energy,
        "jam_power": Save_jam_power,
        "Delay": Save_delay,
        "Reward_Q": Save_utility_Q,
        "Sinr_Q": Save_sinr_Q,
        "BER_Q": Save_ber_Q,
        "Energy_Q": Save_energy_Q,
        "jam_power_Q": Save_jam_power_Q,
        "Delay_Q": Save_delay_Q,
        "Reward_phc": Save_utility_phc,
        "Sinr_phc": Save_sinr_phc,
        "BER_phc": Save_ber_phc,
        "Energy_phc": Save_energy_phc,
        "jam_power_phc": Save_jam_power_phc,
        "Delay_phc": Save_delay_phc,
    }

    """实例化"""
    Agent_U = agent(State_Number_U, Action_Number_U)
    Agent_U_Q = agent(State_Number_U, Action_Number_U)
    Agent_U_phc = agent(State_Number_U, Action_Number_U)
    sum_qtable = np.ones((State_Number_U, Action_Number_U)) * 0
    sum_pitable = np.ones((State_Number_U, Action_Number_U)) * 0
    if hot_booting:

        for hot_episode in tqdm(range(hot_max_episode), ascii=0, desc='Hot_Episode'):
            Agent_U.reset()

            h_user2bs1 = np.random.choice(vv.h_user2bs1_set, p=vv.p_h_user2bs1_set)
            h_user2uav = np.random.choice(vv.h_user2uav_set, p=vv.p_h_user2uav_set)
            if h_user2bs1 == 0.2:
                SINR1_idx_p = random.choice([i for i in range(20)])
            else:
                SINR1_idx_p = random.randint(20,40)
            if h_user2uav == 0.6:
                SINR2_idx_p = random.choice([i for i in range(20)])
            else:
                SINR2_idx_p = random.randint(20, 40)

            # SINR3_idx_p = random.choice([i for i in range(len(States_level['sinr_uav2bs2']))])
            State_J_p = [SINR1_idx_p, SINR2_idx_p]
            State_U_p = [SINR1_idx_p, SINR2_idx_p]





            for hot_step in range(hot_max_time):


                Action_idx_U = Agent_U.choose_action_phc(State_U_p)
                tx_power = 2
                jam_power = P_jam_set[greedy_choose(tx_power, Action_idx_U,h_user2bs1,h_user2uav)]


                sinr_user2bs1 = P_user * h_user2bs1 / (jam_power * h_jam2bs1 + noise)
                sinr_user2uav = P_user * h_user2uav / (jam_power * h_jam2uav + noise)
                sinr_uav2bs2 = 6

                SINR1_idx = np.abs(sinr_user2bs1 - States_level['sinr_user2bs1']).argmin()
                SINR2_idx = np.abs(sinr_user2uav - States_level['sinr_user2uav']).argmin()
                # SINR3_idx = np.abs(sinr_uav2bs2 - States_level['sinr_uav2bs2']).argmin()

                State_U = [SINR1_idx, SINR2_idx]

                sinr_temp = max(sinr_user2bs1, min(sinr_user2uav, sinr_uav2bs2))
                ber = 0.5 * math.erfc(math.sqrt(sinr_temp / 2))
                if tx_power > 0:
                    delay = 100
                else:
                    delay = 0
                # print(ber)
                utility = Action_idx_U*sinr_temp - Action_idx_U*C_u +(1-Action_idx_U)*sinr_user2bs1  # - C_t * delay
                utility_J = -utility - C_j * jam_power

                # ==================== 存储经验 ====================#
                # Agent_U.put_hb(State_U_p, Action_idx_U, utility, State_U)
                # Agent_U.learn_hb()
                Agent_U.learn_renew_phc(State_U_p, Action_idx_U, State_U, utility)
                State_U_p = State_U
                v_obu = np.random.choice(vv.v_set, p=vv.p_v_set)
                h_user2bs1 = vv.gain_obtain_h_user2bs1(v_obu, h_user2bs1)
                h_user2uav = vv.gain_obtain_h_user2uav(v_obu, h_user2uav)
            # sum_qtable = sum_qtable + Agent_U.ql.q_table
            # sum_pitable = sum_pitable + Agent_U.ql.pi_table
            Q_table_hb = Agent_U.ql.q_table.copy()
            pi_table_hb = Agent_U.ql.pi_table.copy()
            print(Q_table_hb.max(),pi_table_hb.max())

        # ave_qtable = sum_qtable/(hot_max_episode)
        # ave_pitable = sum_pitable/(hot_max_episode)

        # Agent_U.reset()
        # pi_table_hb = Agent_U.save_pi()

        for episode in tqdm(range(max_episode), ascii=0, desc='Episode'):

            # Agent_U.ql.pi_table = pi_table_hb.copy()
            # Agent_U.ql.q_table = Q_table_hb.copy()

            # Agent_U.ql.pi_table = pi_table_hb
            Agent_U.load(Q_table_hb,pi_table_hb)

            Agent_U_Q.reset()
            Agent_U_phc.reset()
            # print(Agent_U.ql.pi_table.max() , '1')
            # print(Agent_U_Q.ql.pi_table.max() , '2')
            # print(Agent_U_phc.ql.pi_table.max() ,'3')


            """环境状态初始化"""
            # SINR1_idx_p = random.choice([i for i in range(len(States_level['sinr_user2bs1']))])
            # SINR2_idx_p = random.choice([i for i in range(len(States_level['sinr_user2uav']))])
            # SINR3_idx_p = random.choice([i for i in range(len(States_level['sinr_uav2bs2']))])
            tx_power_p = 2
            h_user2bs1 = np.random.choice(vv.h_user2bs1_set, p=vv.p_h_user2bs1_set)
            h_user2uav = np.random.choice(vv.h_user2uav_set, p=vv.p_h_user2uav_set)
            if h_user2bs1 == 0.2:
                SINR1_idx_p = random.choice([i for i in range(20)])
            else:
                SINR1_idx_p = random.randint(20, 40)
            if h_user2uav == 0.6:
                SINR2_idx_p = random.choice([i for i in range(20)])
            else:
                SINR2_idx_p = random.randint(20, 40)
            jam_power_p = P_jam_set[random.randint(0, (len(P_jam_set)) - 1)]

            """构造状态向量"""
            State_U_p = [SINR1_idx_p, SINR2_idx_p]
            State_U_p_Q = [SINR1_idx_p, SINR2_idx_p]
            State_U_p_phc = [SINR1_idx_p, SINR2_idx_p]
            for time_step in range(max_time):
                """根据观测到的状态执行动作"""
                Action_idx_U = Agent_U.choose_action_phc(State_U_p)
                Action_idx_U_Q = Agent_U_Q.choose_action(State_U_p_Q)
                Action_idx_U_phc = Agent_U_phc.choose_action_phc(State_U_p_phc)

                tx_power = 2
                tx_power_Q = 2
                tx_power_phc = 2

                jam_power = P_jam_set[greedy_choose(tx_power,Action_idx_U,h_user2bs1,h_user2uav)]
                jam_power_Q = P_jam_set[greedy_choose(tx_power_Q,Action_idx_U_Q,h_user2bs1,h_user2uav)]
                jam_power_phc = P_jam_set[greedy_choose(tx_power_phc,Action_idx_U_phc,h_user2bs1,h_user2uav)]
                """动作执行后，计算状态向量元素指标以更新状态"""
                sinr_user2bs1 = P_user * h_user2bs1 / (jam_power * h_jam2bs1 + noise)
                sinr_user2uav = P_user * h_user2uav / (jam_power * h_jam2uav + noise)
                sinr_uav2bs2 = 6

                sinr_user2bs1_Q = P_user * h_user2bs1 / (jam_power_Q * h_jam2bs1 + noise)
                sinr_user2uav_Q = P_user * h_user2uav / (jam_power_Q * h_jam2uav + noise)
                sinr_uav2bs2_Q = 6

                sinr_user2bs1_phc = P_user * h_user2bs1 / (jam_power_phc * h_jam2bs1 + noise)
                sinr_user2uav_phc = P_user * h_user2uav / (jam_power_phc * h_jam2uav + noise)
                sinr_uav2bs2_phc = 6

                SINR1_idx = np.abs(sinr_user2bs1 - States_level['sinr_user2bs1']).argmin()
                SINR2_idx = np.abs(sinr_user2uav - States_level['sinr_user2uav']).argmin()
                # SINR3_idx = np.abs(sinr_uav2bs2 - States_level['sinr_uav2bs2']).argmin()

                SINR1_idx_Q = np.abs(sinr_user2bs1_Q - States_level['sinr_user2bs1']).argmin()
                SINR2_idx_Q = np.abs(sinr_user2uav_Q - States_level['sinr_user2uav']).argmin()
                # SINR3_idx_Q = np.abs(sinr_uav2bs2_Q - States_level['sinr_uav2bs2']).argmin()

                SINR1_idx_phc = np.abs(sinr_user2bs1_phc - States_level['sinr_user2bs1']).argmin()
                SINR2_idx_phc = np.abs(sinr_user2uav_phc - States_level['sinr_user2uav']).argmin()
                # SINR3_idx_phc = np.abs(sinr_uav2bs2_phc - States_level['sinr_uav2bs2']).argmin()
                # SINR1_idx = greedy_choose(tx_power)
                # SINR2_idx = greedy_choose(tx_power)
                # SINR3_idx = Action_idx_U

                State_U = [SINR1_idx, SINR2_idx]
                State_U_Q = [SINR1_idx_Q, SINR2_idx_Q]
                State_U_phc = [SINR1_idx_phc, SINR2_idx_phc]
                """更新后的状态"""
                ber1 = math.erfc(math.sqrt(sinr_user2bs1 / 2))
                ber2 = math.erfc(math.sqrt(min(sinr_user2uav, sinr_uav2bs2) / 2))
                ber = 0.5 * min(ber1, ber2)
                sinr_temp = max(sinr_user2bs1, min(sinr_user2uav, sinr_uav2bs2))

                ber1 = math.erfc(math.sqrt(sinr_user2bs1_Q / 2))
                ber2 = math.erfc(math.sqrt(min(sinr_user2uav_Q, sinr_uav2bs2_Q) / 2))
                ber_Q = 0.5 * min(ber1, ber2)
                sinr_temp_Q = max(sinr_user2bs1_Q, min(sinr_user2uav_Q, sinr_uav2bs2_Q))

                ber1 = math.erfc(math.sqrt(sinr_user2bs1_phc / 2))
                ber2 = math.erfc(math.sqrt(min(sinr_user2uav_phc, sinr_uav2bs2_phc) / 2))
                ber_phc = 0.5 * min(ber1, ber2)
                sinr_temp_phc = max(sinr_user2bs1_phc, min(sinr_user2uav_phc, sinr_uav2bs2_phc))

                """计算增益(也是指标)"""
                utility = Action_idx_U*sinr_temp - Action_idx_U*C_u +(1-Action_idx_U)*sinr_user2bs1

                utility_Q = Action_idx_U_Q*sinr_temp_Q - Action_idx_U_Q*C_u +(1-Action_idx_U_Q)*sinr_user2bs1_Q

                utility_phc = Action_idx_U_phc*sinr_temp_phc - Action_idx_U_phc*C_u + (1-Action_idx_U_phc)*sinr_user2bs1_phc

                Agent_U.learn_renew_phc(State_U_p, Action_idx_U, State_U, utility)
                Agent_U_Q.learn_renew(State_U_p_Q, Action_idx_U_Q, State_U_Q ,utility_Q)
                Agent_U_phc.learn_renew_phc(State_U_p_phc, Action_idx_U_phc, State_U_phc, utility_phc)
                State_U_p = State_U
                State_U_p_Q = State_U_Q
                State_U_p_phc = State_U_phc
                v_obu = np.random.choice(vv.v_set, p=vv.p_v_set)
                h_user2bs1 = vv.gain_obtain_h_user2bs1(v_obu, h_user2bs1)
                h_user2uav = vv.gain_obtain_h_user2uav(v_obu, h_user2uav)
                # ================== 保存 ====================================#
                save_dict["Reward"][episode, time_step] = utility - 1
                save_dict["Sinr"][episode, time_step] = sinr_temp
                save_dict["BER"][episode, time_step] = ber + 0.001
                save_dict["Energy"][episode, time_step] = tx_power + 100
                save_dict["jam_power"][episode, time_step] = jam_power
                save_dict["Reward_Q"][episode, time_step] = utility_Q - 1
                save_dict["Sinr_Q"][episode, time_step] = sinr_temp_Q
                save_dict["BER_Q"][episode, time_step] = ber_Q + 0.001
                save_dict["Energy_Q"][episode, time_step] = tx_power_Q + 100
                save_dict["jam_power_Q"][episode, time_step] = jam_power
                save_dict["Reward_phc"][episode, time_step] = utility_phc - 1
                save_dict["Sinr_phc"][episode, time_step] = sinr_temp_phc
                save_dict["BER_phc"][episode, time_step] = ber_phc + 0.001
                save_dict["Energy_phc"][episode, time_step] = tx_power_phc + 100
                save_dict["jam_power_phc"][episode, time_step] = jam_power
        # ================================ 数据保存及画图 ================================#
        save_mat_path = os.path.join(mat_res_path, alg_name + "_results", time_str)
        os.makedirs(save_mat_path)
        mat_path = os.path.join(save_mat_path, "results.mat")
        sio.savemat(mat_path, save_dict)
        print(time_str)










    # ============================= learning start ===========================#



if __name__ == '__main__':
    main()
