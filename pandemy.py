#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import yaml
import random
from scipy.integrate import odeint


# master equations
def sird(X, t):
    S = X[0]
    J = X[1]
    f0 = - alpha * S * J
    f1 = alpha * S * J - beta * J - delta * J
    f2 = beta * J
    f3 = delta * J
    return [f0, f1, f2, f3]


def event(bound1, bound2, bound3):
    rn = random.random()
    if rn <= bound1:
        return 0
    elif bound1 < rn <= bound2:
        return 1
    elif bound2 < rn <= bound3:
        return 2
    else:
        return 3


def pandemy(init_data, Period, nhist):
    # init_data = [params['popul']['healthy'],
    #             params['popul']['injured'],
    #             params['popul']['dead'],
    #             params['popul']['recovered']]
    history = np.zeros([4*nhist, Period])
    for hist in range(1, nhist+1):
        print("### history #", hist)
        healthy = init_data[0]
        injured = init_data[1]
        dead = init_data[2]
        recover = init_data[3]
        history[(hist-1)*4: 4 + (hist-1)*4, 0] = [healthy, injured, dead, recover]
        for day in range(1, Period):
            if injured == 0:
                return history
            else:
                new = 0
                for item in range(1, injured + 1):
                    ev = event(delta, delta + beta, delta + beta + alpha * healthy)
                    if ev == 0:
                        new = new - 1
                        dead = dead + 1
                    elif ev == 1:
                        new = new - 1
                        recover = recover + 1
                    elif ev == 2:
                        new = new + 1
                        healthy = healthy - 1
                    else:
                        pass
                injured = injured + new

                history[(hist-1)*4: 4 + (hist-1)*4, day] = [healthy, injured, dead, recover]
    ndays = np.linspace(0, Period-1, Period)
    history = np.vstack([ndays, history])
    return history

# plotter
def plotter(S, J, R, D, Time, data):

    ME = np.mean(data[:,2:-1:4], 1)
    STD = np.std(data[:,2:-1:4], 1)
    DAYS = data[:,0]
    rare = 1

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor='w', edgecolor='k')
    # ax.plot(Time, S, '-', color='blue', linewidth=2, label='Healthy')
    ax.plot(Time, J, '-', color='red', linewidth=2, label='Injured')
    # ax.plot(Time, R, '-', color='green', linewidth=2, label='Recovered')
    # ax.plot(Time, D, '-', color='black', linewidth=2, label='Dead')
    ax.set_xlabel('Time, in days', fontsize=14)
    ax.set_ylabel('Injured', fontsize=14)
    ax.set_title('Test COVID-19 model', fontsize=14)
    ax.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # ax.plot(DAYS, data[1:, 1:len(data[1, :-1]):rare], '.', color='tab:blue', markersize=1)
    ax.fill_between(DAYS, ME - 3 * STD, ME + 3 * STD, alpha=0.2)
    ax.plot(DAYS, ME, '--', color='black', linewidth=3)
    ax.set_title('Mean and STD', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.yscale('symlog')
    # plt.xlim([0, 100])
    # plt.ylim([10, 10000])
    ax.grid(True)

    # plt.show()
    name = 'ode_solution.jpg'
    fig.savefig(name)


def main(params):
    init_data = [params['popul']['healthy'],
                 params['popul']['injured'],
                 params['popul']['dead'],
                 params['popul']['recovered']]
    Time = np.linspace(0, params['times']['total'], params['times']['n_steps'])
    Period = params['times']['total']
    nhist = params['times']['nhist']
    solution = odeint(sird, init_data, Time)
    S = solution[:, 0]
    J = solution[:, 1]
    R = solution[:, 2]
    D = solution[:, 3]

    history = pandemy(init_data, Period, nhist)
    np.savetxt('pandemy.txt', history.transpose(), fmt='%d', delimiter=' ', newline='\n',
                  header='', footer='', comments='# ', encoding=None)  #
    data = np.genfromtxt('pandemy.txt')
    plotter(S, J, R, D, Time, data)
    print('done!')


if __name__ == '__main__':
    # load the model parameters
    params = yaml.load(open('params.yaml', 'r'))
    # three global constants
    alpha = params['prob']['injure'] / params['popul']['total'] / params['times']['disease']  # injuring rate
    beta = params['prob']['recov'] / params['times']['disease']  # recovering rate
    delta = params['prob']['death'] / params['times']['disease']  # death rate
    population = params['popul']['total']

    main(params)
