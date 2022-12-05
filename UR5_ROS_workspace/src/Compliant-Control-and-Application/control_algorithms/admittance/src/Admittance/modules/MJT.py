#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def get_MJT(tf, xf, t0=0, x0=0, N=100):
    """
    𝜏 is the normalized time and equal to 𝑡/𝑡
    """
    # t = np.linspace(t0,tf,N)
    dt = (tf - t0) / N
    tau = np.linspace(0, 1, N)
    xt = x0 + (xf - x0) * (6 * np.power(tau, 5) - 15 * np.power(tau, 4) + 10 * np.power(tau, 3))

    vt = np.gradient(xt) / dt
    at = np.gradient(vt) / dt
    Jt = np.gradient(at) / dt
    return xt, vt, at, Jt


if __name__ == "__main__":
    xt, vt, at, Jt = get_MJT(3, 5)
    plt.plot(xt, label='pos')
    plt.plot(vt, label='vel')
    plt.plot(at, label='acc')
    plt.plot(Jt, label='jerk')
    plt.show()
