from scipy.io import savemat
from tvb.simulator.lab import *
import time
import numpy as np
import multiprocessing as mp
import os
import argparse


def main(region_id):
    """ TVB Simulation to generate raw source space dynamics, unit in mV, and ms
    :param region_id: int; source region id, with parameters generating interictal spike activity
    """
    if not os.path.isdir('../source/raw_nmm/a{}/'.format(region_id)):
        os.mkdir('../source/raw_nmm/a{}/'.format(region_id))
    start_time = time.time()
    print('------ Generate data of region_id {} ----------'.format(region_id))
    conn = connectivity.Connectivity.from_file(
        source_file=os.getcwd() + '/../anatomy/connectivity_998.zip')  # connectivity provided by TVB
    conn.configure()

    # define A value
    num_region = conn.number_of_regions
    a_range = [3.5]
    # A的shape是[num_region,1]
    A = np.ones((num_region, len(a_range))) * 3.25  # the normal A value is 3.25
    A[region_id, :] = a_range  # 让一个区域作为激活源

    # define mean and std
    # 这个平均值和标准差 第一行是平均值 第二行是标准差 第一行是jansen-rit模型的一个参数
    # mean_and_std = np.array([[0.087, 0.08, 0.083], [1, 1.7, 1.5]]) 之前只生成了第一个
    mean_and_std = np.array([[0.087, 0.08, 0.083], [1, 1.7, 1.5]])
    for iter_a in range(A.shape[1]):  # A.shape[1]为1 母鸡有啥用
        use_A = A[:, iter_a]  # 其实use_A就是A
        for iter_m in range(mean_and_std.shape[1]):  # 遍历三遍 mean_and_std.shape[1]值为3

            # jrm模型 mu为平均发射速率 默认值是0.22; v0 PSP发射阈值-达到50%的发射速率
            # p_max 最大输入发射速率 默认值为0.32;   p_min 最小的发射速率 默认值为0.12
            jrm = models.JansenRit(A=use_A, mu=np.array(mean_and_std[0][iter_m]),
                                   v0=np.array([6.]), p_max=np.array([0.15]), p_min=np.array([0.03]))

            phi_n_scaling = (jrm.a * 3.25 * (jrm.p_max - jrm.p_min) * 0.5 * mean_and_std[1][iter_m]) ** 2 / 2.
            sigma = np.zeros(6)
            sigma[4] = phi_n_scaling

            # set the random seed for the random intergrator  --使用下面代码 确保在执行包含随机性的积分计算时，结果是可复现的
            randomStream = np.random.mtrand.RandomState(0)
            noise_class = noise.Additive(random_stream=randomStream, nsig=sigma)
            integ = integrators.HeunStochastic(dt=2 ** -1, noise=noise_class)

            sim = simulator.Simulator(
                model=jrm,
                connectivity=conn,
                coupling=coupling.SigmoidalJansenRit(a=np.array([1.0])),
                integrator=integ, #integrators.HeunStochastic(dt=2 ** -1, noise=noise.Additive(nsig=sigma)) 我改成integ
                monitors=(monitors.Raw(),)
            ).configure()

            # run 200s of simulation, cut it into 20 pieces, 10s each. (Avoid saving large files)
            # 为了保准能存下数据 总共运行12.5s 分成20片 每片2.5s   因为区域的量也减少了一倍 数据量可以减少很多
            for iii in range(5):  # 原本是20 改为5
                siml = 2000  # 每片时间  原本是 1e4  改为2000
                out = sim.run(simulation_length=siml)
                (t, data), = out
                data = (data[:, 1, :, :] - data[:, 2, :, :]).squeeze().astype(np.float32)  # 这行是干什么不知道 tvb文档也没有对这个返回值进行说明
                print("iii:", iii)  # 这行只是让我看程序在没在运行
                # in the fsaverage5 mapping, there is no vertices corresponding to region 7,325,921, 949, so change label 994-998 to those id
                data[:, 7] = data[:, 994]
                data[:, 325] = data[:, 997]
                data[:, 921] = data[:, 996]
                data[:, 949] = data[:, 995]
                data = data[:, :994]
                # 第三个{}可以不用看 对理解没有帮助
                savemat('../source/raw_nmm/a{}/mean_iter_{}_a_iter_{}_{}.mat'.format(region_id, iter_m, region_id, iii),
                        {'time': t, 'data': data, 'A': use_A})
    print('Time for', region_id, time.time() - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TVB Simulation')
    parser.add_argument('--a_start', type=int, default=0, metavar='REGION_ID', help='start region id')
    parser.add_argument('--a_end', type=int, default=1, metavar='REGION_ID', help='end region id')
    args = parser.parse_args()
    os.environ["MKL_NUM_THREADS"] = "1"
    start_time = time.time()
    # RUN THE CODE IN PARALLEL
    # 先跑0-5 再跑5-10
    processes = [mp.Process(target=main, args=(x,)) for x in range(args.a_start, args.a_end)]
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    # NO PARALLEL
    # for x in range(args.a_start, args.a_end):  # 测试不同id的脑源区域为激活时候  原本是10 现在变为5 减少数据量
    #     main(x)
    #     print("x:", x)
    print('Total_time', time.time() - start_time)
