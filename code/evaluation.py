from experiment import *
import dataloader
import os

def eval(mr, dataset, k, lmd=10, iter=500, version="", p=2):
    print(dataset)
    mat, missing_mat_list = dataloader.loadDataset(dataset, mr)

    methods = ['SMFL_sgd', 'SMFL_multi']
    results = []
    times = []
    for i in range(len(missing_mat_list)):
        results.append([])
        times.append([])

    for idx, mat_missing in enumerate(missing_mat_list):
        np.random.seed(idx+100000)
        print("----iteration:",idx,"----")
    
        SMFL_multi, _, _, t6 = SMFLTest(mat, mat_missing, lmd, method=2, k=k, iter=iter, cluster=True)
        print('SMFL_multi: {}'.format(SMFL_multi))
        results[idx].append(SMFL_multi)
        times[idx].append(t6)


    results = np.array(results)
    print("results")
    print(results)
    result_avg = np.mean(results, axis=0)
    print("results average")
    print(result_avg)
    return result_avg[0]


if __name__ == "__main__":    
    np.set_printoptions(suppress=True)
    version = "test"
    dataset = 'economic'
    paras = parameters[dataset]
    mr = 10

    eval(mr, dataset, k=paras['k'], lmd=paras['lmd'], iter=paras['iter'], version=version, p=paras['p'])
    print(f"version:{version} finished")