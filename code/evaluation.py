from experiment import *
import dataloader
import os

def eval(mr, dataset, k, lmd=10, iter=500, version="", p=2):
    print(dataset)
    mat, missing_mat_list = dataloader.loadDataset(dataset, mr)

    methods = ['SMC-sgd', 'SMC-multi', 'SMCL_sgd', 'SMCL_multi']
    results = []
    times = []
    for i in range(len(missing_mat_list)):
        results.append([])
        times.append([])

    for idx, mat_missing in enumerate(missing_mat_list):
        np.random.seed(idx*100)
        print("----iteration:",idx,"----")
        SMC_sgd, _ , _, t1= SMCTest(mat, mat_missing, lmd, method=1, k=k, iter=iter, p=p)
        print('SMC-sgd: {}'.format(SMC_sgd))
        results[idx].append(SMC_sgd)
        times[idx].append(t1)

        SMC_multi, _ , _, t2= SMCTest(mat, mat_missing, lmd, method=2, k=k, iter=iter, p=p)
        print('SMC-multi: {}'.format(SMC_multi))
        results[idx].append(SMC_multi)
        times[idx].append(t2)


        SMCL_sgd, _, _, t5 = SMCLTest(mat, mat_missing, lmd, method=1, k=k, iter=iter, cluster=True)
        print('SMCL_sgd: {}'.format(SMCL_sgd))
        results[idx].append(SMCL_sgd)
        times[idx].append(t5)

        SMCL_multi, _, _, t6 = SMCLTest(mat, mat_missing, lmd, method=2, k=k, iter=iter, cluster=True)
        print('SMCL_multi: {}'.format(SMCL_multi))
        results[idx].append(SMCL_multi)
        times[idx].append(t6)



    results = np.array(results)
    print("results")
    print(results)
    result_avg = np.mean(results, axis=0)
    print("results average")
    print(result_avg)
    


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parameters = {
        'k': 6,
        'lmd': 10,
        'iter': 500,
        'p': 2,
    }
    version = "test"
    dataset = 'california'
    mr = 10
    eval(mr, dataset, k=parameters['k'], lmd=parameters['lmd'], iter=parameters['iter'], version=version, p=parameters['p'])
    print(f"version:{version} finished")