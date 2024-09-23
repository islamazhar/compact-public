"""Main non-linear function generator.
"""
import sys; sys.path.append("..")

import utils.funcs as nf
from utils.fitter import MPCApproximation, PolyK



import numpy as np



DEBUG = False
save_all = False
profile_time = False
MAX_BREAKS = 1000


def normal_dist(x , mean = 0  , sd = 1):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


# print("Inside sample_poly_fit")

def _merge_breaks(a,
                    b,
                    ms,
                    func,
                    k,
                    tol,
                    f,
                    n,
                    zero_mask=None,
                    derivative_flag=True,
                    error_metric=("sampled", "relative_error"),
                    plain_method="Cheby"):
    """Greedily merging all the mergeable successive pieces in P_{(k, m)}.
    """

    if derivative_flag:
        func_exec = nf.convert_function(func)
    else:
        func_exec = func
    # test merge or not.
    x, y = nf.adaptive_sampling(a, b, func_exec, f, ms, zero_mask)
    model = PolyK(k)
    model.fit(a, b, func, f, n, ms, derivative_flag, method=plain_method, zero_mask=zero_mask)
    
    # error_metric = ("sampled", "relative_error")
    
    if error_metric[0] == "sampled":
            y_pred = model.predict(x, n, f)
            each_error = nf.sampled_error(y, y_pred, f, zero_mask,
                                          error_metric[1])
            max_error = np.max(each_error)
            if DEBUG:
                argm = np.argmax(each_error)
                print("samples: ", len(y_pred))
                print(">>>>>> max_error: ", max_error)
                print("corresponding max error sample x = %.8g, y_true = %.8g, y_pred = %.8g"%(x[argm], y[argm], y_pred[argm]))

    elif error_metric[0] == "analytic":
            interval_coeff = model.coeff.squeeze() * model.scale.squeeze()
            max_error = nf.analytic_error(func,
                                          interval_coeff,
                                          x[0],
                                          x[-1],
                                          f,
                                          zero_mask,
                                          method=error_metric[1])
            if DEBUG:
                print(">> analytic max_error: ", max_error)

    if max_error < tol:
            return True, model.coeff, model.scale
    else:
            return False, None, None
            



def sampled_poly_fit(a,
                     b,
                     k,
                     tol,
                     func,
                     f,
                     n,
                     ms=1e+3,
                     df=None,
                     zero_mask=None,
                     derivative_flag=True,
                     error_metric=("sampled", "relative_error"), # sample is for => sampled_error analytic => relative_error
                     plain_method="OLS"
                     ):
    """Using the SampledPolyFit algorithm to find the valid candidate P_{(k, m)}.
    
    a: start of the input domain.
    b: end of the input domain.
    k: the target order.
    tol: user-defined tolerance.
    func: the target non-linear functions.
    f: the precision factor of target MPC fixed numbers.
    ms: default sampling limits.
    df: the pre-computed derivative function.
    zero_mask: for all value less than zero_mask, will be transformed to zero.
    derivative_flag: flag indicating whether the target func can calculate derivative.
    error_metric: the defined error analysis metric, now we only support sampled relative error and sampled absolute error.
    """


    if DEBUG:
        print("In sampled_poly_fit: start - %.8f, end - %.8f" % (a, b))

    decimal = nf.calculate_decimal(f)

    # FAC algorithm
    result = []
    coeff_list = []
    scale_list = []
    stack = [(a, b)]
    poss_breaks = 1


    # Function conversion
    if derivative_flag and df is None: # df is always None and derivative_flag is always True
        df = nf.find_derivative(func)
        func_exec = nf.convert_function(func)
    else:
        func_exec = func

    while (stack):
        if poss_breaks >= MAX_BREAKS:
            # Too long pieces.
            raise Exception("Breaks exceed the limit.")

        start, end = stack.pop()
        if DEBUG:
            print("Inside while stack Start - %.18f | End - %.18f" % (start, end))

        if (np.double(start).round(decimal) == np.double(end).round(decimal)):
            continue
        # print("Adaptive sampling")
        # Fit polynomial
        x, y = nf.adaptive_sampling(start, end, func_exec, f, ms, zero_mask=zero_mask) # this is not needed we do it again inside fit
        
        if(len(x) == 1):
            print(">>>> y: ", y)
            model.coeff = np.concatenate([y, [0]*(k)])[:, np.newaxis] # [y, 0, 0, 0,0 ] => coeff scale = [1, 1, 1, 1]
            model.scale = np.ones(model.coeff.shape)
        else:
            model = PolyK(k)
            model.fit(start, end, func_exec, f, n, ms=ms, zero_mask=zero_mask)
        
        split_flag = True

        y_pred = model.predict(x, n, f)
        each_error = nf.sampled_error(y, y_pred, f, zero_mask, error_metric[1])
        
        # weighted_error = each_error * normal_dist(x)
        # max_error = weighted_error.sum()/(b-a)
        max_error = np.max(each_error)
        split_flag = (max_error >= tol)
        
        # print(f"max error => {max_error}\t tol = {tol}\tsplit_flag = {split_flag}")
        
        # print(f"split flag =>{split_flag}")
        
        if split_flag and (len(x) > (k + 1)):
            poss_breaks += 1
            stack.append(((start + end) / 2, end))
            stack.append((start, (start + end) / 2))
        else:
            coeff_list.append(model.coeff)
            scale_list.append(model.scale)
            result.append([start, end])
            
            
    # Greedy merge. // without it NFGen was not working well.
    rp = 0
    while (rp < len(result) - 1):
        new_breaks = [result[rp][0], result[rp + 1][1]]
        flag, new_coeff, new_scale = _merge_breaks(new_breaks[0], new_breaks[1], ms, func, k, tol, f, n,
                                                   zero_mask, derivative_flag,
                                                   error_metric, plain_method=plain_method)
        if flag:
            for _ in range(2):
                coeff_list.pop(rp)
                scale_list.pop(rp)
                result.pop(rp)

            coeff_list.insert(rp, new_coeff)
            scale_list.insert(rp, new_scale)
            result.insert(rp, new_breaks)
        else:
            rp += 1
            

    return coeff_list, result, scale_list


"""returns MPCApproximation class
"""

def GenerateMPCApproximation(config_dict):
    """Main function to generate the non-linear functions' config.

    Args:
        config_dict (dict): Dict stores the config information for target non-linear functions.
    """

    # Necessary keys check
    if "function" not in config_dict.keys():
        print("Please indicate the target function F in `function`.")
    if "range" not in config_dict.keys():
        print("Please indicate the input domain [a, b] in `range`.")
    if "tol" not in config_dict.keys():
        print("Please indicate the tolerance \epsilon in `tol`.")
    if "n" not in config_dict.keys():
        print("Please indicate the number length in `n`.")
    if "f" not in config_dict.keys():
        print("Please indicate the resolution in `f`.")
        
    method = "Cheby"


    func = config_dict['function']
    a, b = config_dict['range']
    tol = config_dict['tol']
    n = config_dict['n']
    f = config_dict['f']
    decimal = nf.calculate_decimal(f) # f = 48 default value

    if "max_breaks" not in config_dict.keys():
        MAX_BREAKS = 1000
    else:
        MAX_BREAKS = config_dict["max_breaks"]
        
    

    # Other hints, and default values.
    if "k_max" not in config_dict.keys():
        k_range = range(3, 10)
    else:
        # k_range = range(3, config_dict['k_max'])
        k_range = range(config_dict['k_max'], config_dict['k_max'] + 1)

    if "zero_mask" not in config_dict.keys():
        zero_mask = 1e-8
    else:
        zero_mask = config_dict['zero_mask']

    if "config_file" not in config_dict.keys():
        config_file = './config_file.py'
    else:
        config_file = config_dict['config_file']

    if "ms" in config_dict.keys():
        ms = config_dict['ms']
    else:
        ms = 1000


    # if "default_values" in config_dict.keys(
    # ):  # default values exceeds the input domain.
    #     default_flag = True
    #     left_default = config_dict["default_values"][0]
    #     right_default = config_dict["default_values"][1]
    #     less_break = a - 999
    #     larger_break = b + 999
    # else:
    #     default_flag = False


    candidate_list = []

    error_metric = ("sampled", "relative_error")
    derivative_flag = True 
    
    # Generate the candidate P_{(k, m)}.
    for k in k_range:
        # print(f"k_range => {k_range}")
        try:
            coeff_list, breaks, scale_list = sampled_poly_fit(
                    a,
                    b,
                    k,
                    tol,
                    func,
                    f,
                    n,
                    ms=ms,
                    df = None,
                    zero_mask=zero_mask,
                    error_metric=error_metric,
                    derivative_flag=derivative_flag
                    )
        except Exception as e:
                print(e.args)
                print("failed current k = %d" % k)
                continue

        breaks, coeffA, scale_list = nf.result_orgnize (breaks, coeff_list,
                                                       scale_list)
        breaks = np.array (breaks)
        scale_list = np.array (scale_list)
        
        

        if len(breaks) == 2:
            coeffA = coeffA[np.newaxis, :]
            scale_list = scale_list[np.newaxis, :]
        
        
        candidate_list.append(MPCApproximation(len(breaks), coeffA, decimal, scale_list, breaks, n, f))
        # break
        
        
    return candidate_list[-1] # TODO: return only one


