import compress_methods
import numpy as np
###import matplotlib.pyplot as plt
from typing import Any, List, Dict, Union, Optional
import math
from scipy.spatial import Delaunay
import pickle
import os
from scipy.spatial.distance import directed_hausdorff

def vol_diff():
    # Compute and plot the results of vol differences
    whole_vol_res = dict()
    for i in range(1, 51):
        infile = open('raw_tri_vol/' + str(i) + '.pkl', 'rb')
        raw_tri_vol = pickle.load(infile)
        infile.close()

        infile = open('c_tri_vol/' + str(i) + '.pkl', 'rb')
        c_tri_vol = pickle.load(infile)
        infile.close()

        res_each_cluster = dict()

        for each_tri in c_tri_vol:
            # [v1, v2, ..., vn]
            raw_vol = raw_tri_vol[each_tri]

            for c_tec, c_vol in c_tri_vol[each_tri].items():
                min_diff, max_diff, mean_diff = \
                    min([abs(val - c_vol[idx]) for idx, val in enumerate(raw_vol)]), \
                    max([abs(val - c_vol[idx]) for idx, val in enumerate(raw_vol)]), \
                    sum([abs(val - c_vol[idx]) for idx, val in enumerate(raw_vol)]) / len(raw_vol)

                if c_tec not in res_each_cluster:
                    res_each_cluster[c_tec] = {'min': [min_diff], 'max': [max_diff], 'mean': [mean_diff]}
                else:
                    res_each_cluster[c_tec]['min'].append(min_diff)
                    res_each_cluster[c_tec]['max'].append(max_diff)
                    res_each_cluster[c_tec]['mean'].append(mean_diff)

        f = open('vol_res.txt', 'a')
        f.write("Cluster " + str(i))
        f.write('\n')

        for each_tec, mmm in res_each_cluster.items():
            if each_tec not in whole_vol_res:
                whole_vol_res[each_tec] = {'min': [], 'max': [], 'mean': []}
            for key_meaure, vals in mmm.items():
                if key_meaure == 'min':
                    write_list = [each_tec[0], each_tec[1], key_meaure, min(vals)]
                    whole_vol_res[each_tec]['min'].append(min(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')
                elif key_meaure == 'max':
                    write_list = [each_tec[0], each_tec[1], key_meaure, max(vals)]
                    whole_vol_res[each_tec]['max'].append(max(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')
                elif key_meaure == 'mean':
                    write_list = [each_tec[0], each_tec[1], key_meaure, sum(vals) / len(vals)]
                    whole_vol_res[each_tec]['mean'].append(sum(vals) / len(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')

    f = open('whole_vol_res.txt', 'a')
    for each_tec, mmm in whole_vol_res.items():
        for key_measure, vals in mmm.items():
            write_list = [each_tec[0], each_tec[1], key_measure, sum(vals) / len(vals)]
            f.write(", ".join([str(x) for x in write_list]))
            f.write('\n')

    f.write('\n')


def angular_diff():
    whole_vol_res = dict()
    infile = open('compress_data.pkl', 'rb')
    compress_data = pickle.load(infile)
    infile.close()

    for i in range(1, 51):
        infile = open('raw_tri_vol/' + str(i) + '.pkl', 'rb')
        raw_tri_vol = pickle.load(infile)
        infile.close()

        f = open('cluster_data/' + str(i) + '.txt', 'r')
        loc_ts = dict()
        for ts in f:
            ts = ts.split(',')
            lon, lat = lonlat_to_xyz(float(ts[0]), float(ts[1]))
            time_s = [float(each_one) for each_one in ts[2:]]
            loc_ts[(lon, lat)] = time_s

        res_each_cluster = dict()

        for each_tri in raw_tri_vol:
            p1, p2, p3 = each_tri[0], each_tri[1], each_tri[2]
            raw1, raw2, raw3 = loc_ts[p1], loc_ts[p2], loc_ts[p3]

            for c_tech in compress_data[p1]:
                c1, c2, c3 = compress_data[p1][c_tech], compress_data[p2][c_tech], compress_data[p3][c_tech]
                if c_tech[0] == 'dp' or c_tech[0] == 'vw' or c_tech[0] == 'opt':
                    c_ratio = float(sum([1 if val is not None else 0 for val in c1]) +
                                    sum([1 if val is not None else 0 for val in c2]) +
                                    sum([1 if val is not None else 0 for val in c3])) / (3 * len(c1))
                    c1, c2, c3 = interpolate(c1), interpolate(c2), interpolate(c3)
                else:
                    c_ratio = c_tech[1]

                if (c_tech[0], c_ratio) not in res_each_cluster:
                    res_each_cluster[(c_tech[0], c_ratio)] = {'max': [], 'mean': []}

                h_d = []
                for idx in range(len(raw1)):
                    raw_p1, raw_p2, raw_p3 = np.array([p1[0], p1[1], raw1[idx]]), np.array([p2[0], p2[1], raw2[idx]]), \
                                             np.array([p3[0], p3[1], raw3[idx]])

                    raw_v1, raw_v2 = raw_p3 - raw_p1, raw_p2 - raw_p1

                    raw_normal_1, raw_normal_2, raw_normal_3 = np.cross(raw_v1, raw_v2)
                    raw_len = math.sqrt(raw_normal_1*raw_normal_1 + raw_normal_2 * raw_normal_2 +
                                        raw_normal_3 * raw_normal_3)

                    c_p1, c_p2, c_p3 = np.array([p1[0], p1[1], c1[idx]]), np.array([p2[0], p2[1], c2[idx]]),\
                                       np.array([p3[0], p3[1], c3[idx]])

                    c_v1, c_v2 = c_p3 - c_p1, c_p2 - c_p1

                    c_normal_1, c_normal_2, c_normal_3 = np.cross(c_v1, c_v2)
                    c_len = math.sqrt(c_normal_1 * c_normal_1 + c_normal_2 * c_normal_2 +
                                      c_normal_3 * c_normal_3)

                    cs = max(-1.0, min(1.0, (raw_normal_1 * c_normal_1 + raw_normal_2 * c_normal_2 + raw_normal_3 *
                                             c_normal_3)/(raw_len*c_len)))

                    ##print(cs)

                    h_d.append( math.acos(cs)/math.pi )


                # res_each_cluster[(c_tech[0], c_ratio)]['min'].append(min(h_d))
                res_each_cluster[(c_tech[0], c_ratio)]['max'].append(max(h_d))
                res_each_cluster[(c_tech[0], c_ratio)]['mean'].append(sum(h_d) / len(h_d))

        f = open('angd_res.txt', 'a')
        f.write("Cluster " + str(i))
        f.write('\n')

        for each_tec, mmm in res_each_cluster.items():
            if each_tec not in whole_vol_res:
                whole_vol_res[each_tec] = {'max': [], 'mean': []}
            for key_meaure, vals in mmm.items():
                '''
                if key_meaure == 'min':
                    write_list = [each_tec[0], each_tec[1], key_meaure, min(vals)]
                    whole_vol_res[each_tec]['min'].append(min(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')
                '''
                if key_meaure == 'max':
                    write_list = [each_tec[0], each_tec[1], key_meaure, max(vals)]
                    whole_vol_res[each_tec]['max'].append(max(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')
                elif key_meaure == 'mean':
                    write_list = [each_tec[0], each_tec[1], key_meaure, sum(vals) / len(vals)]
                    whole_vol_res[each_tec]['mean'].append(sum(vals) / len(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')

        print('Done with' + str(i) + '/50')

    f = open('whole_angd_res.txt', 'a')
    for each_tec, mmm in whole_vol_res.items():
        for key_measure, vals in mmm.items():
            write_list = [each_tec[0], each_tec[1], key_measure, sum(vals) / len(vals)]
            f.write(", ".join([str(x) for x in write_list]))
            f.write('\n')

    f.write('\n')


def h_diff():
    whole_vol_res = dict()
    infile = open('compress_data.pkl', 'rb')
    compress_data = pickle.load(infile)
    infile.close()

    for i in range(1, 51):
        infile = open('raw_tri_vol/' + str(i) + '.pkl', 'rb')
        raw_tri_vol = pickle.load(infile)
        infile.close()

        f = open('cluster_data/' + str(i) + '.txt', 'r')
        loc_ts = dict()
        for ts in f:
            ts = ts.split(',')
            lon, lat = lonlat_to_xyz(float(ts[0]), float(ts[1]))
            time_s = [float(each_one) for each_one in ts[2:]]
            loc_ts[(lon, lat)] = time_s

        res_each_cluster = dict()

        for each_tri in raw_tri_vol:
            p1, p2, p3 = each_tri[0], each_tri[1], each_tri[2]
            raw1, raw2, raw3 = loc_ts[p1], loc_ts[p2], loc_ts[p3]

            for c_tech in compress_data[p1]:
                c1, c2, c3 = compress_data[p1][c_tech], compress_data[p2][c_tech], compress_data[p3][c_tech]
                if c_tech[0] == 'dp' or c_tech[0] == 'vw' or c_tech[0] == 'opt':
                    c_ratio = float(sum([1 if val is not None else 0 for val in c1]) +
                                    sum([1 if val is not None else 0 for val in c2]) +
                                    sum([1 if val is not None else 0 for val in c3])) / (3 * len(c1))
                    c1, c2, c3 = interpolate(c1), interpolate(c2), interpolate(c3)
                else:
                    c_ratio = c_tech[1]

                if (c_tech[0], c_ratio) not in res_each_cluster:
                    res_each_cluster[(c_tech[0], c_ratio)] = {'max':[], 'mean':[]}

                h_d = []
                for idx, val in enumerate(raw1):
                    raw_v = [(p1[0], p1[1], val), (p2[0], p2[1], raw2[idx]), (p3[0], p3[1], raw3[idx])]
                    c_v = [(p1[0], p1[1], c1[idx]), (p2[0], p2[1], c2[idx]), (p3[0], p3[1], c3[idx])]
                    h_d.append(max(directed_hausdorff(raw_v, c_v)[0], directed_hausdorff(c_v, raw_v)[0]))

                #res_each_cluster[(c_tech[0], c_ratio)]['min'].append(min(h_d))
                res_each_cluster[(c_tech[0], c_ratio)]['max'].append(max(h_d))
                res_each_cluster[(c_tech[0], c_ratio)]['mean'].append(sum(h_d)/len(h_d))

        f = open('hd_res.txt', 'a')
        f.write("Cluster " + str(i))
        f.write('\n')

        for each_tec, mmm in res_each_cluster.items():
            if each_tec not in whole_vol_res:
                whole_vol_res[each_tec] = {'max': [], 'mean': []}
            for key_meaure, vals in mmm.items():
                '''
                if key_meaure == 'min':
                    write_list = [each_tec[0], each_tec[1], key_meaure, min(vals)]
                    whole_vol_res[each_tec]['min'].append(min(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')
                '''
                if key_meaure == 'max':
                    write_list = [each_tec[0], each_tec[1], key_meaure, max(vals)]
                    whole_vol_res[each_tec]['max'].append(max(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')
                elif key_meaure == 'mean':
                    write_list = [each_tec[0], each_tec[1], key_meaure, sum(vals) / len(vals)]
                    whole_vol_res[each_tec]['mean'].append(sum(vals) / len(vals))
                    f.write(", ".join([str(x) for x in write_list]))
                    f.write('\n')

        print('Done with' + str(i) + '/50')

    f = open('whole_hd_res.txt', 'a')
    for each_tec, mmm in whole_vol_res.items():
        for key_measure, vals in mmm.items():
            write_list = [each_tec[0], each_tec[1], key_measure, sum(vals) / len(vals)]
            f.write(", ".join([str(x) for x in write_list]))
            f.write('\n')

    f.write('\n')


def interpolate(points: List[float]) -> List[float]:
    idx = 0

    while idx < len(points):
        if points[idx] is None:
            start_p = idx - 1
            while idx < len(points):
                if points[idx] is not None:
                    break
                idx += 1

            slope = (points[idx] - points[start_p]) / (idx - start_p)

            points[start_p + 1:idx] = [points[start_p] + i * slope for i in range(1, idx - start_p)]
        else:
            idx += 1

    return points


def lonlat_to_xyz(lon: float, lat: float):
    #Convert angluar to cartesian coordiantes
    r = 6371  # https://en.wikipedia.org/wiki/Earth_radius
    theta = math.pi/2 - math.radians(lat)
    phi = math.radians(lon)
    return r * math.sin(theta) * math.sin(phi), r * math.sin(theta) * math.cos(phi)


def cal_vol(p1: List[float], p2: List[float], p3: List[float]) -> float:
    '''
    # p1 with smallest height -> p3 with highest height
    p1, p2, p3 = sorted([p1, p2, p3], key=lambda x: x[2])
    # Based on Shoelace formula, the base area
    base_1_area = 0.5 * abs((p1[0]-p3[0])*(p2[1]-p1[1]) - (p1[0]-p2[0])*(p3[1]-p1[1]))
    part_1_vol = base_1_area * p1[2]

    # Pyramid
    base_2_area = math.sqrt((p2[1]-p3[1])**2 + (p2[0]-p3[0])**2) * p3[2] - \
                  0.5 * math.sqrt((p2[1]-p3[1])**2 + (p2[0]-p3[0])**2) * (p3[2]-p2[2])
    h_2 = abs((p3[1]-p2[1])*p1[0] - (p3[0]-p2[0])*p1[1] + p3[0]*p2[1] - p3[1]*p2[0])/\
          math.sqrt((p2[1]-p3[1])**2 + (p2[0]-p3[0])**2)
    part_2_vol = base_2_area * h_2

    return part_1_vol + part_2_vol
    '''
    ### 0.5 [x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)]
    base_area = 0.5 * abs(p1[0] * (p2[1]-p3[1]) + p2[0] * (p3[1]-p1[1]) + p3[0] * (p1[1]-p2[1]))
    h = 1/3 * (p1[2]+p2[2]+p3[2])
    return base_area * h


def main():
    # Compress Ratio for PAA and DFT
    ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

    # Error tolerance for DP, VW and OPT
    errors = [15, 25, 35, 50, 65, 80]

    # {(lon, lat): {(compress_tech, params): [......]}}
    if os.path.exists('compress_data.pkl'):
        infile = open('compress_data.pkl', 'rb')
        compress_data = pickle.load(infile)
        infile.close()
    else:
        compress_data = dict()

    for idx in range(1, 51):
        # Generate Triangles and Raw data volume
        # {(pi, pj, pk): [v1, v2, ..., vn]}
        if not os.path.exists('raw_tri_vol/' + str(idx) + '.pkl'):
            raw_vol = dict()
            f = open('cluster_data/' + str(idx) + '.txt', 'r')
            loc_ts = dict()
            for ts in f:
                ts = ts.split(',')
                lon, lat = lonlat_to_xyz(float(ts[0]), float(ts[1]))
                time_s = [float(each_one) for each_one in ts[2:]]
                loc_ts[(lon, lat)] = time_s

            points = np.array(list(loc_ts.keys()))
            tris = Delaunay(points)

            for each_tri in points[tris.simplices]:
                p1, p2, p3 = tuple(each_tri[0]), tuple(each_tri[1]), tuple(each_tri[2])
                ts1, ts2, ts3 = loc_ts[p1], loc_ts[p2], loc_ts[p3]
                for tstamp in range(len(ts1)):
                    raw_vol[(p1, p2, p3)] = raw_vol.get((p1, p2, p3), []) + \
                                            [cal_vol(list(p1) + [ts1[tstamp]],
                                                     list(p2) + [ts2[tstamp]],
                                                     list(p3) + [ts3[tstamp]])]

            with open('raw_tri_vol/' + str(idx) + '.pkl', 'wb') as handle:
                pickle.dump(raw_vol, handle, protocol=pickle.HIGHEST_PROTOCOL)

            f.close()

        # Generate compressed data
        # {pi: {(c_method, ratio/error): [......]}}
        f = open('cluster_data/' + str(idx) + '.txt', 'r')
        for ts in f:
            ts = ts.split(',')
            lon, lat = lonlat_to_xyz(float(ts[0]), float(ts[1]))

            if (lon, lat) not in compress_data:
                print('Start dealing with ', str(lon), str(lat))

                time_s = [float(each_one) for each_one in ts[2:]]
                compress_data[(lon, lat)] = dict()
                c_tools = compress_methods.compress(time_s)
                for ratio in ratios:
                    compress_data[(lon, lat)][('dft', ratio)] = c_tools.dft(ratio)
                    compress_data[(lon, lat)][('paa', ratio)] = c_tools.paa(ratio)

                print('--- Done with dft and paa')

                for error in errors:
                    compress_data[(lon, lat)][('dp', error)] = c_tools.modify_dp(error)
                    compress_data[(lon, lat)][('vw', error)] = c_tools.modify_vw(error)
                    compress_data[(lon, lat)][('opt', error)] = c_tools.modify_opt(error//2)

                print('--- Done with dp, vw and opt')

        f.close()

        print('Done with Compression of Cluster ', str(idx))


        # Generate Triangles and Compressed data volume
        # {(pi, pj, pk): {(c_method, ratio): [v1, v2, ..., vn]}}
        # At that timestamp, error tolerance is not useful -> Error would be measured by other distance

        if not os.path.exists('c_tri_vol/' + str(idx) + '.pkl'):
            infile = open('raw_tri_vol/' + str(idx) + '.pkl', 'rb')
            raw_tri_vol = pickle.load(infile)
            infile.close()
            c_vol = dict()

            for each_tri in raw_tri_vol:
                p1, p2, p3 = each_tri[0], each_tri[1], each_tri[2]
                c_vol[each_tri] = dict()

                for ratio in ratios:
                    dft_ts_1 = compress_data[p1][('dft', ratio)]
                    dft_ts_2 = compress_data[p2][('dft', ratio)]
                    dft_ts_3 = compress_data[p3][('dft', ratio)]

                    paa_ts_1 = compress_data[p1][('paa', ratio)]
                    paa_ts_2 = compress_data[p2][('paa', ratio)]
                    paa_ts_3 = compress_data[p3][('paa', ratio)]

                    for tstamp in range(len(dft_ts_1)):
                        c_vol[each_tri][('dft', ratio)] = \
                            c_vol[each_tri].get(('dft', ratio), []) + \
                            [cal_vol(list(p1) + [dft_ts_1[tstamp]],
                                     list(p2) + [dft_ts_2[tstamp]],
                                     list(p3) + [dft_ts_3[tstamp]])]

                        c_vol[each_tri][('paa', ratio)] = \
                            c_vol[each_tri].get(('paa', ratio), []) + \
                            [cal_vol(list(p1) + [paa_ts_1[tstamp]],
                                     list(p2) + [paa_ts_2[tstamp]],
                                     list(p3) + [paa_ts_3[tstamp]])]

                for error in errors:
                    dp_ts_1 = [x for x in compress_data[p1][('dp', error)]]
                    dp_ts_2 = [x for x in compress_data[p2][('dp', error)]]
                    dp_ts_3 = [x for x in compress_data[p3][('dp', error)]]
                    dp_ratio = float(sum([1 if val is not None else 0 for val in dp_ts_1]) +
                                     sum([1 if val is not None else 0 for val in dp_ts_2]) +
                                     sum([1 if val is not None else 0 for val in dp_ts_3])) / (3*len(dp_ts_1))

                    vw_ts_1 = [x for x in compress_data[p1][('vw', error)]]
                    vw_ts_2 = [x for x in compress_data[p2][('vw', error)]]
                    vw_ts_3 = [x for x in compress_data[p3][('vw', error)]]
                    vw_ratio = float(sum([1 if val is not None else 0 for val in vw_ts_1]) +
                                     sum([1 if val is not None else 0 for val in vw_ts_2]) +
                                     sum([1 if val is not None else 0 for val in vw_ts_3])) / (3 * len(vw_ts_1))

                    opt_ts_1 = [x for x in compress_data[p1][('opt', error)]]
                    opt_ts_2 = [x for x in compress_data[p2][('opt', error)]]
                    opt_ts_3 = [x for x in compress_data[p3][('opt', error)]]
                    opt_ratio = float(sum([1 if val is not None else 0 for val in opt_ts_1]) +
                                      sum([1 if val is not None else 0 for val in opt_ts_2]) +
                                      sum([1 if val is not None else 0 for val in opt_ts_3])) / (3 * len(opt_ts_1))

                    dp_ts_1, dp_ts_2, dp_ts_3, vw_ts_1, vw_ts_2, vw_ts_3, opt_ts_1, opt_ts_2, opt_ts_3 = \
                        interpolate(dp_ts_1), interpolate(dp_ts_2), interpolate(dp_ts_3),\
                        interpolate(vw_ts_1), interpolate(vw_ts_2), interpolate(vw_ts_3),\
                        interpolate(opt_ts_1), interpolate(opt_ts_2), interpolate(opt_ts_3)

                    for tstamp in range(len(dp_ts_1)):
                        c_vol[each_tri][('dp', dp_ratio)] = \
                            c_vol[each_tri].get(('dp', dp_ratio), []) + \
                            [cal_vol(list(p1) + [dp_ts_1[tstamp]],
                                     list(p2) + [dp_ts_2[tstamp]],
                                     list(p3) + [dp_ts_3[tstamp]])]

                        c_vol[each_tri][('vw', vw_ratio)] = \
                            c_vol[each_tri].get(('vw', vw_ratio), []) + \
                            [cal_vol(list(p1) + [vw_ts_1[tstamp]],
                                     list(p2) + [vw_ts_2[tstamp]],
                                     list(p3) + [vw_ts_3[tstamp]])]

                        c_vol[each_tri][('opt', opt_ratio)] = \
                            c_vol[each_tri].get(('opt', opt_ratio), []) + \
                            [cal_vol(list(p1) + [opt_ts_1[tstamp]],
                                     list(p2) + [opt_ts_2[tstamp]],
                                     list(p3) + [opt_ts_3[tstamp]])]

            with open('c_tri_vol/' + str(idx) + '.pkl', 'wb') as handle:
                pickle.dump(c_vol, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('compress_data.pkl', 'wb') as handle:
        pickle.dump(compress_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #vol_diff()
    #h_diff()
    angular_diff()


if __name__ == "__main__":
    main()