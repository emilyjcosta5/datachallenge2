from processing import iterate_through_data, save_space_grp_distribution

h5_path = '/gpfs/alpine/world-shared/stf011/junqi/smc/'
dict_dist = iterate_through_data(h5_path, save_fig=True)
save_space_grp_distribution(dict_dist)
