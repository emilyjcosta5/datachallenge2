from processing import iterate_through_data, save_space_grp_distribution

h5_path = '/gpfs/alpine/world-shared/stf011/junqi/smc/train/'
dict_dist = iterate_through_data(h5_path, save_fig=True, fig_name='dist_on_summit')
save_space_grp_distribution(dict_dist, file_name='dist_on_summit')
