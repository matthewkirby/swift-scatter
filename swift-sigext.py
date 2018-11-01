import matplotlib
#matplotlib.use('Agg')
import sys, os
sys.path.append(os.path.join(os.environ['RICHESTRMCL'], 'catalogs'))
import numpy as np
import configparser as cp
import generate_mass_catalog as gmc
import draw_true_properties as dtp
import draw_observed_properties as dop
import hmf_library as hmflib
import choose_model as cm
import mock_setup as ms
import fit_properties as fp
from pandas import DataFrame
import matplotlib.pyplot as plt


def main():
    # Load the config file in the above directory
    cfg_name = 'config/config-swift-sigext.ini'
    cfgin = cp.ConfigParser()
    cfgin.read(cfg_name)

    # Load lookup tables and other constants
    sinfo, cosmology = ms.import_cosmology(cfgin)
    splines = ms.build_splines(cfgin, sinfo)

    # Load the base model
    modelin = cp.ConfigParser()
    modelin.read('config/model-swift-sigext.ini')
    basemodel = {str(x):float(y) for x,y in modelin.items('Model')}
    meanpath = os.path.join(os.environ['DATADIR'], 'xray-scatter/mocks/swift-sigext/external-data/swift_Lx-M.dat')

    # The model parameters to grid in
    rlist = [0.0] #[float(sys.argv[1])] #[0.0] #np.array([0.4, 0.6])
    if sys.argv[1] == '1':
        sig0lamlist = np.array([0.0001, 0.1, 0.2])
    elif sys.argv[1] == '2':
        sig0lamlist = np.array([0.3, 0.4, 0.5])
    else:
        sig0lamlist = np.array([0.6, 0.7, 0.8])
    # sig0lamlist = [0.5, 0.5] #np.array([0.1, 0.25, 0.4, 0.55, 0.7])
    variations = []
    for r in rlist:
        for s in sig0lamlist:
            variations.append({'r':r, 'sig0lam':s})
    print variations

    drawer = dop.Converter()
    drawer.setup_iCDF_grid()


    lognormal = True


    scattermodel = 'normal'
    fileextension = ''
    if lognormal:
        scattermodel = 'lognormal'
        fileextension = '_lognormal'


    for run_number in np.arange(len(variations)):
        print "Run %i/%i" % (run_number+1, len(variations))

        # Select the model parameters
        model = basemodel.copy()
        model.update(variations[run_number])
        modifier = '{:.1f}_{:.2f}{}'.format(model['r'], model['sig0lam'], fileextension)
        file_names = ms.build_file_names(cfgin, run_number, modifier)

        print("> Drawing masses")
        gmc.generate_mass_catalog(cfgin, model, sinfo, file_names, 13.)
        masscat = np.load(file_names['mass_catalog'] + '.npy')
        catalog = DataFrame(masscat, columns=['lnm200m', 'z'])

        print("> Drawing true and obs richness")
        catalog = dtp.draw_true_richness(catalog, model, cfgin['General']['richnessmean'], scattermodel)

        # Temp
        # catalog['obs_richness'] = np.ones(len(catalog))*0.5
        # subcat = catalog[['lnm200m', 'true_richness', 'obs_richness']]
        # subcat.to_csv(file_names['obs_catalog'], sep='\t', index_label='id', float_format='%.6e')
        # continue

        catalog = dop.draw_observed_richness(catalog, drawer)
        # print("NO RICHNESSSS CUTTTTT")
        catalog = catalog[catalog['obs_richness'] < 35.]
        catalog = catalog[catalog['obs_richness'] > 25.]

        print("> Draw true Lx")
        catalog['lnm500c'] = np.array([splines['mass_conversion'](row['lnm200m'], row['z']) for idx, row in catalog.iterrows()])
        catalog = dtp.draw_true_observable_from_file(catalog, model, meanpath, 1.e42)

        # Save the catalog
        ms.save_observed_subcatalog(catalog, file_names)

    plt.show()



    return





if __name__ == "__main__":
    main()
