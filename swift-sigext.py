import matplotlib
matplotlib.use('Agg')
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
    modifier = sys.argv[1]

    # Load lookup tables and other constants
    sinfo, cosmology = ms.import_cosmology(cfgin)
    splines = ms.build_splines(cfgin, sinfo)
    mgas_rescale = 1.e12

    # Load the base model
    modelin = cp.ConfigParser()
    modelin.read('config/model-swift-sigext.ini')
    basemodel = {str(x):float(y) for x,y in modelin.items('Model')}
    meanpath = os.path.join(os.environ['DATADIR'], 'xray-scatter/mocks/swift-sigext/external-data/swift_Lx-M.dat')

    # The model parameters to grid in
    rlist = [0.8] #np.array([0.4, 0.6])
    sig0lamlist = [0.7] #np.array([0.1, 0.25, 0.4, 0.55, 0.7])
    variations = []
    for r in rlist:
        for s in sig0lamlist:
            variations.append({'r':r, 'sig0lam':s})

    drawer = dop.Converter()
    drawer.setup_iCDF_grid()

    for run_number in np.arange(len(variations)):
        print "Run %i/%i" % (run_number+1, len(variations))

        print("> Selecting model")
        file_names = ms.build_file_names(cfgin, run_number, modifier)
        model = basemodel.copy()
        model.update(variations[run_number])

        print("> Drawing masses")
        gmc.generate_mass_catalog(cfgin, model, sinfo, file_names, 13.)
        masscat = np.load(file_names['mass_catalog'] + '.npy')
        catalog = DataFrame(masscat, columns=['lnm200m', 'z'])

        print("> Drawing true/obs richness and make cut")
        catalog = dtp.draw_true_richness(catalog, model, cfgin['General']['richnessmean'])
        catalog = dop.draw_observed_richness(catalog, drawer)
        catalog = catalog[catalog['obs_richness'] < 35]
        catalog = catalog[catalog['obs_richness'] > 25]

        print("> Draw true Lx")
        catalog['lnm500c'] = np.array([splines['mass_conversion'](row['lnm200m'], row['z']) for idx, row in catalog.iterrows()])
        catalog = dtp.draw_true_observable_from_file(catalog, model, meanpath, 1.e42)
        catalog = catalog[catalog['true_Lx'] > 0.0]
        ms.save_observed_subcatalog(catalog, file_names)

        print("> Fitting mean Lx-lamobs relation")
        print("fitting {} objects".format(len(catalog)))
        fitdata, bestfit = fp.fit_relation(catalog, nbins=20)

        print("> Computing sig_Lx|lamobs")
        catalog['means'] = bestfit[0]*np.power(catalog['obs_richness'].values/30., bestfit[1])
        squarediff = np.power(np.log(catalog['true_Lx'].values) - np.log(catalog['means'].values), 2.)
        scatter = np.sqrt(np.sum(squarediff)/len(squarediff))
        with open('outputs/sigmaLx_at_fixed_lamobs{}.txt'.format(modifier), 'a+') as f:
            f.write("{} {} {}\n".format(model['r'], model['sig0lam'], scatter))

        # print("> Plotting fit")
        # plt.loglog(catalog['obs_richness'], catalog['true_Lx'], '.k', ms=1, alpha=0.1)
        # plt.loglog(fitdata['mean_lamobs'], fitdata['mean_Lx'], 'ob')
        # xs = np.linspace(25, 35, 30)
        # ys = bestfit[0]*pow(xs/30., bestfit[1])
        # plt.loglog(xs, ys, '-r')
        # plt.xlabel(r'$\lambda_{obs}$')
        # plt.ylabel(r'$L_x$')
        # plt.savefig('example_fit_r0.png')


    return





if __name__ == "__main__":
    main()
