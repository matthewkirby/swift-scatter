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
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt


def calcmad(arr):
    return np.median(np.abs(arr - np.median(arr)))



def main():
    outputfile = 'full-output.txt'
    cfgname = 'config/config-swift-sigext.ini'
    modelname = 'config/model-swift-sigext.ini'
    lognormal = True


    # Load the config file in the above directory
    cfgin = cp.ConfigParser()
    cfgin.read(cfgname)
    modelin = cp.ConfigParser()
    modelin.read(modelname)
    basemodel = {str(x):float(y) for x,y in modelin.items('Model')}


    # The model parameters to grid in
    rlist = [0.0] # np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]) #[0.0]
    # sig0lamlist = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
    #                         0.45, 0.5, 0.55, 0.6, 0.65, 0.7]) # [0.30]
    sig0lamlist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    variations = []
    for r in rlist:
        for s in sig0lamlist:
            variations.append({'r':r, 'sig0lam':s})


    # Write the file header
    try:
        os.remove(outputfile)
    except OSError:
        pass
    with open(outputfile, 'a+') as f:
        f.write("1 - No Lx cuts\n")
        f.write("2 - Lx > 1e42\n")
        f.write("3 - Cut 1/155 smallest Lx\n")
        f.write("r\tsigint\tamp1\tslope1\tstd1\tmad1\tamp2\tslope2\tstd2\tmad2\tfraccut\tamp3\tslope3\tstd3\tmad3\n")


    scattermodel = ''
    if lognormal:
        scattermodel = '_lognormal'


    # Do the fits
    for submod in variations:
        print submod['r'], submod['sig0lam']
        # print("> Building file name and laoding the catalog")
        model = basemodel.copy()
        model.update(submod)
        modifier = '{:.1f}_{:.2f}{}'.format(model['r'], model['sig0lam'], scattermodel)
        file_names = ms.build_file_names(cfgin, 0, modifier)
        catalog = read_csv(file_names['obs_catalog'], sep='\t')

        catalog = catalog[catalog['obs_richness'] > 25.]
        catalog = catalog[catalog['obs_richness'] < 35.]


        print("> Making cut catalogs")
        cat2 = catalog.copy()
        cat2 = cat2[cat2['true_lnLx'] > np.log(1.e42)]
        fraccut = 1. - float(len(cat2))/float(len(catalog))
        cat3 = catalog.copy()
        cat3 = cat3.sort_values(['true_lnLx'])
        cat3 = cat3[int(float(len(cat3))/155.):]


        print("> Fitting 1:{}, 2:{}, 3:{}".format(len(catalog), len(cat2), len(cat3)))
        fitdata1, bestfit1 = fp.fit_relation(catalog, nbins=50)
        catalog['<lnLx>'] = bestfit1[0] + bestfit1[1] * np.log(catalog['obs_richness'].values/30.)
        catalog['res'] = catalog['true_lnLx'].values - catalog['<lnLx>'].values

        fitdata2, bestfit2 = fp.fit_relation(cat2, nbins=50)
        cat2['<lnLx>'] = bestfit2[0] + bestfit2[1] * np.log(cat2['obs_richness'].values/30.)
        cat2['res'] = cat2['true_lnLx'].values - cat2['<lnLx>'].values

        fitdata3, bestfit3 = fp.fit_relation(cat3, nbins=50)
        cat3['<lnLx>'] = bestfit3[0] + bestfit3[1] * np.log(cat3['obs_richness'].values/30.)
        cat3['res'] = cat3['true_lnLx'].values - cat3['<lnLx>'].values


        print("> Doing stats")
        std1 = catalog['res'].std()
        kmad1 = 1.4826*calcmad(catalog['res'])
        std2 = cat2['res'].std()
        kmad2 = 1.4826*calcmad(cat2['res'])
        std3 = cat3['res'].std()
        kmad3 = 1.4826*calcmad(cat3['res'])


        # print "{:.3f} {:.3f}".format(std1, kmad1)
        # print "{:.3f} {:.3f}".format(std2, kmad2)
        # print "{:.3f} {:.3f}".format(std3, kmad3)

        # print("> Plotting fit")
        # plt.loglog(cat2['obs_richness'], np.exp(cat2['true_lnLx']), '.k', alpha=0.1, ms=1)
        # plt.loglog(fitdata2['median_lamobs'], np.exp(fitdata2['median_lnLx']), 'ob')
        # xs = np.linspace(25, 35, 30)
        # ys = bestfit2[0] + bestfit2[1]*np.log(xs/30)
        # plt.loglog(xs, np.exp(ys), '-r')
        # plt.xlabel(r'$\lambda_{obs}$')
        # plt.ylabel(r'$\ln L_x$')
        # plt.show()

        # print("> Plotting residuals")
        # plt.hist(catalog['res'].values, bins=200, histtype='step', color='black', label='All', normed=True)
        # plt.axvline(catalog['res'].median(), color='black')
        # plt.hist(cat2['res'].values, bins=200, histtype='step', color='blue', label='Lx > 1e42', normed=True)
        # plt.axvline(cat2['res'].median(), color='black')
        # plt.hist(cat3['res'].values, bins=200, histtype='step', color='red', label='Throw away 1000 smallest Lx', normed=True)
        # plt.axvline(cat3['res'].median(), color='black')
        # plt.xlabel(r'$\ln L_x - \langle \ln L_x|\lambda^{ob}\rangle$')
        # plt.legend()
        # plt.show()


        # Output the results
        with open(outputfile, 'a+') as f:
            f.write("{:.1f}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n"
                    .format(model['r'], model['sig0lam'],
                            bestfit1[0], bestfit1[1], std1, kmad1,
                            bestfit2[0], bestfit2[1], std2, kmad2, fraccut,
                            bestfit3[0], bestfit3[1], std3, kmad3))

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
