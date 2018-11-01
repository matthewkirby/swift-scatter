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
import scipy.optimize as opt


def calcmad(arr):
    return np.median(np.abs(arr - np.median(arr)))

"""
Notes for generalizing
Pass in names of columns for x and y and if they are log (default to true)
"""
def compute_expected_lnlamobs(cat, nbins=30):
    lnmmin = np.min(cat['lnm200m'])
    lnmmax = np.max(cat['lnm200m'])
    bins = np.linspace(lnmmin, lnmmax, nbins+1)

    data = np.zeros((nbins, 4))
    data[:,0] = bins[:-1]
    data[:,1] = bins[1:]
    fitdata = DataFrame(data, columns=['low', 'high', 'median_lnx', 'median_lny'])

    for idx, row in fitdata.iterrows():
        subcat = cat[(cat['lnm200m'] > row['low']) &
                     (cat['lnm200m'] <= row['high'])]
        if len(subcat) == 0:
            row['median_lnx'] = np.NaN
            row['median_lny'] = np.NaN
            continue
        row['median_lnx'] = np.median(subcat['lnm200m'])
        row['median_lny'] = np.median(np.log(subcat['obs_richness']))
    fitdata = fitdata.dropna()

    lnpivot = np.median(cat['lnm200m'])
    xdata = fitdata['median_lnx'] - lnpivot
    ydata = fitdata['median_lny']

    model = lambda x, p1, p2: p1 + p2*x
    fit, cov = opt.curve_fit(model, xdata, ydata)

    cat['expected_lnlamobs'] = fit[0] + fit[1]*(cat['lnm200m'] - lnpivot)
    cat['res_lnlamobs'] = np.log(cat['obs_richness']) - cat['expected_lnlamobs']
    return cat, fitdata


def compute_expected_lnlx(cat, nbins=30):
    cat['ln_lamobs'] = np.log(cat['obs_richness'])
    lnxmin = np.min(cat['ln_lamobs'])
    lnxmax = np.max(cat['ln_lamobs'])
    bins = np.linspace(lnxmin, lnxmax, nbins+1)

    data = np.zeros((nbins, 4))
    data[:,0] = bins[:-1]
    data[:,1] = bins[1:]
    fitdata = DataFrame(data, columns=['low', 'high', 'median_lnx', 'median_lny'])

    for idx, row in fitdata.iterrows():
        subcat = cat[(cat['ln_lamobs'] > row['low']) &
                     (cat['ln_lamobs'] <= row['high'])]
        if len(subcat) == 0:
            row['median_lnx'] = np.NaN
            row['median_lny'] = np.NaN
            continue
        row['median_lnx'] = np.median(subcat['ln_lamobs'])
        row['median_lny'] = np.median(subcat['true_lnLx'])
    fitdata = fitdata.dropna()

    lnpivot = np.median(cat['ln_lamobs'])
    xdata = fitdata['median_lnx'] - lnpivot
    ydata = fitdata['median_lny']

    model = lambda x, p1, p2: p1 + p2*x
    fit, cov = opt.curve_fit(model, xdata, ydata)

    cat['expected_lnLx'] = fit[0] + fit[1]*(cat['ln_lamobs'] - lnpivot)
    cat['res_lnLx'] = cat['true_lnLx'] - cat['expected_lnLx']
    return cat, fitdata


def compute_expected_lnlamtrue(cat, nbins=30):
    cat['ln_lamtrue'] = np.log(cat['true_richness'])
    lnxmin = np.min(cat['lnm200m'])
    lnxmax = np.max(cat['lnm200m'])
    bins = np.linspace(lnxmin, lnxmax, nbins+1)

    data = np.zeros((nbins, 4))
    data[:,0] = bins[:-1]
    data[:,1] = bins[1:]
    fitdata = DataFrame(data, columns=['low', 'high', 'median_lnx', 'median_lny'])

    for idx, row in fitdata.iterrows():
        subcat = cat[(cat['lnm200m'] > row['low']) &
                     (cat['lnm200m'] <= row['high'])]
        if len(subcat) == 0:
            row['median_lnx'] = np.NaN
            row['median_lny'] = np.NaN
            continue
        row['median_lnx'] = np.median(subcat['lnm200m'])
        row['median_lny'] = np.median(subcat['ln_lamtrue'])
    fitdata = fitdata.dropna()

    lnpivot = np.median(cat['lnm200m'])
    xdata = fitdata['median_lnx'] - lnpivot
    ydata = fitdata['median_lny']

    model = lambda x, p1, p2: p1 + p2*x
    fit, cov = opt.curve_fit(model, xdata, ydata)

    cat['expected_lnlamtrue'] = fit[0] + fit[1]*(cat['lnm200m'] - lnpivot)
    cat['res_lnlamtrue'] = cat['ln_lamtrue'] - cat['expected_lnlamtrue']
    return cat, fitdata




def main():
    masshist = True
    lamobsresid = False
    lamtrueresid = False
    lxresid = False

    lognorm = True




    cfgname = '../config/config-swift-sigext.ini'
    modelname = '../config/model-swift-sigext.ini'

    # Load the config file in the above directory
    cfgin = cp.ConfigParser()
    cfgin.read(cfgname)
    modelin = cp.ConfigParser()
    modelin.read(modelname)
    basemodel = {str(x):float(y) for x,y in modelin.items('Model')}


    # The model parameters to grid in
    variations = [{'r':0.0, 'sig0lam':0.50},
                  {'r':0.0, 'sig0lam':0.60}]


    lab1 = 'sigintr={:.2f}'.format(variations[0]['sig0lam'])
    lab2 = 'sigintr={:.2f}'.format(variations[1]['sig0lam'])

    scattertype = ''
    ptitle = 'Gaussian'
    if lognorm:
        scattertype = '_lognormal'
        ptitle = 'Log-Normal'

    file_names = ms.build_file_names(cfgin, 0, '0.0_{:.2f}{}'.format(variations[0]['sig0lam'], scattertype))
    cat1 = read_csv(file_names['obs_catalog'], sep='\t') 
    file_names = ms.build_file_names(cfgin, 0, '0.0_{:.2f}{}'.format(variations[1]['sig0lam'], scattertype))
    cat2 = read_csv(file_names['obs_catalog'], sep='\t') 

    cat1 = cat1[cat1['obs_richness'] > 0.0]
    cat2 = cat2[cat2['obs_richness'] > 0.0]


    if lamobsresid:
        print("Making lamobs plots")
        cat1, fitdata1 = compute_expected_lnlamobs(cat1)
        cat2, fitdata2 = compute_expected_lnlamobs(cat2)

        plt.loglog(np.exp(cat2['lnm200m']), cat2['obs_richness'], '.', color='g', alpha=0.1, ms=1, label=lab2)
        plt.loglog(np.exp(cat1['lnm200m']), cat1['obs_richness'], '.', color='k', alpha=0.1, ms=1, label=lab1)
        plt.loglog(np.exp(cat1['lnm200m']), np.exp(cat1['expected_lnlamobs']), '-k', label=lab1)
        plt.loglog(np.exp(cat2['lnm200m']), np.exp(cat2['expected_lnlamobs']), '-g', label=lab2)
        plt.xlabel(r'$M_{200m}$')
        plt.ylabel(r'$\lambda^{ob}$')
        plt.legend()
        # plt.tight_layout()
        # plt.savefig('lamobs-scatter.png')
        plt.show()
        plt.clf()

        lnmbandlow, lnmbandhigh = np.log(9.e13), np.log(1.e14)
        subcat1 = cat1[cat1['lnm200m'] > lnmbandlow]
        subcat1 = subcat1[subcat1['lnm200m'] < lnmbandhigh]
        subcat2 = cat2[cat2['lnm200m'] > lnmbandlow]
        subcat2 = subcat2[subcat2['lnm200m'] < lnmbandhigh]

        plt.hist(subcat1['res_lnlamobs'], bins=50, label=lab1, histtype='step')
        plt.hist(subcat2['res_lnlamobs'], bins=50, label=lab2, histtype='step')
        plt.xlabel(r'$\ln\lambda^{ob} - \langle \lambda^{ob}|M\rangle$')
        plt.gca().set_yscale('log')
        plt.title('For $\ln M_{{200m}} \in [{:.2f}, {:.2f}]$'.format(lnmbandlow, lnmbandhigh))
        plt.legend()
        plt.tight_layout()
        plt.savefig('lamobs-resid.png')
        plt.clf()


    if lamtrueresid:
        print("Making lamtrue plots")
        # cat1, fitdata1 = compute_expected_lnlamtrue(cat1)
        # cat2, fitdata2 = compute_expected_lnlamtrue(cat2)

        plt.loglog(np.exp(cat2['lnm200m']), cat2['true_richness'], '.', color='g', alpha=0.1, ms=1, label=lab2)
        plt.loglog(np.exp(cat1['lnm200m']), cat1['true_richness'], '.', color='k', alpha=0.1, ms=1, label=lab1)
        
        # xs = np.linspace(min(cat2['lnm200m']), max(cat2['lnm200m']))
        # plt.loglog(np.exp(cat2['lnm200m']), np.exp(cat2['expected_lnlamtrue']), '-g', label=lab2)

        plt.title(ptitle)
        plt.xlabel(r'$M_{200m}$')
        plt.ylabel(r'$\lambda^{tr}$')
        plt.legend()
        plt.show()
        plt.clf()



    # Make the richness cut
    cat1 = cat1[cat1['obs_richness'] > 25.]
    cat1 = cat1[cat1['obs_richness'] < 35.]
    cat2 = cat2[cat2['obs_richness'] > 25.]
    cat2 = cat2[cat2['obs_richness'] < 35.]


    # Make the mass histogram
    if masshist:
        print("Making mass histogram")
        plt.hist(cat1['lnm200m'], bins=50, label=lab1, histtype='step')
        plt.hist(cat2['lnm200m'], bins=50, label=lab2, histtype='step')
        plt.axvline(np.log(1.e13), color='k')
        plt.gca().set_yscale('log')
        plt.xlabel(r'$\ln M_{200m}$')
        plt.legend()
        # plt.savefig('mass-hist.png')
        # plt.tight_layout()
        plt.show()
        plt.clf()


    # Make the Lx histogram 
    if lxresid:
        print("Making Lx plots")
        cat1, fitdata1 = compute_expected_lnlx(cat1)
        cat2, fitdata2 = compute_expected_lnlx(cat2)

        plt.loglog(cat1['obs_richness'], np.exp(cat1['true_lnLx']), '.', color='k', alpha=0.1, ms=1, label=lab1)
        plt.loglog(cat2['obs_richness'], np.exp(cat2['true_lnLx']), '.', color='g', alpha=0.1, ms=1, label=lab2)
        plt.loglog(cat1['obs_richness'], np.exp(cat1['expected_lnLx']), '-k', label=lab1)
        plt.loglog(cat2['obs_richness'], np.exp(cat2['expected_lnLx']), '-g', label=lab2)
        plt.xlabel(r'$\lambda^{ob}$')
        plt.ylabel(r'$L_x$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('lx-scatter.png')
        plt.clf()

        cat1_kmad2 = 1.4826*calcmad(cat1[cat1['true_lnLx'] > np.log(1.e42)]['res_lnLx'])
        cat2_kmad2 = 1.4826*calcmad(cat2[cat2['true_lnLx'] > np.log(1.e42)]['res_lnLx'])
        
        plt.hist(cat1['res_lnLx'], bins=50, label=lab1+' $\sigma_{{Lx|\lambda^{{ob}}}} = {:.3f}$'.format(cat1_kmad2), histtype='step')
        plt.hist(cat2['res_lnLx'], bins=50, label=lab2+' $\sigma_{{Lx|\lambda^{{ob}}}} = {:.3f}$'.format(cat2_kmad2), histtype='step')
        plt.xlabel(r'$\ln L_x - \langle \ln L_x | \lambda^{ob} \rangle$')
        plt.gca().set_yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig('lx-resid.png')







    return







if __name__ == "__main__":
    main()
