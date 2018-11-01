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


    cfgname = '../config/config-swift-sigext.ini'
    modelname = '../config/model-swift-sigext.ini'

    # Load the config file in the above directory
    cfgin = cp.ConfigParser()
    cfgin.read(cfgname)
    modelin = cp.ConfigParser()
    modelin.read(modelname)
    basemodel = {str(x):float(y) for x,y in modelin.items('Model')}


    # The model parameters to grid in
    rset = 0.0
    # siglist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    siglist = [0.5, 0.6]
    variations = [{'r':rset, 'sig0lam':sig} for sig in siglist]



    # Make the mass histogram
    print("Making mass histogram")
    for var in variations:
        modifier = '{:.1f}_{:.2f}_lognormal'.format(var['r'], var['sig0lam'])
        file_names = ms.build_file_names(cfgin, 0, modifier)
        cat = read_csv(file_names['obs_catalog'], sep='\t')

        cat = cat[cat['obs_richness'] > 25.]
        cat = cat[cat['obs_richness'] < 35.]

        lab = 'sigintr={}'.format(var['sig0lam'])
        plt.hist(cat['lnm200m'], bins=50, label=lab, histtype='step')

    plt.axvline(np.log(1.e13), color='k')
    plt.gca().set_yscale('log')
    plt.xlabel(r'$\ln M_{200m}$')
    plt.legend()
    # plt.savefig('mass-hist.png')
    # plt.tight_layout()
    plt.show()
    plt.clf()






    return







if __name__ == "__main__":
    main()
