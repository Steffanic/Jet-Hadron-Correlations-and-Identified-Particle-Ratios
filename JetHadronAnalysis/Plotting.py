from ROOT import TH1D #type: ignore
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List

def plotTH1(histograms: Union[List[TH1D], TH1D], data_labels: Union[List[str], str], error_bands: Optional[Union[List, np.ndarray, float, int]], error_band_labels: Optional[Union[List, str]],  title: str, xtitle: str, ytitle: str, output_path: str, logy: bool = False, logx: bool = False):
    plt.figure()
    if not isinstance(histograms, list):
        histograms = [histograms]
    if not isinstance(data_labels, list):
        data_labels = [data_labels]

    for histogram, data_label in zip(histograms, data_labels):
            
        # check that the histogram is a TH1D, if a th2d is passed, then weird non-errorsa can happen
        assert isinstance(histogram, TH1D)
        # first conver the histogram to a content array and an error array, also make the x-axis array
        x = np.array([histogram.GetBinCenter(i) for i in range(1, histogram.GetNbinsX()+1)])
        content = np.zeros(histogram.GetNbinsX())
        error = np.zeros(histogram.GetNbinsX())
        for i in range(1, histogram.GetNbinsX()+1):
            content[i-1] = histogram.GetBinContent(i)
            error[i-1] = histogram.GetBinError(i)
        # now plot the histogram
        plt.errorbar(x=x, y=content, yerr=error, fmt='o', label=data_label)

    if error_bands is not None:
        assert len(error_bands) == len(histograms)
        if not isinstance(error_bands, list):
            error_bands = [error_bands]
        if not isinstance(error_band_labels, list):
            error_band_labels = [error_band_labels]
        for error_band, error_band_label, histogram in zip(error_bands, error_band_labels, histograms):
            assert isinstance(error_band, (np.ndarray, float, int))
            if isinstance(error_band, (float, int)):
                error_band = np.full(histogram.GetNbinsX(), error_band)
            x = np.array([histogram.GetBinCenter(i) for i in range(1, histogram.GetNbinsX()+1)])
            content = np.zeros(histogram.GetNbinsX())
            for i in range(1, histogram.GetNbinsX()+1):
                content[i-1] = histogram.GetBinContent(i)
            assert error_band.shape == content.shape
            plt.fill_between(x, content-error_band, content+error_band, alpha=0.3, label=error_band_label)

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
   
def plotArray(x_arrays: Union[List[np.ndarray], np.ndarray], y_arrays: Union[List[np.ndarray], np.ndarray], error_arrays:Union[List[np.ndarray], np.ndarray], data_labels:Union[List[str], str], error_bands_x:Optional[Union[List, np.ndarray, float, int]], error_bands_y:Optional[Union[List, np.ndarray, float, int]], error_band_labels:Optional[Union[List, str]], title: str, xtitle: str, ytitle: str, output_path: str, logy: bool = False, logx: bool = False):
    plt.figure()
    if not isinstance(x_arrays, list):
        x_arrays = [x_arrays]
    if not isinstance(y_arrays, list):
        y_arrays = [y_arrays]
    if not isinstance(error_arrays, list):
        error_arrays = [error_arrays]
    if not isinstance(data_labels, list):
        data_labels = [data_labels]
    for x_array, y_array, error_array, data_label in zip(x_arrays, y_arrays, error_arrays, data_labels):
        
        plt.errorbar(x_array, y_array, yerr=error_array, fmt='o', label=data_label)

    if error_bands_x is not None:
        if not isinstance(error_bands_x, list):
            error_bands_x = [error_bands_x]
        if not isinstance(error_bands_y, list):
            error_bands_y = [error_bands_y]
        if not isinstance(error_band_labels, list):
            error_band_labels = [error_band_labels]
        for error_band_x, error_band_y, error_band_label in zip(error_bands_x, error_bands_y,error_band_labels):
            assert isinstance(error_band, (np.ndarray, float, int))
            if isinstance(error_band, (float, int)):
                error_band = np.full(y_array.shape, error_band)
            assert error_band.shape == array.shape
            plt.fill_between(x, array-error_band, array+error_band, alpha=0.3, label=error_band_label)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.savefig(output_path)
    plt.close()

def plotArrays(x_data:dict, y_data:dict, error:dict, data_label:dict, format_style:dict, error_bands:Optional[dict], error_bands_label:Optional[dict], title: str, xtitle: str, ytitle: str, output_path: str, logy: bool = False, logx: bool = False):
    plt.figure()
    for key in x_data.keys():
        if error[key] is None:
            plt.plot(x_data[key], y_data[key], format_style[key], label=data_label[key])   
        plt.errorbar(x_data[key], y_data[key], yerr=error[key], fmt=format_style[key], label=data_label[key])

    if error_bands is not None and error_bands_label is not None:
        for key in error_bands.keys():
            if not isinstance(error_bands[key], list):
                error_bands[key] = [error_bands[key]]
            if not isinstance(error_bands_label[key], list):
                error_bands_label[key] = [error_bands_label[key]]
            for band, label in zip(error_bands[key], error_bands_label[key]):
                assert isinstance(error_bands[key], (np.ndarray, float, int))
                if isinstance(error_bands[key], (float, int)):
                    error_bands[key] = np.full(y_data[key].shape, error_bands[key])
                assert error_bands[key].shape == y_data[key].shape
                plt.fill_between(x_data[key], y_data[key]-error_bands[key], y_data[key]+error_bands[key], alpha=0.3, label=error_bands_label[key])
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
   