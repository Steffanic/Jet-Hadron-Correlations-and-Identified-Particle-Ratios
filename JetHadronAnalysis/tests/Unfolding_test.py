import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from JetHadronAnalysis.Unfolding import Unfolder, ResponseMatrix
import pprint

@pytest.fixture
def mock_response_matrix():
    mock_response_matrix =np.array([
                            [0.6, 0.1, 0.05],
                            [0.05, 0.6, 0.15],
                            [0.05, 0.3, 0.5],
                            [0.3, 0.3, 0.3]
                            ])
    return mock_response_matrix

@pytest.fixture
def mock_response_matrix_error():
    mock_response_matrix_error = np.array([
                            [0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.1]
                            ])
    return mock_response_matrix_error

@pytest.fixture
def mock_cause():
    mock_cause = np.array([1000, 100, 300])
    return mock_cause

@pytest.fixture
def mock_data(mock_cause, mock_response_matrix):
    mock_data = np.dot(mock_response_matrix, mock_cause)
    return mock_data

@pytest.fixture
def mock_data_error(mock_data):
    mock_data_error = mock_data * 0.1
    return mock_data_error

@pytest.fixture
def mock_effects():
    pionEffect={'pionCause': 0.58, 'protonCause': 0.03, 'kaonCause': 0.14}
    protonEffect={'pionCause': 0.00, 'protonCause': 0.41, 'kaonCause': 0.10}
    kaonEffect={'pionCause': 0.07, 'protonCause': 0.40, 'kaonCause': 0.29}
    trashEffect={'pionCause': 0.35, 'protonCause': 0.16, 'kaonCause': 0.47}
    mock_effects = [pionEffect, protonEffect, kaonEffect, trashEffect]
    return mock_effects

@pytest.fixture
def mock_effect_errors():
    pionEffectError={'pionCauseError': 0.44, 'protonCauseError': 0.18, 'kaonCauseError': 0.08}
    protonEffectError={'pionCauseError': 0.02, 'protonCauseError': 0.41, 'kaonCauseError': 0.15}
    kaonEffectError={'pionCauseError': 0.10, 'protonCauseError': 0.47, 'kaonCauseError': 0.47}
    trashEffectError={'pionCauseError': 0.33, 'protonCauseError': 0.84, 'kaonCauseError': 0.55}
    mock_effects = [pionEffectError, protonEffectError, kaonEffectError, trashEffectError]
    return mock_effects

@pytest.fixture
def unfolder_w_mock_response(mock_response_matrix, mock_response_matrix_error):
    response = ResponseMatrix()
    response.matrix = mock_response_matrix
    response.error = mock_response_matrix_error
    unfolder = Unfolder(response)
    return unfolder

@pytest.fixture
def unfolder(mock_effects, mock_effect_errors):
    response = ResponseMatrix()
    response.buildResponseMatrix(*mock_effects)
    response.buildResponseMatrixError(*mock_effect_errors)
    unfolder = Unfolder(response)
    return unfolder

def test_build_response_matrix(mock_effects):
    response = ResponseMatrix()
    response.buildResponseMatrix(*mock_effects)
    assert np.allclose(response.matrix, np.array([
                            [0.58, 0.03, 0.14],
                            [0.00, 0.41, 0.10],
                            [0.07, 0.40, 0.29],
                            [0.35, 0.16, 0.47]]))
    
@pytest.fixture 
def mock_observations():
    mock_observations = np.array([ 19035.96622961,  1482.66430983,  3514.01694456, 31874.04273328 ])
    return mock_observations

@pytest.fixture
def mock_observations_error(mock_observations):
    mock_observations_error = mock_observations * 0.1
    return mock_observations_error


def test_unfold_mock_data(unfolder, mock_observations, mock_observations_error):
    unfolded = unfolder.unfold(mock_observations, mock_observations_error, None, None, test_statistic='chi2', test_statistic_stopping=1.0)
    refolded = unfolder.refold(unfolded["unfolded"])
    print("unfolded: ", pprint.pformat(unfolded["unfolded"]))
    print("stat_error: ", pprint.pformat(unfolded["stat_err"]))
    print("sys_error: ", pprint.pformat(unfolded["sys_err"]))
    print("refolded: ", pprint.pformat(refolded))
    print(f"unfolding_matrix: {pprint.pformat(unfolded['unfolding_matrix'])}")
    print(f"num_iterations: {pprint.pformat(unfolded['num_iterations'])}")
    print(f"ts_iter: {pprint.pformat(unfolded['ts_iter'])}")
    assert np.allclose(refolded, mock_observations)

def test_unfolding_ks(unfolder, mock_data, mock_data_error):
    # normalize the mock data and error
    # mock_data = mock_data / np.sum(mock_data)
    # mock_data_error = mock_data_error / np.sum(mock_data)
    unfolded_dict_ks = unfolder.unfold(mock_data, mock_data_error, None, None, test_statistic='ks', test_statistic_stopping=0.01)
    unfolded_ks = np.array(unfolded_dict_ks["unfolded"])
    stat_error_ks = unfolded_dict_ks["stat_err"]
    sys_error_ks = unfolded_dict_ks["sys_err"]
    unfolding_matrix_ks = unfolded_dict_ks["unfolding_matrix"]
    num_iterations_ks = unfolded_dict_ks["num_iterations"]
    ts_iter_ks = unfolded_dict_ks["ts_iter"]
    refolded_ks = unfolder.refold(unfolded_ks)
    print("unfolded: ", pprint.pformat(unfolded_ks))
    print("stat_error: ", pprint.pformat(stat_error_ks))
    print("sys_error: ", pprint.pformat(sys_error_ks))
    print("refolded: ", pprint.pformat(refolded_ks))
    print(f"unfolding_matrix: {pprint.pformat(unfolding_matrix_ks)}")
    print(f"num_iterations: {pprint.pformat(num_iterations_ks)}")
    print(f"ts_iter: {pprint.pformat(ts_iter_ks)}")

    assert np.allclose(refolded_ks, mock_data)

    
def test_unfolding_chi2(unfolder, mock_data, mock_data_error):
    # mock_data = mock_data / np.sum(mock_data)
    # mock_data_error = mock_data_error / np.sum(mock_data)
    unfolded_dict_chi2 = unfolder.unfold(mock_data, mock_data_error, None, None, test_statistic='chi2', test_statistic_stopping=0.01)
    unfolded_chi2 = np.array(unfolded_dict_chi2["unfolded"])
    stat_error_chi2 = unfolded_dict_chi2["stat_err"]
    sys_error_chi2 = unfolded_dict_chi2["sys_err"]
    unfolding_matrix_chi2 = unfolded_dict_chi2["unfolding_matrix"]
    num_iterations_chi2 = unfolded_dict_chi2["num_iterations"]
    ts_iter_chi2 = unfolded_dict_chi2["ts_iter"]
    refolded_chi2 = unfolder.refold(unfolded_chi2)
    print("unfolded: ", pprint.pformat(unfolded_chi2))
    print("stat_error: ", pprint.pformat(stat_error_chi2))
    print("sys_error: ", pprint.pformat(sys_error_chi2))
    print("refolded: ", pprint.pformat(refolded_chi2))
    print(f"unfolding_matrix: {pprint.pformat(unfolding_matrix_chi2)}")
    print(f"num_iterations: {pprint.pformat(num_iterations_chi2)}")
    print(f"ts_iter: {pprint.pformat(ts_iter_chi2)}")

    assert np.allclose(refolded_chi2, mock_data)
    
def test_unfolding_bf(unfolder, mock_data, mock_data_error):
    # mock_data = mock_data / np.sum(mock_data)
    # mock_data_error = mock_data_error / np.sum(mock_data)
    unfolded_dict_bf = unfolder.unfold(mock_data, mock_data_error, None, None, test_statistic='bf', test_statistic_stopping=0.01)
    unfolded_bf = np.array(unfolded_dict_bf["unfolded"])
    stat_error_bf = unfolded_dict_bf["stat_err"]
    sys_error_bf = unfolded_dict_bf["sys_err"]
    unfolding_matrix_bf = unfolded_dict_bf["unfolding_matrix"]
    num_iterations_bf = unfolded_dict_bf["num_iterations"]
    ts_iter_bf = unfolded_dict_bf["ts_iter"]
    refolded_bf = unfolder.refold(unfolded_bf)
    print("unfolded: ", pprint.pformat(unfolded_bf))
    print("stat_error: ", pprint.pformat(stat_error_bf))
    print("sys_error: ", pprint.pformat(sys_error_bf))
    print("refolded: ", pprint.pformat(refolded_bf))
    print(f"unfolding_matrix: {pprint.pformat(unfolding_matrix_bf)}")
    print(f"num_iterations: {pprint.pformat(num_iterations_bf)}")
    print(f"ts_iter: {pprint.pformat(ts_iter_bf)}")


    assert np.allclose(refolded_bf, mock_data)

def test_unfolding_rmd(unfolder, mock_data, mock_data_error):
    # mock_data = mock_data / np.sum(mock_data)
    # mock_data_error = mock_data_error / np.sum(mock_data)
    unfolded_dict_rmd = unfolder.unfold(mock_data, mock_data_error, None, None, test_statistic='rmd', test_statistic_stopping=0.01)
    unfolded_rmd = np.array(unfolded_dict_rmd["unfolded"])
    stat_error_rmd = unfolded_dict_rmd["stat_err"]
    sys_error_rmd = unfolded_dict_rmd["sys_err"]
    unfolding_matrix_rmd = unfolded_dict_rmd["unfolding_matrix"]
    num_iterations_rmd = unfolded_dict_rmd["num_iterations"]
    ts_iter_rmd = unfolded_dict_rmd["ts_iter"]
    refolded_rmd = unfolder.refold(unfolded_rmd)
    print("unfolded: ", pprint.pformat(unfolded_rmd))
    print("stat_error: ", pprint.pformat(stat_error_rmd))
    print("sys_error: ", pprint.pformat(sys_error_rmd))
    print("refolded: ", pprint.pformat(refolded_rmd))
    print(f"unfolding_matrix: {pprint.pformat(unfolding_matrix_rmd)}")
    print(f"num_iterations: {pprint.pformat(num_iterations_rmd)}")
    print(f"ts_iter: {pprint.pformat(ts_iter_rmd)}")


    assert np.allclose(refolded_rmd, mock_data)
