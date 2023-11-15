import pickle

def getParameterByTriggerAndHadronMomentumForParticleSpecies(analysisType, parameter_name, particle_species, dbCursor, with_reduced_chi2=False):
    # define the SQL query
    query = f"SELECT trigger_momentum_bin, associated_momentum_bin, {parameter_name}, {parameter_name}_error{', reduced_chi2' if with_reduced_chi2 else ''} FROM fit_parameters WHERE analysis_type = ? AND particle_species = ? ORDER BY trigger_momentum_bin, associated_momentum_bin;"
    # execute the query
    result = dbCursor.execute(query, (f"{analysisType.name}", f"{particle_species.name}",))
    # get the result
    result = result.fetchall()
    # convert the result to a list
    # return the result
    return result

def getParameters(analysisType, triggerJetMomentumBin, associatedHadronMomentumBin, particle_species, dbCursor):
    # define the SQL query
    query = f"SELECT background_level, v2, v3, v4, va2, va4, reduced_chi2 FROM fit_parameters WHERE analysis_type = ? AND trigger_momentum_bin = ? AND associated_momentum_bin = ? AND particle_species = ?;"
    # execute the query
    result = dbCursor.execute(query, (f"{analysisType.name}", f"{triggerJetMomentumBin.value}", f"{associatedHadronMomentumBin.value}", f"{particle_species.name}",))
    # get the result
    result = result.fetchone()
    # convert the result to a list
    # return the result
    return result

def getParameterErrors(analysisType, triggerJetMomentumBin, associatedHadronMomentumBin, particle_species, dbCursor):
    # define the SQL query
    query = f"SELECT background_level_error, v2_error, v3_error, v4_error, va2_error, va4_error, covariance_matrix, reduced_chi2 FROM fit_parameters WHERE analysis_type = ? AND trigger_momentum_bin = ? AND associated_momentum_bin = ? AND particle_species = ?;"
    # execute the query
    result = dbCursor.execute(query, (f"{analysisType.name}", f"{triggerJetMomentumBin.value}", f"{associatedHadronMomentumBin.value}", f"{particle_species.name}",))
    # get the result
    result = result.fetchone()
    # convert the result to a list
    # return the result
    # convert the covariance matrix from a blob to a numpy array using pickle
    covariance_matrix = pickle.loads(result[6])
    return result, covariance_matrix